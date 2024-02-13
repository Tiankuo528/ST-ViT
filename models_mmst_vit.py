import torch
from torch import nn
from einops import rearrange, repeat

from attention import SpatialTransformer, TemporalTransformer

from models_pvt_simclr import PVTSimCLR
               
                                                                          

class MMST_ViT(nn.Module):
    def __init__(self, out_dim=2,  num_week=5,
                 pvt_backbone=None, context_dim=9, dim=64, batch_size=64, depth=4, heads=3, pool='cls', dim_head=64,
                 dropout=0., emb_dropout=0., scale_dim=4):    #num_grid=64
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.batch_size = batch_size
        self.pvt_backbone = pvt_backbone

        #self.proj_context = nn.Linear(num_year * num_long_term_seq * context_dim, num_short_term_seq * dim)

        #Create position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_week+1, dim))
        
        #Create class embedding
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.space_transformer = SpatialTransformer(dim, depth, heads, dim_head, mult=scale_dim, dropout=dropout)  

        #Create embedding dropout value
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool
        
        #Create temporal embedding
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        #Create Temporal Transformer Encoder
        self.temporal_transformer = TemporalTransformer(dim, depth, heads, dim_head, mult=scale_dim, dropout=dropout)
        
        #Create the Norm layer (LN)
        self.norm1 = nn.LayerNorm(dim)
        
        #Create classifier head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )

    #process a batch of input tensors x through a Pyramid Vision Transformer (PVT) backbone model to deal with large batches of data and avoid out-of-memory on GPU
    def forward_features(self, x):
        # Convert grayscale images to RGB by repeating the channel 3 times
        x = x.repeat(1, 1, 3, 1, 1)
        x = rearrange(x, 'b t c h w -> (b t) c h w') #reshape x into a flattened batch size combining original batch size and temporal dimension
        #ys = rearrange(ys, 'b t g n d -> (b t g) n d')

        # prevent the number of grids from being too large to cause out of memory
        B = x.shape[0]#Get original batch size
        #the internal batch size is used to control memory usage by limiting the number of items processed by the PVT backbone at once
        n = B // self.batch_size if B % self.batch_size == 0 else B // self.batch_size + 1  # the number of sub-batches

        x_hat = torch.empty(0).to(x.device)  #initializing x_hat as an empty tensor on the same device as x
        for i in range(n): #loop each sub-batch
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            x_tmp = x[start:end] #select current sub-batch
            x_hat_tmp = self.pvt_backbone(x_tmp) #processed by the PVT backbone
            #Pyramid Vision Transformer (PVT) backbone:PVT divides the image into a pyramid of resolutions or scales, processing each level with Transformers that adapt to the varying spatial dimensions
            # PVT processes images through multiple stages, each with a reduced spatial resolution but an increased feature dimension. This hierarchical structure enables the model to efficiently capture multi-scale features.
            # PVT can aggregate features at different levels of granularity by varying the size of the feature maps and the depth of the Transformer layers at different stages of the pyramid
            # PVT addresses issue of traditional Transformer models that suffer from quadratic computational complexity with respect to input size, by reducing the spatial resolution at higher pyramid levels, thus reducing the computational burden of self-attention mechanisms. 
            
            x_hat = torch.cat([x_hat, x_hat_tmp], dim=0)  #concatenating each sub-batch's output x_hat_tmp to x_hat in a memory-efficient manner

        return x_hat

    def forward(self, x):   
        
        b, t, _, _, _ = x.shape
        x = self.forward_features(x)
        
        x = rearrange(x, '(b t) d -> b t d', b=b, t=t)   #x is reshaped to (b, t, feature_dim)
         
        #cls_space_tokens = repeat(self.space_token.squeeze(0), '() d -> b t d', b=b, t=t)
        cls_space_tokens = self.space_token 
        x = torch.cat((cls_space_tokens, x), dim=1)         #dim=1  represents the sequence length dimension ([batch size, sequence length + 1, feature dimension])
                                                            #dim=2  concatenation along the feature dimension
        
        x += self.pos_embedding[:,:(t+1),:] 
        x = self.dropout(x)
        x = self.temporal_transformer(x)
        #global average pooling over the time-step dimension
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # dim=1:sequence length (time steps in a sequence), self.pool == 'mean' is applies the mean pooling operation described above
                                                               # Otherwise, it assumes that the pooling strategy is to use the first element of the sequence (e.g., the CLS token)
        x = self.mlp_head(x)
        
        return x 


if __name__ == "__main__":
    # x.shape = B, T, G, C, H, W     #B, T, C, H, W
    x = torch.randn((1, 5, 1, 224, 224))  
    
    # # ys.shape = B, T, G, N1, d
    # ys = torch.randn((1, 6, 10, 28, 9))
    # # yl.shape = B, T, N2, d
    # yl = torch.randn((1, 5, 12, 9))

    pvt = PVTSimCLR("pvt_tiny", out_dim=512, context_dim=9)
    model = MMST_ViT(out_dim=2, pvt_backbone=pvt, dim=512)  

    # print(model)

    z = model(x)
    print(z)
    print(z.shape)
