import torch
from torch import nn
from einops import rearrange, repeat

from attention import SpatialTransformer, TemporalTransformer
from timm import create_model

               
                                                                          

class MST_ViT(nn.Module):
    def __init__(self, out_dim=2,  
                 swin_backbone=None, context_dim=9, dim=64, batch_size=64,  
                 dropout=0., emb_dropout=0.):    #num_grid=64
        super().__init__()

        #assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.batch_size = batch_size
        self.swin_backbone = swin_backbone  # Using Swin Transformer backbone
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))  #class token
        self.dropout = nn.Dropout(emb_dropout)
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        ) 


    #process a batch of input tensors x to deal with large batches of data and avoid out-of-memory on GPU
    def forward_features(self, x):
        # prevent the number of grids from being too large to cause out of memory
        B = x.shape[0]#Get original batch size
        #the internal batch size is used to control memory usage by limiting the number of items processed by the PVT backbone at once
        n = B // self.batch_size if B % self.batch_size == 0 else B // self.batch_size + 1  # the number of sub-batches
        x_hat = torch.empty(0).to(x.device)  #initializing x_hat as an empty tensor on the same device as x
        for i in range(n): #loop each sub-batch
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            x_tmp = x[start:end] #select current sub-batch
            x_hat_tmp = self.swin_backbone(x_tmp)  
            x_hat = torch.cat([x_hat, x_hat_tmp], dim=0)  #concatenating each sub-batch's output x_hat_tmp to x_hat in a memory-efficient manner
        return x_hat

    def forward(self, x):   
        b = x.shape[0]  # Extract batch size from input
        x = self.forward_features(x)
        cls_space_tokens = repeat(self.space_token, '() n d -> b n d', b=b)
        if x.dim() == 2:  # if x is [batch, feature_dim]
            x = x.unsqueeze(1)  # x becomes [batch, 1, feature_dim]
        x = torch.cat((cls_space_tokens, x), dim=1)         #dim=1  represents the sequence length dimension ([batch size, sequence length + 1, feature dimension])
                                                            #dim=2  concatenation along the feature dimension

        x = self.dropout(x)
        x = self.norm(x)
        #global average pooling over the time-step dimension
        x = x.mean(dim=1)                                                         
        x = self.mlp_head(x)
        
        return x 


if __name__ == "__main__":
    
    # x.shape = B, C, H, W 
    x = torch.randn((1, 3, 224, 224))  
    swin_model = create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=512)
    model = MST_ViT(out_dim=2, swin_backbone=swin_model, dim=512)
    z = model(x)
    print(z)
    print(z.shape)



