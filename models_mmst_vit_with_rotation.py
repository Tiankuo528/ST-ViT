import torch
from torch import nn
from einops import rearrange, repeat
from timm import create_model  # Assuming you have `timm` installed

from attention import SpatialTransformer, TemporalTransformer
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time=None, angle=None):
        device = time.device if time is not None else angle.device
        half_dim = self.dim // 2 // 2  # Divide by 4 because we need to split dimensions between time and angle, and sin and cos
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        if time is not None:
            time_embeddings = time[:, None] * embeddings[None, :]
            time_embeddings = torch.cat((time_embeddings.sin(), time_embeddings.cos()), dim=-1)
        
        if angle is not None:
            angle_embeddings = angle[:, None] * embeddings[None, :]
            angle_embeddings = torch.cat((angle_embeddings.sin(), angle_embeddings.cos()), dim=-1)

        if time is not None and angle is not None:
            # Ensure the dimensions concatenated do not exceed the total expected dimensions
            embeddings = torch.cat((time_embeddings, angle_embeddings), dim=-1)
            return embeddings[:, :self.dim]  # Make sure to cut off any excess dimensions
        elif time is not None:
            return time_embeddings
        else:
            return angle_embeddings


class MMST_ViT(nn.Module):
    def __init__(self, out_dim=2, num_week=5, num_rotation=260, swin_model=None, dim=768, batch_size=64, depth=4, heads=3, pool='cls', dim_head=64, dropout=0., emb_dropout=0., scale_dim=4):
        super().__init__()
        assert pool in {'cls', 'mean'}, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.batch_size = batch_size
        self.backbone = swin_model if swin_model else create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)  # Load Swin Transformer without classifier
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = TemporalTransformer(dim, depth, heads, dim_head, mult=scale_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))
        self.pos_embedding = SinusoidalPositionEmbeddings(768)

    def forward_features(self, x):
        x = rearrange(x, 'b t r c h w -> (b t r) c h w')
        B = x.shape[0]
        n = B // self.batch_size if B % self.batch_size == 0 else B // self.batch_size + 1
        x_hat = torch.empty(0).to(x.device)
        for i in range(n):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            x_tmp = x[start:end]
            x_hat_tmp = self.backbone(x_tmp)
            x_hat = torch.cat([x_hat, x_hat_tmp], dim=0)
        return x_hat


    def forward(self, x):   
        b, t, r, _, _, _ = x.shape
        x = self.forward_features(x)
        
        time_indices = torch.arange(t, device=x.device)
        rotation_indices = torch.arange(r, device=x.device)

        sinusoidal_embeddings_time = self.pos_embedding(time=time_indices)
        sinusoidal_embeddings_rotation = self.pos_embedding(angle=rotation_indices)

        x = rearrange(x, '(b t r) d -> b t r d', b=b, t=t, r=r)

        sinusoidal_embeddings_time = sinusoidal_embeddings_time.unsqueeze(0).unsqueeze(2).expand(b, t, r, -1)
        sinusoidal_embeddings_rotation = sinusoidal_embeddings_rotation.unsqueeze(0).unsqueeze(1).expand(b, t, r, -1)
        
        sinusoidal_embeddings = torch.cat((sinusoidal_embeddings_time, sinusoidal_embeddings_rotation), dim=-1)

        # print("x dimensions:", x.shape)
        # print("sinusoidal_embeddings dimensions:", sinusoidal_embeddings.shape)

        x += sinusoidal_embeddings
    

        # flatten t and r dimentions and add class token
        x = rearrange(x, 'b t r d -> b (t r) d')  #[b, t*r, dim]
        cls_space_tokens = self.space_token.expand(b, 1, -1)  # Expand space token across the batch
        x = torch.cat((cls_space_tokens, x), dim=1)         #dim=1  represents the sequence length dimension ([batch size, sequence length + 1, feature dimension])
                                                            #dim=2  concatenation along the feature dimension
          #[b, (t*r) + 1, dim] 
          
        x = self.dropout(x)
        x = self.temporal_transformer(x)
        
        #Pooling and classication 
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # dim=1:sequence length (time steps in a sequence), self.pool == 'mean' is applies the mean pooling operation described above
                                                               # Otherwise, it assumes that the pooling strategy is to use the first element of the sequence (e.g., the CLS token)
        x = self.mlp_head(x)
        
        #embeddings = self.some_embedding_layer(x) ##the embeddings that will be used with the SupConLoss
        
        return x  


if __name__ == "__main__":
    
    # x.shape = B, T, R, C, H, W
    x = torch.randn((1, 5, 10, 3, 224, 224))   #(1, 5, 260, 3, 224, 224)
    
    swin_model = create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)  # Initialize Swin without classifier
    model = MMST_ViT(out_dim=2, swin_model=swin_model, dim=768) #512

    z = model(x)
    print(z)  #tensor([[ 0.6211, -0.8780]], grad_fn=<AddmmBackward0>)  #probabilities assigned to each class for the sample
    print(z.shape) #torch.Size([1, 2])  #batch size, # of labels
