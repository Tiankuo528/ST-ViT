import torch
from torch import nn
from einops import rearrange, repeat
from timm import create_model  # Ensure timm is installed

class MMST_ViT(nn.Module):
    def __init__(self, out_dim=2, swin_model=None, dim=768, batch_size=64, depth=4, heads=3, pool='cls', dim_head=64, dropout=0., emb_dropout=0., scale_dim=4):
        super().__init__()
        assert pool in {'cls', 'mean'}, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.pool = pool
        self.batch_size = batch_size
        self.backbone = swin_model if swin_model else create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)  # Load Swin Transformer without classifier
        self.dropout = nn.Dropout(emb_dropout)
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * scale_dim, dropout=dropout),
            num_layers=depth
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )

    def forward_features(self, x):
        x = self.backbone(x)
        # If backbone outputs a flat feature map, adapt dimensions to add a sequence dimension
        if len(x.shape) == 2:  # Shape is [batch, features]
            x = rearrange(x, 'b (h w) -> b h w', h=1)  # Change 'h' according to desired sequence pieces
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        b, n, _ = x.shape  # n is the sequence length

        # Handle class token
        cls_tokens = self.space_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        x = self.transformer(x)

        # Pooling and classification
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # Choose pooling based on the 'pool' setting
        x = self.mlp_head(x)
        return x

if __name__ == "__main__":
    x = torch.randn((1, 3, 224, 224))  # B, C, H, W
    swin_model = create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
    model = MMST_ViT(out_dim=2, swin_model=swin_model, dim=768)
    z = model(x)
    print(z)
    print(z.shape)