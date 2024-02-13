import models_pvt
from attention import MultiModalTransformer
from torch import nn


class PVTSimCLR(nn.Module):

    def __init__(self, base_model, out_dim=512, num_head=8, mm_depth=2, dropout=0., pretrained=True, gated_ff=True, context_dim=None):
        super(PVTSimCLR, self).__init__()

        self.context_dim = context_dim
        self.backbone = models_pvt.__dict__[base_model](pretrained=pretrained)
        num_ftrs = self.backbone.head.in_features

        self.proj = nn.Linear(num_ftrs, out_dim)

        self.proj_context = nn.Linear(context_dim, out_dim)

        if context_dim is not None:
            self.proj_context = nn.Linear(context_dim, out_dim)
            self.norm1 = nn.LayerNorm(context_dim)
            dim_head = out_dim // num_head
            self.mm_transformer = MultiModalTransformer(out_dim, mm_depth, num_head, dim_head, context_dim=out_dim, dropout=dropout)
        else:
            self.mm_transformer = None  # Or an alternative module that does not require context
        

    def forward(self, x, context=None):
        h = self.backbone.forward_features(x)  # shape = B, N, D
        h = h.squeeze()
        # project to targeted dim
        x = self.proj(h)
        
        if self.context_dim is not None and context is not None:
            context = self.norm1(context)
            context = self.proj_context(context)
            x = self.mm_transformer(x, context=context)
        else:
            # Handle the case when context_dim is not provided or context is None
            # e.g., bypass the multi-modal transformer or use an alternative approach
            pass

        # return the classification token
        return x[:, 0]
