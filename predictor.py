import torch
import torch.nn as nn
from vit import ViT, VitBlock
from utils import *

class Predictor(nn.Module):
    def __init__(self, encoder_dim, num_patches, pred_dim=384, num_layers=3, num_heads=8):
        super().__init__()
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, pred_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        
        self.in_proj = nn.Linear(encoder_dim, pred_dim)
        self.out_proj = nn.Linear(pred_dim, encoder_dim)
        
        self.blocks = nn.ModuleList([VitBlock(pred_dim, num_heads=8) for _ in range(num_layers)])
        
        
    def forward(self, context_embeds, context_indices, target_indices):
        x = self.in_proj(context_embeds)
        x = x + gather_tokens(self.pos_embed, context_indices)
        
        mask_tokens = self.mask_token + gather_tokens(self.pos_embed, target_indices)
        x = torch.cat([x, mask_tokens], dim=1)

        for block in self.blocks:
            x = block(x)
            
        # only keep mask preds
        x = x[:, -target_indices.shape[1]:, :]
        
        return self.out_proj(x)