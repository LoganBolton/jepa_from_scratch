import torch
import torch.nn as nn
from utils import *


class VitBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim))
        
    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        h = x + h
        h = h + self.mlp(self.norm2(h))
        return h

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, hidden_dim=512, num_heads=8, num_layers=6):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        
        self.blocks = nn.ModuleList([VitBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)

        self.patch_proj = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def patchify(self, x):
        x = self.patch_proj(x) # (b, c, h, w)
        x = x.flatten(2) # (b, c, hw)
        x = x.transpose(-1, -2) # (b, hw, c)
        return x 
    
    def forward(self, x, mask_indices=None):
        x = self.patchify(x)
        x = x + self.pos_embed # (b, num_patches, D)
        
        if mask_indices is not None:
            x = gather_tokens(x, mask_indices)
            
        for block in self.blocks:
            x = block(x)
            
        x = self.final_norm(x)
        return x