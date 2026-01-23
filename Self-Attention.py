# Self-Attention

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads):
        super(SelfAttention, self).__init__()
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.channels = channels
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels,channels),
            nn.GELU(),
            nn.Linear(channels,channels),
        )

    def forward(self,x):
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).transpose(1, 2).contiguous()
        x_ln = self.ln(x)
        attn_out, _ = self.mha(x_ln, x_ln, x_ln)
        x = attn_out + x
        x_ff = self.ff_self(x)
        x = x + x_ff
        return x.swapaxes(2, 1).view(B, C, H, W)