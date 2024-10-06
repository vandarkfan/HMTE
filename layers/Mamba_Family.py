import torch
import torch.nn as nn
import torch.nn.functional as F

class AM_Layer(nn.Module):
    def __init__(self, self_attention, mamba, d_model, dropout):
        super(AM_Layer, self).__init__()
        self.self_attention = self_attention
        self.mamba = mamba
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask=None):
        out, attn = self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )
        x = x + self.dropout(out)
        x = self.norm1(x)

        x = x + self.mamba(x)
        x = self.norm2(x)

        return x, attn


class Mamba_Layer(nn.Module):
    def __init__(self, mamba, d_model):
        super(Mamba_Layer, self).__init__()
        self.mamba = mamba
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.mamba(x)
        x = self.norm(x)

        return x