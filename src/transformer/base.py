import torch
from torch import nn

from .attention import MHA, RelativeMHA


class Block(nn.Module):
    '''
        A Universal Transformer-based Block
    '''
    def __init__(self, embed_dim, layer_list: nn.ModuleList):
        super().__init__()
        self.layer_list = layer_list
        self.norm_list = [nn.LayerNorm(embed_dim) for _ in layer_list]

    def forward(self, x):
        for layer, norm in zip(self.layer_list, self.norm_list):
            x_ = norm(x)
            _x = layer(x_)
            if isinstance(_x, tuple):
                x = x + _x[0]
            else: # isinstance(_x, Tensor)
                x = x + _x
        return x
