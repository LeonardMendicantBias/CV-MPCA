import torch
from torch import nn, Tensor

from .base import MHA


class RelativeMHA(MHA):

    def __init__(self,
        rel,
        *args, 
        **kwargs
    ):
        self.rel = rel
        super().__init__(*args, **kwargs)

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((self.num_heads, 1, 1))))

    def _get_relative_distance(self, s, t):
        coords_h, coords_w = torch.arange(s), torch.arange(t)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww

    def forward(self, x: Tensor, mask: Tensor=None):

        pass

