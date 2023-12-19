import torch
from torch import nn

from .attention import MHA, RelativeMHA
from torchvision.models import swin_v2_t, Swin_V2_T_Weights


class Swin(nn.Module):

    def __init__(self, is_trainable=True):
        super().__init__()

        swin = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        swin = nn.Sequential(*list(swin.children())[0])
        children = list(swin.children())
        self.swin = nn.ModuleList([nn.Sequential(*children[2*i:2*i+2]) for i in range(4)])
        if not is_trainable:
            for param in self.swin.parameters():
                param.requires_grad = False
        # self.swin.eval()

        self.embed_dim = children[0][0].out_channels
        self.embed_dims = [self.embed_dim*2**(i) for i in range(4)]

        # self.fpn = FeaturePyramidNetwork(
        #     self.embed_dims, self.embed_dim,
        #     # ExtraFPNBlock(),
        # )        

    def forward(self, x):
        return [x:=stage(x) for stage in self.swin]
        # xs = self.fpn({
        #     f'feat{i}': x.permute(0, 3, 1, 2)
        #     for i, x in enumerate(xs)
        # })

        # return xs
    