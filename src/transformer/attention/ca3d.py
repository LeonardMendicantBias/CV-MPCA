import torch
from torch import nn, Tensor

from .base import MHA

import torch.nn.functional as F


class MHCA(nn.Module):

    def __init__(self,
        embed_dim: int,
        num_levels: int,
        pc_range,
        cameras#: Dict[str, utils.Camera],
    ):
        super().__init__()
        self.cameras = cameras
        self.num_cams = len(cameras)
        self.num_levels = num_levels

        self.dropout = nn.Dropout(0.1)
        # the attention layer produces weights for each hierarchical level on each camera/view
        # self.attention_weights = nn.Linear(embed_dim, len(cameras)*num_levels)
        self.attention_weights = nn.Linear(embed_dim, len(cameras))
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.attention_weights.weight)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, query, value, anchor):
        B, T, _ = query.size()  # T: number of queries

        attention_weights = self.attention_weights(query).sigmoid().view(
            B, T, self.num_cams, 1 #self.num_levels, 1
        )

        xs = []
        for key, features in value.items():
            capture = self.cameras[key]
            # camera_features = []
            
            # for feature_idx, feature in enumerate(features.values()):
            _, D, H, W = features.shape
            # print('anchor', anchor.shape)
            anchor = anchor*(self.pc_range[:, 1] - self.pc_range[:, 0]) + self.pc_range[:, 0]
            pixels = capture(anchor.flatten(0, 1), (H, W, 1))

            bounding_x = (pixels[..., 0] < 0).logical_or(pixels[..., 0] > H)
            bounding_y = (pixels[..., 1] < 0).logical_or(pixels[..., 1] > W)
            bounding = torch.logical_or(bounding_x, bounding_y)
            bounding = bounding.view(B, T, 1)

            pixels[:, 0] /= H
            pixels[:, 1] /= W
            pixels = pixels.view(B, T, 1, 2)  # dummy dimension for spatial coordinate to use grid_sample

            sample_feature = F.grid_sample(features, pixels, align_corners=True)  # B, D, T, 1
            sample_feature = sample_feature.squeeze(-1).permute(0, 2, 1)  # B, T, D
            sample_feature = sample_feature * bounding.logical_not()
            # camera_features.append(sample_feature)
            xs.append(sample_feature)
            # xs.append(torch.stack(camera_features, dim=-2))
        xs = torch.stack(xs, dim=-2)
        
        output = xs*attention_weights
        output = output.sum(-2)#.sum(-2)#.sum(-1)
        output = self.output_proj(output)
        return self.dropout(output)