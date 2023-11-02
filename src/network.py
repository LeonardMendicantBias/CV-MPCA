import math
import copy
from collections import OrderedDict

from typing import Dict, List, Tuple, Optional, Callable, Any

import numpy as np
from pysolotools.core.models import Frame, Capture

from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from torchvision.ops import FeaturePyramidNetwork, MLP
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock
# from torch.utils.data import ConcatDataset, DataLoader

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import lightning.pytorch as pl

import utils

import my_trainer


class CVMPCA(pl.LightningModule):

    def __init__(self,
        captures: List[Capture],
        capture_lookup: Dict,
        num_classes: int,
        num_layers: int,
        num_query: int,
        pc_range,
        # alpha,
    ):
        super().__init__()
        self.cameras = nn.ModuleDict({
            capture.id: utils.Camera.from_unity(capture)
            for capture in captures
        })
        self.capture_lookup = capture_lookup

        self.single_view = SingleView(is_trainable=True)
        embed_dim = self.single_view.embed_dim
        embed_dims = self.single_view.embed_dims

        self.query = nn.Embedding(num_query, embed_dim)
        self.to_anchor = nn.Linear(embed_dim, 3)

        self.decoder = Decoder(
            cameras=self.cameras,
            num_levels=len(embed_dims),
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_layers=len(embed_dims),
            pc_range=pc_range
        )

        self.loss = my_trainer.SetCriterion(
            num_ue=num_query,
            num_sbs=len(self.cameras), num_classes=num_classes,
            num_layers=num_layers, pc_range=pc_range,
        )

        self._init_weight()
        self._init_log_var()
        # self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))

    def _init_weight(self):
        nn.init.xavier_uniform_(self.query.weight)
        
        nn.init.xavier_uniform_(self.to_anchor.weight)
        nn.init.zeros_(self.to_anchor.bias)
    
    def forward(self, image_dicts: Dict[str, Dict[str, Tensor]]):
        keys = list(image_dicts.keys())  # camera ids
        B = image_dicts[keys[0]].size(0)

        visual_features = {
            key: self.single_view(image)
            for key, image in image_dicts.items()
        }

        query = self.query.weight.unsqueeze(0).expand(B, -1, -1)
        anchor = self.to_anchor(query).sigmoid()

        # decoder
        cat_list, position_list, los_list = self.decoder(
            query=query,
            value=visual_features,
            anchor=anchor
        )

        return cat_list, position_list, los_list

    def training_step(self, batch, batch_idx):
        image_dict, object_list = batch
        cat_list, position_list, los_list = self(image_dict)  # (B, L, n_query, C/3/S)
        # prediction_list = [(cat, pos, los) for cat, pos, los in zip(cat_list, position_list, los_list)]
        losses = self.loss(object_list, (cat_list, position_list, los_list))

        loss_cls = torch.mean(torch.stack([loss['cls'] for loss in losses]))
        loss_reg = torch.mean(torch.stack([loss['reg'] for loss in losses]))
        loss_reg_2 = torch.mean(torch.stack([loss['reg_2'] for loss in losses]))
        loss_los = torch.mean(torch.stack([loss['los'] for loss in losses]))
        # auc = torch.mean(torch.stack([loss['auc'] for loss in losses]))
        acc = np.mean([loss['acc'].cpu().numpy() for loss in losses])

        self.log_dict({
            "train_loss/cls": loss_cls,
            "train_loss/reg": loss_reg,
            "train_loss/reg_2": loss_reg_2,
            "train_loss/los": loss_los,
            # "train_loss/auc": auc,
            "train_loss/acc": acc,
        })

        return 1e-8*loss_cls + 1e-5*loss_los + loss_reg_2


    def _init_log_var(self):
        self._train_predictions = []
        self._train_labels = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000, eta_min=1e-6
        )
        return [optimizer], [scheduler]
        # return optimizer


class SingleView(nn.Module):

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

        self.fpn = FeaturePyramidNetwork(
            self.embed_dims, self.embed_dim,
            # ExtraFPNBlock(),
        )

        # TODO: use depth prediction
        # self.depth_predictor = nn.Sequential(
        #     nn.Conv2d(64, 1, kernel_size=1, stride=1),
        #     nn.Softmax()  # perform softmax on each spatial location
        # )

    def forward(self, x):
        xs = [x:=stage(x) for stage in self.swin]
        xs = self.fpn({
            f'feat{i}': x.permute(0, 3, 1, 2)
            for i, x in enumerate(xs)
        })

        return xs
    

class Decoder(nn.Module):

    def __init__(self,
        cameras: Dict[str, utils.Camera],
        num_classes: int,
        embed_dim: int,
        num_layers: int,
        num_levels: int,
        pc_range,
    ):
        super().__init__()
        self.decoders = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads=8, num_levels=num_levels, cameras=cameras, pc_range=pc_range)
            for _ in range(num_layers)
        ])
        
        _num_pred = num_layers
        _cls_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes)
        )
        self.cls_layers = nn.ModuleList([copy.deepcopy(_cls_net) for _ in range(_num_pred)])

        _pos_net = nn.Sequential(
            # nn.LayerNorm(2*embed_dim),
            nn.Linear(embed_dim, 2*embed_dim),
            nn.LeakyReLU(),
            nn.Linear(2*embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, 3)
        )
        self.pos_layers = nn.ModuleList([copy.deepcopy(_pos_net) for _ in range(_num_pred)])
        
        _los_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, len(cameras))
        )
        self.los_layers = nn.ModuleList([copy.deepcopy(_los_net) for _ in range(_num_pred)])

    def _init_weight(self):
        for layer in self.cls_layers + self.los_layers + self.pos_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(self.output_proj.weight)
                nn.init.zeros_(self.output_proj.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
    

    @staticmethod
    def _inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)
    
    def forward(self, query, value, anchor):
        cat_list, position_list, los_list = [], [], []
        for idx, (decoder, cls_layer, pos_layer, los_layer) in enumerate(zip(
            self.decoders, self.cls_layers, self.pos_layers, self.los_layers#, self.offsets
        )):
            visual_features = {
                key: list(feature.values())[idx] for key, feature in value.items()
            }

            query = decoder(query, visual_features, anchor)  # aggregate visual features into each query
            cat = cls_layer(query)
            offset = pos_layer(query)
            los = los_layer(query)
            # query = query + offset_layer(offset)

            anchor = self._inverse_sigmoid(anchor) + offset
            anchor = anchor.sigmoid()
            # query = query + anchor

            cat_list.append(cat)
            position_list.append(anchor)
            los_list.append(los)

        return torch.stack(cat_list, dim=1), torch.stack(position_list, dim=1), torch.stack(los_list, dim=1)
        # return cat_list, position_list, los_list


# class Block(nn.Module):
class DecoderBlock(nn.Module):

    def __init__(self,
        embed_dim: int,
        num_heads: int,
        num_levels: int,
        cameras: Dict[str, utils.Camera],
        pc_range,
    ):
        super().__init__()

        self.mha_norm = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)

        self.mhca_norm = nn.LayerNorm(embed_dim)
        self.mhca = MHCA(embed_dim, num_levels, pc_range, cameras)

        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, [2*embed_dim, embed_dim], activation_layer=nn.GELU, dropout=0.1)

        self._init_weight()
        
    def _init_weight(self):
        for layer in self.mlp.children():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                # if layer.bias:
                torch.nn.init.zeros_(layer.bias)

    def forward(self, query, value, anchor):
        x = query

        x = x + self.mhca(self.mhca_norm(x), value, anchor)

        x_ = self.mha_norm(x)
        o, _ = self.mha(x_, x_, x_)
        x = x + o

        x = x + self.mlp(self.mlp_norm(x))

        return x


class MHCA(nn.Module):

    def __init__(self,
        embed_dim: int,
        num_levels: int,
        pc_range,
        cameras: Dict[str, utils.Camera],
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

        self.register_buffer('pc_range',
            torch.tensor(pc_range, dtype=torch.float)
        )

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
