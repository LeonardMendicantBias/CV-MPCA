import math
import copy
# from collections import OrderedDict
from dataclasses import dataclass, field

from typing import Dict, List, Tuple, Optional, Callable, Any, Union

from pysolotools.core.models import Frame, Capture

import torch
from torch import nn, Tensor

from torchvision.ops import FeaturePyramidNetwork, MLP, sigmoid_focal_loss
from scipy.optimize import linear_sum_assignment

import torch.nn.functional as F
import lightning.pytorch as pl

import utils
import transformer
import reader


class VoxelMHA(transformer.MHA):
    '''
        Given a collection voxels, 
        aggregate the corresponding and coorelated features


        Multi-head attention for voxel features
        For each query/class, find the corresponding features from each camera to form the voxel features
        N voxels contain visual features and accumulated attention weights
        Select top K(*C) voxels that are likely to be the target class based on attention weights
        Aggregate the features of the selected voxels to form the class features
    '''
    def __init__(self, cameras, *args, **kwargs):
        self.cameras = cameras
        super().__init__(*args, **kwargs) 
        self.cameras = nn.ModuleDict({camera_key: camera for camera_key, camera in self.cameras.items()})
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(10)))

    def build_key(self):
        return nn.ModuleDict({
            camera_key: nn.Linear(self.embed_dim, self.embed_dim)
            for camera_key in self.cameras.keys()
        })
    
    def build_value(self):
        return nn.ModuleDict({
            camera_key: nn.Linear(self.embed_dim, self.embed_dim)
            for camera_key in self.cameras.keys()
        })
    
    def _calculate_logits(self, q: Tensor, k: Tensor):
        B, C, K, _, D = q.shape
        B, C, K, N, D = k.shape
        logit_scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).mT
        # print(q.shape, k.shape, attn.shape)
        return attn * logit_scale
    
    def _get_camera_features(self, voxels, features) -> Dict[str, Tensor]:
        camera_features = {}  # obtain the corresponding features (convert voxels into features)
        B, C, K, N = voxels.shape[:4]
        for camera_key in self.cameras.keys():
            camera, feature = self.cameras[camera_key], features[camera_key]  # (B, D, H, W)
            feature = feature.unsqueeze(1).unsqueeze(1).expand(-1, C, K, -1, -1, -1)  # (B, C, K, D, H, W)
            H, W = feature.shape[-2:]
            
            pixel_coor = camera(voxels, [H, W, 1])  # (B, C, K, N, 2)
            pixel_coor, feature = pixel_coor.flatten(0, 2), feature.flatten(0, 2)
            # print(voxels.shape, pixel_coor.shape, feature.shape)
            bounding = torch.logical_and(
                torch.logical_and(pixel_coor[..., 0] >= 0, pixel_coor[..., 0] <= H),
                torch.logical_and(pixel_coor[..., 1] >= 0, pixel_coor[..., 1] <= W)
            )  # (BCK, N)

            # for grid_sample compatibility. (B, N(*K), 1, 2)
            pixel_coor = pixel_coor.unsqueeze(-2)
            camera_feature = F.grid_sample(feature, pixel_coor, align_corners=True).squeeze(-1)
            camera_feature = camera_feature*bounding.unsqueeze(1)#.unsqueeze(0)  # (BCK, D, N)
            camera_feature = camera_feature.permute(0, 2, 1)  # 
            camera_feature = camera_feature.reshape(B, C, K, N, -1)

            camera_features[camera_key] = camera_feature
        return camera_features

    def forward(self,
        voxels: Tensor, tk: int,
        query: Tensor, key: Tensor=None, value: Tensor=None,
        mask: Tensor=None
    ):  # aggregate corresponding and correlated features from each camera
        '''
            Args:
                - Voxels: (B, N(*K), 3)
        '''
        if key is None: key = query
        if value is None: value = key
        B, C, K, N, _ = voxels.shape
        b_idx = torch.arange(B, device=voxels.device)#.unsqueeze(1)

        _query = self.query(query)  # (B, C, K, D)

        # (B, C, K, N, 3) and Dict[str, (B, C, K, D, H, W)] -> Dict[str, (B, C, K, N, D)]
        camera_features = self._get_camera_features(voxels, value)

        # Dict[str, (B, C, K, N, D)]
        key = {cam_key: self.key[cam_key](item) for cam_key, item in camera_features.items()}
        value = {cam_key: self.value[cam_key](item) for cam_key, item in camera_features.items()}

        # voxel-class correlation
        logits = {
            cam_key: self._calculate_logits(_query.unsqueeze(-2), item).squeeze(-2)
            for cam_key, item in key.items()
        }  # (B, C, K, N)
        scores = {cam_key: self._calculate_attention(item, mask) for cam_key, item in logits.items()}
        scores = {cam_key: self.attention_dropout(item) for cam_key, item in scores.items()}
        voxel_scores = torch.stack([score for score in scores.values()], dim=-1).sum(-1)

        _, flatten_voxel_idx = voxel_scores.flatten(-2, -1).topk(tk, dim=-1)  # (B, C, K, N)
        space_idx, voxel_idx = flatten_voxel_idx // N, flatten_voxel_idx % N
        
        # for (cam_key, s), (_, v) in zip(scores.items(), value.items()):
        #     print(cam_key, s.shape, v.shape, space_idx.shape, voxel_idx.shape)
        #     print(
        #         (s.gather(
        #             2, space_idx.unsqueeze(-1).expand(-1, -1, -1, N)  # choose all voxels of the top spaces
        #         ).gather(
        #             -1, voxel_idx.unsqueeze(-2)#.expand(-1, -1, K, -1)  # choose top voxels in the top spaces
        #         ).squeeze(-2)[..., None] * v.gather(
        #             2, space_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, N, v.shape[-1])
        #         ).gather(
        #             -2, voxel_idx.unsqueeze(-2).unsqueeze(-1).expand(-1, -1, -1, -1, v.shape[-1])
        #         ).squeeze(-3)).shape
        #     )
        #     break

        # # select the corresponding features of the top ranked voxels
        o = {
            # cam_key: s[b_idx, cls_idx, voxel_idx][..., None] + v[b_idx, voxel_idx]
            cam_key: s.gather(
                2, space_idx.unsqueeze(-1).expand(-1, -1, -1, N)  # choose all voxels of the top spaces
            ).gather(
                -1, voxel_idx.unsqueeze(-2)#.expand(-1, -1, K, -1)  # choose top voxels in the top spaces
            ).squeeze(-2)[..., None] * v.gather(
                2, space_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, N, v.shape[-1])
            ).gather(
                -2, voxel_idx.unsqueeze(-2).unsqueeze(-1).expand(-1, -1, -1, -1, v.shape[-1])
            ).squeeze(-3)
            for (cam_key, s), (_, v) in zip(scores.items(), value.items())
        }
        
        # aggregate the appropriate features
        o = torch.stack([item for item in o.values()], dim=1).sum(1)  # (B, C, N)

        z = self.linear(o)
        return self.dropout(z), (space_idx, voxel_idx)

class Block(nn.Module):

    def __init__(self,
        cameras,
        embed_dim, n_heads
    ):
        super().__init__()

        self.mhca_q_norm = nn.LayerNorm(embed_dim)
        # self.mhca_kv_norm = nn.LayerNorm(embed_dim)
        self.mhca = VoxelMHA(
            cameras=cameras,
            embed_dim=embed_dim, num_heads=n_heads,
            attention_dropout=0.1, dropout=0.1,
        )

        # self.mha_norm = nn.LayerNorm(embed_dim)
        # self.mha = nn.MultiheadAttention(embed_dim, n_heads, dropout=0.1, batch_first=True)

        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, [2*embed_dim, embed_dim], activation_layer=nn.GELU, dropout=0.1)

    def forward(self, voxels: Tensor, tk: int, query, value):
        _x = query  # (B, C, K, D)
        B, _, _, D = _x.shape
        x, (space_idx, voxel_idx) = self.mhca(
            voxels, tk,
            self.mhca_q_norm(query),  # 
            value
        )
        # cls_idx, _ = indices
        # print(_x.shape, space_idx.shape, voxel_idx.shape)
        x = x + _x.gather(
            2, space_idx.unsqueeze(-1).expand(-1, -1, -1, 96)
        )

        # x_ = self.mha_norm(x)
        # o, _ = self.mha(x_, x_, x_)
        # x = x_ + o

        x = x + self.mlp(self.mlp_norm(x))

        return x, (space_idx, voxel_idx)

class CVMPCA(pl.LightningModule):
# class CVMPCA(nn.Module):

    def __init__(self,
        captures: List[Capture],
        n_classes: int,
        spaces: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        voxel_size: Tuple[float, float, float],
        ratio: Union[int, List[int]]
    ):
        super().__init__()
        self.n_classes = n_classes
        # self.spaces = spaces
        self.register_buffer('spaces', torch.tensor(spaces))
        self.register_buffer('voxel_size', torch.tensor(voxel_size))

        _cameras = {
            capture.id: utils.Camera.from_unity(capture)
            for capture in captures
        }
        self.n_cameras = len(_cameras)
        self.cameras = _cameras

        self.swin = transformer.Swin(is_trainable=True)
        embed_dim = self.swin.embed_dim
        embed_dims = self.swin.embed_dims
        self.n_layers = len(embed_dims)
        if isinstance(ratio, int):
            ratio = [ratio]*self.n_layers
        assert len(ratio) == self.n_layers, "The length of ratio must be equal to the number of layers"
        self.ratio = ratio

        self.fpn = FeaturePyramidNetwork(embed_dims, embed_dim)

        self.decoder = nn.ModuleList([
            Block(_cameras, embed_dim, 12)
            for _ in range(self.n_layers)
        ])

        self._output_dim = n_classes + 3 + len(_cameras)
        _head = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim),
            nn.GELU(),
            nn.Linear(2*embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_classes + 3 + len(_cameras))
        )

        self.heads = nn.ModuleList([_head for _ in range(self.n_layers)])

        self.query = nn.Embedding(n_classes, embed_dim)

        self.register_buffer(
            "next_voxel_size",
            torch.stack([self.voxel_size/self.ratio[i]**i for i in range(self.n_layers)])
        )

    def _init_weight(self):
        nn.init.xavier_uniform_(self.query.weight)
        for layer in self.cls_layers + self.los_layers + self.pos_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(self.output_proj.weight)
                nn.init.zeros_(self.output_proj.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def _single_view(self, images: Tensor) -> Dict[str, Dict[str, Tensor]]:
        return self.fpn({
            f'feat{i}': x.permute(0, 3, 1, 2)
            for i, x in enumerate(self.swin(images))
        })

    @staticmethod
    def _voxelize(space, voxel_size):
        X = torch.arange(space[0][0], space[0][1] + voxel_size[0]/4, voxel_size[0], device=voxel_size.device)
        Y = torch.arange(space[1][0], space[1][1] + voxel_size[1]/4, voxel_size[1], device=voxel_size.device)
        Z = torch.arange(space[2][0], space[2][1] + voxel_size[2]/4, voxel_size[2], device=voxel_size.device)
        
        grid_x, grid_y, grid_z = torch.meshgrid(X, Y, Z, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)

    def forward(self, top_k, image_dict: Dict[str, Dict[str, Tensor]]):
        keys = list(image_dict.keys())  # camera ids
        B = image_dict[keys[0]].size(0)

        visual_features = {
            key: self._single_view(images)
            for key, images in image_dict.items()
        }

        x = self.query.weight.unsqueeze(0).unsqueeze(-2).expand(B, -1, -1, -1)
        # decoder
        output_list, voxel_list = [], []
        spaces = self.spaces.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, self.n_classes, -1, -1, -1)
        for i, (decoder, head) in enumerate(zip(self.decoder, self.heads)):
            next_voxel_size = self.voxel_size/self.ratio[i]**i
            
            voxels = torch.stack([
                torch.stack([
                    torch.stack([self._voxelize(s, next_voxel_size) for s in sp])
                    for sp in space
                ], dim=0)
                for space in spaces
            ], dim=0)
            B, C, K, N, _ = voxels.shape

            x, (space_idx, voxel_idx) = decoder(
                voxels + next_voxel_size/2, top_k,
                x, {key: list(features.values())[~i] for key, features in visual_features.items()}
            )
            
            top_voxels = voxels.gather(
                2, space_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, N, 3)
            ).gather(
                3, voxel_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 3)
            ).squeeze(-2)

            spaces = torch.stack([top_voxels, top_voxels + next_voxel_size], dim=-1)

            output = head(x)

            voxel_list.append(top_voxels + next_voxel_size/2)
            output_list.append(output)

        return torch.stack(output_list, dim=1), torch.stack(voxel_list, dim=1)
    
    @torch.no_grad()
    def _match(self, preds, labels):  # (B, L, P, C) and (B, G, C)
        (outputs, voxel_list) = preds
        label_cls, label_pos, label_link = labels.category, labels.position, labels.los
        label_cls_ = label_cls.argmax(-1)

        keys = outputs.keys()
        B, L, C, P, _ = outputs[keys[0]].shape
        G = label_cls.shape[1]
        _device = outputs.device

        b_ids = torch.arange(B, device=_device)
        pred_ids = torch.empty((B, L, C, G), dtype=torch.long, device=_device)
        label_ids = torch.empty((B, L, C, G), dtype=torch.long, device=_device)

        for l in range(L):
            # pred_cls = outputs[:, l, ..., :self.n_classes].log_softmax(-1)
            # pred_pos = outputs[:, l, ..., self.n_classes:self.n_classes+3]#.sigmoid()
            # pred_link = outputs[:, l, ..., -self.n_cameras:]
            pred_cls = outputs['classification'][:, l].log_softmax(-1)
            pred_pos = outputs['position'][:, l]#.sigmoid()
            pred_link = outputs['link'][:, l]
            pred_voxel = voxel_list[:, l]
            
            selected_pred_cls = pred_cls[b_ids.unsqueeze(-1), label_cls_, :, label_cls_]
            selected_pred_pos = pred_pos[b_ids.unsqueeze(-1), label_cls_]
            selected_voxel = pred_voxel[b_ids.unsqueeze(-1), label_cls_]
            selected_pred_link = pred_link[b_ids.unsqueeze(-1), label_cls_]

            # loss_pos = F.mse_loss(
            #     # selected_pred_pos*self.next_voxel_size.unsqueeze(1).unsqueeze(1) + selected_voxel,
            #     selected_pred_pos + selected_voxel,
            #     label_pos.unsqueeze(-2).expand(-1, -1, P, -1),
            #     reduction='none'
            # ).mean(-1)
            # loss_link = sigmoid_focal_loss(
            #     selected_pred_link, label_link.unsqueeze(-2).expand(-1, -1, P, -1),
            #     alpha=0.75, reduction='none'
            # ).mean(-1)

            C = -selected_pred_cls# + loss_link + loss_pos
            for b in range(B):
                label_indices, pred_indices = linear_sum_assignment(C[b, ...].float().cpu().numpy())

                pred_ids[b, l] = torch.LongTensor(pred_indices).to(_device)
                label_ids[b, l] = torch.LongTensor(label_indices).to(_device)

        return (pred_ids, label_ids)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000, eta_min=1e-6
        )
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        image_dict, object_list = batch
        
        num_query = 50
        
        output_list, voxel_list = self(num_query, image_dict)

        _device = output_list.device
        B, L, C, P, _ = output_list.shape
        b_ids = torch.arange(B, device=_device)[..., None, None, None]
        l_ids = torch.arange(L, device=_device)[None, ..., None, None]
        c_ids = torch.arange(C, device=_device)[None, None, ..., None]

        pred_ids, label_ids = self._match((output_list, voxel_list), object_list)
        
        pred_cls = output_list[..., :self.n_classes]
        pred_pos = output_list[..., self.n_classes:self.n_classes+3]#.sigmoid()
        pred_link = output_list[..., -self.n_cameras:]


        class_label = torch.zeros_like(pred_cls)
        class_label[b_ids, l_ids, c_ids, pred_ids] = 1
        class_label[..., 0] = 1 # every thing is a background
        # only a few have non-background labels
        class_label[b_ids, l_ids, c_ids, label_ids] = object_list.category.to(pred_cls.dtype).unsqueeze(1).unsqueeze(1)

        cross_entropy = F.cross_entropy(
            pred_cls.flatten(0, -2), class_label.flatten(0, -2).argmax(-1),
            label_smoothing=0.1
        )
        
        # pos_loss = F.l1_loss(
        #     # pred_pos[b_ids, l_ids, c_ids, label_ids]*self.next_voxel_size.unsqueeze(1).unsqueeze(1).unsqueeze(0) +
        #     # voxel_list[b_ids, l_ids, c_ids, label_ids],
        #     pred_pos[b_ids, l_ids, c_ids, label_ids] + voxel_list[b_ids, l_ids, c_ids, label_ids],
        #     object_list.position.unsqueeze(1).unsqueeze(1).expand(-1, L, C, -1, -1)
        # )

        # los_link_loss = F.mse_loss(
        #     pred_link[b_ids, l_ids, c_ids, label_ids],
        #     object_list.los.unsqueeze(1).unsqueeze(1).expand(-1, L, C, -1, -1)
        # )
        
        loss = cross_entropy# + los_link_loss + pos_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_ce", cross_entropy, prog_bar=True)
        # self.log("train_pos", pos_loss, prog_bar=True)
        # self.log("train_link", los_link_loss, prog_bar=True)
        return loss
