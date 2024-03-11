import math
import copy
# from collections import OrderedDict
from dataclasses import dataclass, field

from typing import Dict, List, Tuple, Optional, Callable, Any, Union, AnyStr, Generic

import numpy as np
from pysolotools.core.models import Frame, Capture

import torch
from torch import nn, Tensor

from torchvision.ops import FeaturePyramidNetwork, MLP, sigmoid_focal_loss, StochasticDepth
from scipy.optimize import linear_sum_assignment

import torch.nn.functional as F
import lightning.pytorch as pl

import utils
import transformer
import reader


class DeformableMHA(nn.Module):

    def __init__(self,
        cameras: Dict[str, utils.Camera],
        embed_dim,
        n_heads, n_points, #n_levels,
        attention_dropout, dropout,
    ):
        super().__init__()
        self.cameras, self.n_cameras = cameras, len(cameras)
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_points = n_points
        # self.n_levels = n_levels
        self._head_dim = embed_dim // n_heads

        # utilize the same layer for all cameras since the inputs are different due to ray_pe
        self.sampling_offsets = nn.Linear(embed_dim, 2*n_heads*n_points)
        
        self.query = nn.Linear(embed_dim, n_heads*self._head_dim)
        self.key = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.output_dropout = nn.Dropout(dropout)

        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        # divide the 360 degree into `n_heads` parts
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # 7 points in each head have the same value
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 2).repeat(1, self.n_points, 1)
        # shift each point
        for i in range(self.n_points):
            # grid_init[:, :, i, :] *= i + 1
            grid_init[:, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def _headify(self, x: Tensor):
        return x.view(*x.shape[:2], self.n_heads, self._head_dim).transpose(1, 2)
    
    def _project(self, points, camera, feature):
        H, W = feature.shape[-2:]

        pixel_coor = camera(points, [H, W, 1])  # (B, N, 2)
        pixel_coor[..., 0] = 2*pixel_coor[..., 0] / H - 1
        pixel_coor[..., 1] = 2*pixel_coor[..., 1] / W - 1
        pixel_coor = pixel_coor.unsqueeze(-2)

        camera_feature = F.grid_sample(feature, pixel_coor, align_corners=True).squeeze(-1)  # (B, D, N)
        return camera_feature.permute(0, 2, 1)  # (B, N, D)
    
    def forward(self, 
        query: Tensor,  # (B, K, D)
        points: Tensor,  # (B, K, 3)
        features: Dict[str, Tensor],  # features per camera (B, D, H, W)
    ):
        B, K, _ = query.shape
        
        camera_features = torch.stack([
            self._project(points, self.cameras[cam_key], features[cam_key])  # (B, K, D)
            for cam_key in self.cameras.keys()
        ], dim=2)  # (B, K, V, D)
        query_ = query.unsqueeze(2) + camera_features  # (B, K, V, D)
        
        offsets = self.sampling_offsets(query_).view(B, K, len(self.cameras), self.n_heads, self.n_points, 2)

        k = {
            cam_key: self.key(features[cam_key]).view(B, self.n_heads, self._head_dim, *features[cam_key].shape[-2:])
            for cam_key in self.cameras.keys()
        }  # (B, head, D_, H, W)
        v = {
            cam_key: self.value(features[cam_key]).view(B, self.n_heads, self._head_dim, *features[cam_key].shape[-2:])
            for cam_key in self.cameras.keys()
        }  # (B, head, D_, H, W)

        keys, values = [], []
        for cam_idx, cam_key in enumerate(self.cameras.keys()):
            k_, v_ = k[cam_key], v[cam_key]
            H, W = k[cam_key].shape[-2:]
            pixel_coor = self.cameras[cam_key](points, [H, W, 1])
            offset = offsets[:, :, cam_idx] / torch.tensor([H, W], device=offsets.device)
            
            pixel_coor = pixel_coor.unsqueeze(-2).unsqueeze(-2) + offset  # (B, K, head, P, 2)
            pixel_coor[..., 0] = 2*(pixel_coor[..., 0] / H) - 1 # (H - 1)) - 1
            pixel_coor[..., 1] = 2*(pixel_coor[..., 1] / W) - 1 # (W - 1)) - 1

            keys.append(
                F.grid_sample(
                    k_.flatten(0, 1),  # (B*head, D_, H, W)
                    pixel_coor.transpose(1, 2).flatten(0, 1),  # (B*head, K, P, 2)
                    align_corners=True
                ).permute(0, 2, 3, 1).view(B, self.n_heads, K, self.n_points, self._head_dim)  # (B*head, D_, K, P)
            )  # (B*head, D', K, P) -> (B, head, K, P, D')
            values.append(
                F.grid_sample(
                    v_.flatten(0, 1),  # (B*head, D_, H, W)
                    pixel_coor.transpose(1, 2).flatten(0, 1),  # (B*head, K, 2)
                    align_corners=True
                ).permute(0, 2, 3, 1).view(B, self.n_heads, K, self.n_points, self._head_dim)  # (B*head, D_, K, P)
            )
        keys = torch.stack(keys, dim=3).flatten(3, 4)  #  (B, head, K, V*P, D_)
        values = torch.stack(values, dim=3).flatten(3, 4)  #  (B, head, K, V*P, D_)
        q = self.query(query).view(B, K, self.n_heads, self._head_dim).transpose(1, 2)  # (B, head, K, D_)
        # print(q.shape, keys.shape, values.shape)
        
        logits = (q.unsqueeze(-2) @ keys.mT) / math.sqrt(self._head_dim)
        logits = logits.squeeze(-2)
        # print('logits', logits.shape)
        attention_scores = logits.softmax(dim=-1)
        attention_scores = self.attention_dropout(attention_scores)
        # print('attention_scores', attention_scores.shape, values.shape)

        weighted = attention_scores.unsqueeze(-1) * values
        weighted = weighted.sum(-2)
        weighted = weighted.view(B, K, self.embed_dim)
        # print('weighted', weighted.shape)

        return self.output_dropout(self.output_proj(weighted))


class Block(nn.Module):

    def __init__(self,
        cameras, 
        space, voxel_size,
        embed_dim, n_heads,
        attention_dropout, dropout, stochastic_depth_prob=0.1
    ):
        super().__init__()
        self.register_buffer('space', torch.tensor(space))
        self.register_buffer('voxel_size', torch.tensor(voxel_size))

        self.proj_attn_norm = nn.LayerNorm(embed_dim)
        # self.mhca_kv_norm = nn.LayerNorm(embed_dim)
        self.proj_attn = DeformableMHA(
            cameras=cameras, embed_dim=embed_dim,
            n_heads=n_heads, n_points=8, #n_levels=4,
            attention_dropout=attention_dropout, dropout=dropout,
        )

        self.mha_norm = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)

        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, [4*embed_dim, embed_dim], activation_layer=nn.GELU, dropout=dropout)
        
        # self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row") 1- (1-l/L)*1 + l/L*stochastic_depth_prob
        # self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.stochastic_depth = nn.Identity()
        # dropout is included in each layer

    def forward(self, query, points: Tensor, value):
        # x = query + self.stochastic_depth(self.proj_attn(self.proj_attn_norm(query), points.detach(), value))
        
        # _x = x
        # x_ = self.mha_norm(x)
        # x, _ = self.mha(x_, x_, x_)
        # x = _x + self.stochastic_depth(x)

        x_ = self.mha_norm(query)
        x, _ = self.mha(x_, x_, x_)
        x = query + self.stochastic_depth(x)

        x = x + self.stochastic_depth(self.proj_attn(self.proj_attn_norm(x), points.detach(), value))

        x = x + self.stochastic_depth(self.mlp(self.mlp_norm(x)))

        return x


class CVCA(pl.LightningModule):
    
    def __init__(self,
        captures: List[Capture],
        n_classes: int,
        space: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        voxel_size: Tuple[float, float, float],
        ratio: Union[int, List[int]],
        target_ids: int
    ):
        super().__init__()
        self.n_classes = n_classes
        self.register_buffer('space', torch.tensor(space))
        self.register_buffer('voxel_size', torch.tensor(voxel_size))
        self.register_buffer('voxels', self._voxelize())

        _cameras = {capture.id: utils.Camera.from_unity(capture) for capture in captures}
        self.n_cameras = len(_cameras)
        self.cameras = nn.ModuleDict(_cameras)

        self.backbone = None
        self.mhca = None
        self.ca = None

    def _voxelize(self):
        space, voxel_size = self.space, self.voxel_size
        X = torch.arange(space[0][0], space[0][1] + voxel_size[0]/4, voxel_size[0]) #, device=voxel_size.device)
        Y = torch.arange(space[1][0], space[1][1] + voxel_size[1]/4, voxel_size[1]) #, device=voxel_size.device)
        Z = torch.arange(space[2][0], space[2][1] + voxel_size[2]/4, voxel_size[2]) #, device=voxel_size.device)
        
        grid_x, grid_y, grid_z = torch.meshgrid(X, Y, Z, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
    
    def forward(self, image_dicts: Dict[str, Dict[str, Tensor]]):
        img_features = {cam: self.backbone(features) for cam, features in image_dicts.items()}

        return 


class CVMPCA(pl.LightningModule):
# class CVMPCA(nn.Module):

    def __init__(self,
        captures: List[Capture],
        n_classes: int,
        space: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        voxel_size: Tuple[float, float, float],
        ratio: Union[int, List[int]],
        target_ids: int
    ):
        super().__init__()
        self.n_classes = n_classes
        self.register_buffer('space', torch.tensor(space))
        self.register_buffer('voxel_size', torch.tensor(voxel_size))
        self.register_buffer('voxels', self._voxelize())
        self.target_ids = target_ids

        _cameras = {
            capture.id: utils.Camera.from_unity(capture)
            for capture in captures
        }
        self.n_cameras = len(_cameras)
        self.cameras = nn.ModuleDict(_cameras)

        embed_dim = 96
        embed_dims = [embed_dim*2**(i) for i in range(4)]
        self.embed_dim = embed_dim

        self.n_layers = 3
        if isinstance(ratio, int): ratio = [ratio]*self.n_layers
        
        self.ratio = ratio

        self.fpn = FeaturePyramidNetwork(embed_dims, self.embed_dim)
        self.ray_pe = nn.Conv2d(3, self.embed_dim, kernel_size=1, stride=1, padding=0)
        self.pos_pe = nn.Linear(3, self.embed_dim)

        self.embedding = nn.Linear(1+self.embed_dim, self.embed_dim)

        _cls_head = nn.Linear(self.embed_dim, n_classes, bias=False)
        _offset_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            # nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, 3)
        )
        _ca_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4*embed_dim, len(self.cameras))
        )
        
        self.blocks = nn.ModuleList([
            Block(
                cameras=_cameras, space=space, voxel_size=voxel_size,
                embed_dim=self.embed_dim, n_heads=1,
                attention_dropout=0.1, dropout=0.1,
            )
            for _ in range(self.n_layers)
        ])
        self.heads = nn.ModuleList([
            nn.ModuleDict({
                "logits": copy.deepcopy(_cls_head),
                "offset": copy.deepcopy(_offset_head),
                "ca": copy.deepcopy(_ca_head),
            }) for _ in range(self.n_layers+1)
        ])

        empty_weight = torch.ones(self.n_classes)
        empty_weight[0] = 0.1
        self.register_buffer("empty_weight", empty_weight)

        nn.init.xavier_uniform_(self.embedding.weight.data)
        nn.init.constant_(self.embedding.bias.data, 0.)
        for head_dict in self.heads:
            for layer in head_dict['offset'].children():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight.data)
                    nn.init.constant_(layer.bias.data, 0.)

        self._n_steps = 0
    
    def _voxelize(self):
        space, voxel_size = self.space, self.voxel_size
        X = torch.arange(space[0][0], space[0][1] + voxel_size[0]/4, voxel_size[0]) #, device=voxel_size.device)
        Y = torch.arange(space[1][0], space[1][1] + voxel_size[1]/4, voxel_size[1]) #, device=voxel_size.device)
        Z = torch.arange(space[2][0], space[2][1] + voxel_size[2]/4, voxel_size[2]) #, device=voxel_size.device)
        
        grid_x, grid_y, grid_z = torch.meshgrid(X, Y, Z, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
    
    def _get_voxel_features(self, voxels, features) -> Dict[str, Tensor]:
        camera_features = {}  # obtain the corresponding features (convert voxels into features)
        
        for camera_key in self.cameras.keys():
            feature_dict = features[camera_key]
            camera = self.cameras[camera_key]
            feature = list(feature_dict.values())[0]  # (B, D, H, W)
            B, _, H, W = feature.shape
            
            # project voxel coordinate onto the camera
            pixel_coor = camera(voxels, [H, W, 1])  # (N, 2)
            pixel_coor[..., 0] = 2*(pixel_coor[..., 0] / (H - 1)) - 1
            pixel_coor[..., 1] = 2*(pixel_coor[..., 1] / (W - 1)) - 1

            # for grid_sample compatibility. (B, N, 1, 2)
            pixel_coor = pixel_coor.unsqueeze(0).repeat(B, 1, 1).unsqueeze(-2)
            camera_feature = F.grid_sample(
                feature, pixel_coor, align_corners=True,
                mode='bilinear', padding_mode='zeros'
            ).squeeze(-1)  # B, D, N
            camera_features[camera_key] = camera_feature.permute(0, 2, 1)

        return torch.stack([item for item in camera_features.values()], dim=-1).sum(-1)

    @staticmethod
    def inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1/x2)
    
    def _process_view(self, feature_dict: Tensor) -> Dict[str, Dict[str, Tensor]]:
        return self.fpn(feature_dict)
    
    @staticmethod
    def create_grid(h, w, device=None):
        x = torch.arange(0, w, dtype=torch.float, device=device)
        y = torch.arange(0, h, dtype=torch.float, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        return torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones(h*w, device=device)], dim=-1)
    
    def forward(self, ue_list: List[Dict], topk: int, features: List[Dict[str, Tensor]]):
        visual_features = {
            camera_key: self._process_view(feature_dict)
            for camera_key, feature_dict in features.items()
        }
        
        for key, feature_dict in visual_features.items():
            pos_pe = self.pos_pe(self.cameras[key].translation)[None, ..., None, None]
            for key_, feature in feature_dict.items():
                B, D, H, W = feature.shape
                grid = self.create_grid(H, W, device=feature.device)
                rays = self.cameras[key].pix2ray(grid, [H, W, 1])
                rays = rays.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
                
                visual_features[key][key_] = self.ray_pe(rays) + feature# + pos_pe

        # aggregate visual features for each voxel
        voxels_ = self.voxels + self.voxel_size/2  # (N, 3)
        voxel_features = self._get_voxel_features(voxels_, visual_features)  # (B, N, D)
        
        # classify(/offset/ca) each voxel
        output_dict = {key: head(voxel_features) for key, head in self.heads[0].items()}
        coor = voxels_ + output_dict['offset'].tanh()*self.voxel_size*2
        output_dict['position'] = coor
        output_list = [output_dict]

        # TODO: support multiple target ids
        # choose top K of target ids only (e.g., phone)
        logits = output_dict['logits']  # (B, N, C)
        B = logits.shape[0]
        # _, indices = logits.view(B, -1).topk(topk, dim=-1)
        _, indices = logits[..., self.target_ids].view(B, -1).topk(topk, dim=-1)
        top_proposals_index = indices // self.n_classes  # (B, K) which voxel
        top_proposals_class = indices % self.n_classes   # (B, K) which class

        b_ids = torch.arange(B)
        coor = coor[b_ids[..., None], top_proposals_index]  # (B, K, 3)
        selected_features = voxel_features[b_ids[..., None], top_proposals_index]  # (B, K, D)
        selected_voxels = voxels_.unsqueeze(0).repeat(B, 1, 1)[b_ids[..., None], top_proposals_index]
        # (B, K) @ (K, D)
        cls_emb = logits[b_ids[..., None], top_proposals_index].softmax(-1) @ self.heads[0]['logits'].weight

        query = selected_features + cls_emb
        required_rates, instanceIds = [], []
        for b in range(B):
            temp_coor, ue = coor[b], ue_list[b]  # (\hat{K}, 3), (K, 3)
            # get indices of object with target ids
            filter_ids = [i for i, cat in enumerate(ue.category) if cat[self.target_ids].sum() > 0]
            ue = reader.Object(
                instanceId=ue.instanceId[filter_ids],
                category=ue.category[filter_ids],
                position=ue.position[filter_ids],
                los=ue.los.mT[filter_ids],
                required_rate=ue.required_rate[filter_ids],
                ca=ue.ca.mT[filter_ids],
            )

            distance = torch.cdist(temp_coor, ue.position)
            best_match = distance.argmin(dim=1)
            required_rate = ue.required_rate[best_match]
            required_rates.append(required_rate)
        required_rates = torch.stack(required_rates, dim=0)
        query = torch.cat([required_rates[..., None], query], dim=-1)
        query = self.embedding(query)

        for idx, (block, head_dict) in enumerate(zip(self.blocks, self.heads[1:])):
            feature = {
                key: list(features.values())[0]
                # key: list(features.values())[~idx]
                for key, features in visual_features.items()
            }
            query = block(query, coor, feature)

            output_dict = {key: head(query) for key, head in head_dict.items()}
            coor = coor.detach() + output_dict['offset'].tanh()*self.voxel_size*2
            output_dict['position'] = coor 
            # the idx-th embedding is train by and only by the idx-th layer
            
            output_list.append(output_dict)

        return {
            "output_list": output_list,
            "required_rates": required_rates,
        }
    
    # @torch.no_grad()
    # def _match(self, output_list, labels):
    #     output_list, points = outputs
    
    @torch.no_grad()
    def _match(self, output_list, labels):  # (L, B, K, ?), (B, K, ?)
        index_list = []  # (B, L)

        for b_idx in range(len(labels)):
            tgt_cls = labels[b_idx].category.argmax(-1)
            tgt_pos = labels[b_idx].position
            tgt_ca = labels[b_idx].ca.mT
            # print('tgt_ca', tgt_ca.shape)

            level_list = []
            for l in range(len(output_list)):
                logits = output_list[l]['logits'][b_idx]
                pos = output_list[l]['position'][b_idx]
                ca = output_list[l]['ca'][b_idx]
                # print(f'ca_{l}', tgt_ca.shape)

                # if l > 0:
                #     filter_ids = [i for i, tgt in enumerate(tgt_cls) if tgt in [self.target_ids]]
                #     tgt_cls = tgt_cls[filter_ids]
                #     tgt_pos = tgt_pos[filter_ids]
                #     tgt_ca = tgt_ca[filter_ids]

                out_prob = logits.sigmoid()
                log_scores = -out_prob[:, tgt_cls]  # (BK, 1)

                pos_scores = torch.cdist(pos, tgt_pos, p=1)  # (BK, 1)
                
                ca_scores = sigmoid_focal_loss(
                    ca.unsqueeze(1).expand(-1, tgt_ca.shape[0], -1),  # (N, -1, M) -> (N, K, M)
                    tgt_ca.unsqueeze(0).expand(ca.shape[0], -1, -1),  # (-1, K, M) -> (N, K, M)
                    alpha=-1, gamma=1#, reduction='mean'
                ).sum(-1)

                if l == 0:
                    C = 1.*log_scores + 1.*ca_scores + .6*pos_scores
                else:
                    C = 1.*log_scores + 1.*ca_scores + .6*pos_scores
                
                pred_ids, tgt_ids = linear_sum_assignment(C.cpu().numpy())
                level_list.append((pred_ids, tgt_ids))
                
            index_list.append(level_list)
        return index_list

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000, eta_min=1e-6
        )
        return [optimizer], [scheduler]
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def _get_num_query(self, n_steps):
        if self.training:
            if (level := n_steps // 196) < 10:
                # return self.voxels.shape[0] 
                return min(12000, self.voxels.shape[0])
            if (level := n_steps // 196) < 12:
                # return self.voxels.shape[0] // 4
                return 10000
            if (level := n_steps // 196) < 14:
                # return self.voxels.shape[0] // 4
                return 5000
            if (level := n_steps // 196) < 16:
                # return self.voxels.shape[0] // 4
                return 2000
            if (level := n_steps // 196) < 20:
                # return self.voxels.shape[0] // 4
                return 1000
            return 200
        return 200

    def training_step(self, batch, batch_idx):
        image_dict, object_list, _ = batch
        
        num_query = self._get_num_query(self._n_steps)
        # print('num_query', num_query)
        output_dict = self(object_list, num_query, image_dict)
        output_list = output_dict['output_list']
        index_list = self._match(output_list, object_list)
        
        loss = 0
        level = self._n_steps // 196
        self._n_steps += 1
        # level = 10
        ce_losses, pos_losses, ca_losses = [], [], []
        dec_ce_losses, dec_pos_losses, dec_ca_losses = [], [], []
        for b_idx in range(len(index_list)):
            losses = []
            tgt_cls = object_list[b_idx].category.argmax(-1)
            tgt_pos = object_list[b_idx].position
            tgt_ca = object_list[b_idx].ca.mT #.sigmoid()

            for l_idx in range(len(output_list)):
                coeff = float(level >= l_idx)
                pred_ids, tgt_ids = index_list[b_idx][l_idx]

                logits = output_list[l_idx]['logits'][b_idx]
                # print('train', l_idx, logits.shape)
                pos = output_list[l_idx]['position'][b_idx]
                ca = output_list[l_idx]['ca'][b_idx]#.sigmoid()

                target_classes = torch.zeros_like(logits)
                target_classes[pred_ids, tgt_cls] = 1
                loss_ce = sigmoid_focal_loss(
                    logits, target_classes,
                    # alpha=.95,
                    alpha=-1,
                    gamma=2,
                    reduction='mean'
                )
                if l_idx == 0:
                    # print(loss_ce.item())
                    ce_losses.append(loss_ce.item())
                else:
                    dec_ce_losses.append(loss_ce.item())

                loss_pos = F.l1_loss(pos[pred_ids], tgt_pos[tgt_ids], reduction='mean')
                if l_idx == 0:
                    pos_losses.append(loss_pos.item())
                else:
                    dec_pos_losses.append(loss_pos.item())

                target_ca = torch.zeros_like(ca)
                # print(target_ca.shape, target_ca[pred_ids].shape, tgt_ca[tgt_ids].shape)
                target_ca[pred_ids] = tgt_ca[tgt_ids]
                # loss_ca = 
                if l_idx == 0:
                    loss_ca = sigmoid_focal_loss(
                        ca[pred_ids], tgt_ca[tgt_ids],
                        alpha=-1, gamma=1, reduction='mean'
                    )
                    # loss_ca = sigmoid_focal_loss(
                    #     # ca[pred_ids], tgt_ca[tgt_ids],
                    #     ca, target_ca,
                    #     alpha=0.95, gamma=2, reduction='mean'
                    # )
                    ca_losses.append(loss_ca.item())
                else:
                    loss_ca = sigmoid_focal_loss(
                        ca[pred_ids], tgt_ca[tgt_ids],
                        alpha=-1, gamma=1, reduction='mean'
                    )
                    # loss_ca = sigmoid_focal_loss(
                    #     ca[pred_ids], tgt_ca[tgt_ids],
                    #     # ca, target_ca,
                    #     alpha=-1, gamma=1, reduction='mean'
                    # )
                    dec_ca_losses.append(loss_ca.item())
                    # print(loss_ca)
                    losses.append(loss_ca)
            
                if l_idx == 0:
                    loss += coeff*(loss_ce + .6*loss_ca + .8*loss_pos)
                else:
                    loss += coeff*(loss_ce + .6*loss_ca + .8*loss_pos)

        # acc_cls = ((pred_cls>0.5)*1. == target_classes).float().mean()
        # acc_link = ((pred_link[idx]>0.5)*1. == target_link).float().mean()
        # print(loss_pos.item())
            
        # print(min(level, self.n_layers-1), loss_poses[min(level, self.n_layers-1)])

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_pos", np.mean(pos_losses), prog_bar=True)
        self.log("train_dec_pos", np.mean(dec_pos_losses), prog_bar=True)

        self.log("train_ce", np.mean(ce_losses), prog_bar=True)
        self.log("train_ca", np.mean(ca_losses), prog_bar=True)
        self.log("train_dec_ce", np.mean(dec_ce_losses), prog_bar=True)
        self.log("train_dec_ca", np.mean(dec_ca_losses), prog_bar=True)

        return loss
        # return {
        #     "loss": loss,
        #     "loss_ce": loss_ce,
        #     "loss_pos": loss_pos,
        #     "loss_link": loss_link,
        #     # "acc_cls": acc_cls,
        #     # "acc_link": acc_link,
        # }
