import math
import copy
# from collections import OrderedDict
from dataclasses import dataclass, field

from typing import Dict, List, Tuple, Optional, Callable, Any, Union

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


def absolute2norm(coordinate, space):
    '''
        Normalize the absolute coordinate into the range of [0, 1]
    '''
    return (coordinate - space[..., 0]) / (space[..., 1] - space[..., 0])


def norm2absolute(coordinate, space):
    '''
        Denormalize the coordinate into the range of [-space, space]
    '''
    return coordinate * (space[..., 1] - space[..., 0]) + space[..., 0]


class DeformaleMHA(nn.Module):

    def __init__(self,
        cameras: Dict[str, utils.Camera],
        embed_dim, n_heads, n_points,
        attention_dropout, dropout,
    ):
        super().__init__()
        self.cameras, self.n_cameras = cameras, len(cameras)
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_points = n_points
        self._head_dim = embed_dim // n_heads

        self.sampling_offsets = nn.Linear(embed_dim, 2*n_heads*n_points)
        self.attention_weights = nn.Linear(embed_dim, n_heads*n_points)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.output_dropout = nn.Dropout(dropout)
        
        self.proj = nn.Linear(self.n_cameras*embed_dim, embed_dim)

        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 2).repeat(1, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.proj.weight.data)
        nn.init.constant_(self.proj.bias.data, 0.)

    def _headify(self, x: Tensor):
        return x.view(*x.shape[:2], self.n_heads, self._head_dim).transpose(1, 2)
    
    def _project(self, points, camera, feature):
        H, W = feature.shape[-2:]

        pixel_coor = camera(points, [H, W, 1])  # (B, N, 2)
        pixel_coor[..., 0] = 2*pixel_coor[..., 0] / (H - 1) - 1
        pixel_coor[..., 1] = 2*pixel_coor[..., 1] / (W - 1) - 1
        bounding = torch.logical_and(
            torch.logical_and(pixel_coor[..., 0] >= 0, pixel_coor[..., 0] <= H),
            torch.logical_and(pixel_coor[..., 1] >= 0, pixel_coor[..., 1] <= W)
        )  # (B, N)
        pixel_coor = pixel_coor.unsqueeze(-2)

        camera_feature = F.grid_sample(feature, pixel_coor, align_corners=True).squeeze(-1)
        camera_feature = camera_feature*bounding.unsqueeze(1)  # (BCK, D, N)
        return camera_feature.permute(0, 2, 1)  # 
    
    def forward(self, 
        query: Tensor,  # (B, K, D)
        points: Tensor,  # (B, N, 3)
        feature_dict: Dict[str, Tensor],
    ):
        B, K, _ = query.shape
        # sampling
        
        xs = []
        for camera_key in self.cameras.keys():
            feature = feature_dict[camera_key]
            camera = self.cameras[camera_key]  # (B, D, H, W)
            H, W = feature.shape[-2:]

            camera_feature = self._project(points, camera, feature)
            _query = camera_feature + query

            sampling_offset = self.sampling_offsets(_query).reshape(B, K, self.n_heads, self.n_points, 2)#.tanh()
            attention_weight = self.attention_weights(_query).view(B, K, self.n_heads, self.n_points, 1)
            attention_weight = attention_weight.softmax(-2)
            attention_weight = self.attention_dropout(attention_weight)
            
            H, W = feature.shape[-2:]
            pixel_coor = camera(points, [H, W, 1])  # (B, N, 2)
            sampling_offset = sampling_offset / torch.tensor([H, W], device=sampling_offset.device)
            pixel_coor = pixel_coor.unsqueeze(-2).unsqueeze(-2) + sampling_offset
            # pixel_coor[..., 0] = 2*pixel_coor[..., 0] / (H - 1) - 1
            # pixel_coor[..., 1] = 2*pixel_coor[..., 1] / (W - 1) - 1
            pixel_coor[..., 0] = 2*(pixel_coor[..., 0] / (H - 1)) - 1
            pixel_coor[..., 1] = 2*(pixel_coor[..., 1] / (W - 1)) - 1
            bounding = torch.logical_and(
                torch.logical_and(pixel_coor[..., 0] >= 0, pixel_coor[..., 0] <= H),
                torch.logical_and(pixel_coor[..., 1] >= 0, pixel_coor[..., 1] <= W)
            )  # (B, N)
            # pixel_coor = pixel_coor.unsqueeze(-2)
            feature_phead = feature.view(B, self.n_heads, self._head_dim, H, W)

            camera_feature = F.grid_sample(
                feature_phead.flatten(0, 1),
                pixel_coor.transpose(1, 2).flatten(0, 1),
                align_corners=True,
            ).view(B, self.n_heads, self._head_dim, K, self.n_points)
            camera_feature = camera_feature.permute(0, 3, 1, 4, 2) * bounding.unsqueeze(-1)

            weighted_feature = (camera_feature * attention_weight).sum(-2).view(B, K, self.embed_dim)
            out_feature = self.output_dropout(self.output_proj(weighted_feature))

            xs.append(out_feature)
        xs = torch.cat(xs, dim=-1)
        return self.proj(xs)


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

        self.mha_norm = nn.LayerNorm(embed_dim)
        self.mha_norm_2 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)

        self.proj_attn_norm = nn.LayerNorm(embed_dim)
        # self.mhca_kv_norm = nn.LayerNorm(embed_dim)
        self.proj_attn = DeformaleMHA(
            cameras=cameras,
            embed_dim=embed_dim, n_heads=n_heads, n_points=4,
            attention_dropout=attention_dropout, dropout=dropout,
        )

        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, [2*embed_dim, embed_dim], activation_layer=nn.GELU, dropout=dropout)
        
        # self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row") 1- (1-l/L)*1 + l/L*stochastic_depth_prob
        self.stochastic_depth = nn.Identity()
        # dropout is included in each layer

    def forward(self, query, point_emb, points: Tensor, value):
        # _x = query  # (B, C, K, D)
        x = self.mha_norm(query + point_emb)
        x, _ = self.mha(x, x, self.mha_norm_2(query))
        x = query + point_emb + x

        x = query + self.stochastic_depth(
            self.proj_attn(self.proj_attn_norm(query), points, value)
        )

        x = x + self.stochastic_depth(self.mlp(self.mlp_norm(x)))

        return x


class Decoder(nn.Module):

    def __init__(self,
        cameras,
        space, voxel_size,
        n_classes,
        n_layers,
        embed_dim, n_heads,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_layers = n_layers
        # self.space, self.voxel_size = space, voxel_size
        self.register_buffer('space', torch.tensor(space))
        self.register_buffer('voxel_size', torch.tensor(voxel_size))
        
        self.blocks = nn.ModuleList([
            Block(
                cameras=cameras, space=space, voxel_size=voxel_size,
                embed_dim=embed_dim, n_heads=n_heads,
                attention_dropout=0.1, dropout=0.1,
            )
            for _ in range(n_layers)
        ])

        # _share_head = nn.Sequential(
        #     nn.Linear(embed_dim, 2*embed_dim),
        #     nn.LayerNorm(2*embed_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(2*embed_dim, embed_dim),
        #     nn.LayerNorm(embed_dim),
        #     nn.GELU(),
        # )

        _cls_head = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(2*embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(embed_dim, n_classes)
        )
        _pos_head = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(2*embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(embed_dim, 3)
        )
        _link_head = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(2*embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(embed_dim, len(cameras))
        )
        
        self.heads = nn.ModuleList([
            nn.ModuleDict({
                "classification": copy.deepcopy(_cls_head),
                "position": copy.deepcopy(_pos_head),
                "link": copy.deepcopy(_link_head),
                # "classification": copy.deepcopy(_cls_head)
            }) for _ in range(n_layers)
        ])

    @staticmethod
    def inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1/x2)

    def forward(self,
        x: Tensor,  # (B, K, D)
        points: Tensor,  # (B, N, D)
        feature_dict: Dict[str, Tensor]  # Dict[str, Dict[str, Tensor]]
    ): 
        output_list = []
        for idx, (block, head_dict) in enumerate(zip(self.blocks, self.heads)):
            coeff = self.n_layers - idx
            feature = {key: list(features.values())[~idx] for key, features in feature_dict.items()}
            x = block(x, points, feature)

            output_list.append({
                key: head(x)
                for key, head in head_dict.items()# if key != 'position'
            })

            # pos_emb = head_dict['position'](x)
            # points = (self.inverse_sigmoid(points) + pos_emb).sigmoid()
            # output_list[idx]['position'] = points#.detach()

            points = points + output_list[idx]['position'].sigmoid()*self.voxel_size*coeff

        return output_list


class CVMPCA(pl.LightningModule):
# class CVMPCA(nn.Module):

    def __init__(self,
        captures: List[Capture],
        n_classes: int,
        space: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        voxel_size: Tuple[float, float, float],
        ratio: Union[int, List[int]]
    ):
        super().__init__()
        self.n_classes = n_classes
        self.coeff = 4
        self.register_buffer('space', torch.tensor(space))
        self.register_buffer('voxel_size', torch.tensor(voxel_size))
        self.register_buffer('voxels', self._voxelize())

        _cameras = {
            capture.id: utils.Camera.from_unity(capture)
            for capture in captures
        }
        self.n_cameras = len(_cameras)
        self.cameras = nn.ModuleDict(_cameras)

        self.swin = transformer.Swin(is_trainable=True)
        embed_dim = self.swin.embed_dim
        embed_dims = self.swin.embed_dims
        self.n_layers = len(embed_dims)
        if isinstance(ratio, int): ratio = [ratio]*self.n_layers
        assert len(ratio) == self.n_layers, "The length of ratio must be equal to the number of layers"
        self.ratio = ratio

        self.fpn = FeaturePyramidNetwork(embed_dims, embed_dim)
        self.cnn_layer = nn.Conv2d(3, embed_dim, kernel_size=1, stride=1, padding=0)
        self.head = nn.Linear(embed_dim, n_classes, bias=False)

        _emb = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            # nn.Linear(embed_dim, embed_dim, bias=False)
        )
        self.reference_points = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3)
        )

        self.cartesian2emb = nn.ModuleList([
            _emb
            # copy.deepcopy(_emb)
            for _ in range(self.n_layers)
        ])
        # self.emb2cartesian = nn.ModuleList([
        #     _emb2
        #     # copy.deepcopy(_emb)
        #     for _ in range(self.n_layers)
        # ])

        _cls_head = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2*embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, n_classes)
        )
        _pos_head = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2*embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 3)
        )
        _link_head = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2*embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, len(self.cameras))
        )
        
        self.blocks = nn.ModuleList([
            Block(
                cameras=_cameras, space=space, voxel_size=voxel_size,
                embed_dim=embed_dim, n_heads=6,
                attention_dropout=0.1, dropout=0.1,
            )
            for _ in range(self.n_layers)
        ])
        
        self.heads = nn.ModuleList([
            nn.ModuleDict({
                "classification": copy.deepcopy(_cls_head),
                "position": copy.deepcopy(_pos_head),
                "link": copy.deepcopy(_link_head),
            }) for _ in range(self.n_layers)
        ])

        # self.decoder = Decoder(
        #     cameras=self.cameras,
        #     space=space, voxel_size=voxel_size,
        #     n_classes=n_classes, n_layers=self.n_layers,
        #     embed_dim=embed_dim, n_heads=6,
        # )

        empty_weight = torch.ones(self.n_classes)
        empty_weight[0] = 0.1
        self.register_buffer("empty_weight", empty_weight)

    def _original(self, indices):
        n_bins = (self.space[..., 1] - self.space[..., 0]) / self.voxel_size
        X, Y, Z = n_bins

        x, y, z = (indices//Z)//Y, (indices//Z)%Y, indices%Z
        
        return torch.stack([x, y, z], dim=-1)#.long()
    
    def _voxelize(self):
        space, voxel_size = self.space, self.voxel_size
        X = torch.arange(space[0][0], space[0][1] + voxel_size[0]/4, voxel_size[0]) #, device=voxel_size.device)
        Y = torch.arange(space[1][0], space[1][1] + voxel_size[1]/4, voxel_size[1]) #, device=voxel_size.device)
        Z = torch.arange(space[2][0], space[2][1] + voxel_size[2]/4, voxel_size[2]) #, device=voxel_size.device)
        
        grid_x, grid_y, grid_z = torch.meshgrid(X, Y, Z, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
    
    def _get_voxel_features(self, voxels, features) -> Dict[str, Tensor]:
        camera_features = {}  # obtain the corresponding features (convert voxels into features)
        
        B, N, _ = voxels.shape[:4]
        for camera_key in self.cameras.keys():
            feature_dict = features[camera_key]
            camera, feature = self.cameras[camera_key], list(feature_dict.values())[0]  # (B, D, H, W)
            H, W = feature.shape[-2:]
            
            pixel_coor = camera(voxels, [H, W, 1])  # (B, N, 2)
            # pixel_coor[..., 0] = 2*pixel_coor[..., 0] / (H - 1) - 1
            # pixel_coor[..., 1] = 2*pixel_coor[..., 1] / (W - 1) - 1
            pixel_coor[..., 0] = 2*(pixel_coor[..., 0] / (H - 1)) - 1
            pixel_coor[..., 1] = 2*(pixel_coor[..., 1] / (W - 1)) - 1
            bounding = torch.logical_and(
                torch.logical_and(pixel_coor[..., 0] >= 0, pixel_coor[..., 0] <= H),
                torch.logical_and(pixel_coor[..., 1] >= 0, pixel_coor[..., 1] <= W)
            )  # (B, N)

            # for grid_sample compatibility. (B, N(*K), 1, 2)
            pixel_coor = pixel_coor.unsqueeze(-2)
            camera_feature = F.grid_sample(feature, pixel_coor, align_corners=True).squeeze(-1)
            camera_feature = camera_feature*bounding.unsqueeze(1)  # (BCK, D, N)
            camera_feature = camera_feature.permute(0, 2, 1)  # 

            camera_features[camera_key] = camera_feature
        return torch.stack([item for item in camera_features.values()], dim=-1).sum(-1)

    def _single_view(self, images: Tensor) -> Dict[str, Dict[str, Tensor]]:
        return self.fpn({
            f'feat{i}': x.permute(0, 3, 1, 2)
            for i, x in enumerate(self.swin(images))
        })
    
    @staticmethod
    def create_grid(h, w, device=None):
        x = torch.arange(0, w, dtype=torch.float, device=device)
        y = torch.arange(0, h, dtype=torch.float, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        return torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones(h*w, device=device)], dim=-1)
    
    @staticmethod
    def inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1/x2)

    def forward(self, top_k, image_dict: Dict[str, Tensor]):
        keys = list(image_dict.keys())  # camera ids
        B = image_dict[keys[0]].size(0)
        
        visual_features = {
            key: self._single_view(images)
            for key, images in image_dict.items()
        }
        # TODO: include GPE
        for key, feature_dict in visual_features.items():
            for key_, feature in feature_dict.items():
                B, D, H, W = feature.shape
                grid = self.create_grid(H, W, device=feature.device)
                rays = self.cameras[key].pix2ray(grid, [H, W, 1])
                rays = rays.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
                
                visual_features[key][key_] = self.cnn_layer(rays) + feature
                
        voxels = self.voxels.unsqueeze(0).repeat(B, 1, 1)  # (B, N, 3)

        # aggregate visual features from all view into each voxel
        voxel_features = self._get_voxel_features(voxels + self.voxel_size/2, visual_features)  # (B, N, D)
        
        logits = self.head(voxel_features)  # (B, N, C)
        _, indices = logits.view(B, -1).topk(top_k, dim=-1)
        top_proposals_index = indices // self.n_classes  # (B, K) which voxel
        top_proposals_class = indices % self.n_classes   # (B, K) which class

        # refinement
        voxels_ = voxels + self.voxel_size/2
        points = voxels_[torch.arange(B).unsqueeze(-1), top_proposals_index]
        point_logits = self.reference_points(points).sigmoid()

        # construct query: feature + class
        selected_voxels = voxel_features[torch.arange(B).unsqueeze(-1), top_proposals_index]  # (B, K, D)
        cls_emb = logits[torch.arange(B).unsqueeze(-1), top_proposals_class].softmax(-1) @ self.head.weight
        query = selected_voxels + cls_emb
        
        output_list = []
        for idx, (emb, block, head_dict) in enumerate(zip(
            self.cartesian2emb, self.blocks, self.heads
        )):
            feature = {key: list(features.values())[~idx] for key, features in visual_features.items()}
            x = block(
                query,
                emb(point_logits), norm2absolute(point_logits, self.space),
                feature
            )

            output_dict = {key: head(x) for key, head in head_dict.items()}

            point_logits = (self.inverse_sigmoid(point_logits.detach()) + output_dict['position']).sigmoid()
            output_dict['position'] = point_logits
            # points = norm2absolute(new_point, self.space)
            # the idx-th embedding is train by and only by the idx-th layer
            
            output_list.append(output_dict)


        # refine the query with features
        # output_list = self.decoder(query, points, visual_features)  # (B, L, N, C + 3 + len(cameras))
        
        return output_list, points
    
    @torch.no_grad()
    def _match(self, outputs, labels):
        output_list, points = outputs

        index_list = []
        for idx, output_dict in enumerate(output_list):
            coeff = self.n_layers - idx

            pred_cls = output_dict['classification']
            # pred_pos = points + output_dict['position'].sigmoid()*self.voxel_size*coeff
            # points = points + output_dict['position'].sigmoid()*self.voxel_size*coeff
            pred_pos = norm2absolute(output_dict['position'], self.space)
            pred_link = output_dict['link']  # (B, K, C/3/M)

            B, K, P = pred_cls.shape
            
            tgt_ids = torch.cat([tgt.category for tgt in labels]).argmax(-1)
            tgt_pos = torch.cat([tgt.position for tgt in labels])
            tgt_link = torch.cat([tgt.los for tgt in labels]).sigmoid()
            # print(tgt_ids.shape, tgt_pos.shape, tgt_link.shape)

            out_prob = pred_cls.flatten(0, 1).softmax(-1)
            log_scores = -out_prob[:, tgt_ids]
            
            loss_pos = torch.cdist(pred_pos.flatten(0, 1), tgt_pos, p=1)

            # temporally borrow cdist for binary classification
            loss_link = torch.cdist(pred_link.flatten(0, 1), tgt_link, p=1)

            C = .5*log_scores + 2*loss_link + 2*loss_pos
            C = C.view(B, K, -1).cpu()
            
            sizes = [len(obj.category) for obj in labels]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            index_list.append([
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices
            ])
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
    
    def _get_num_query(self, batch_idx):
        if self.training:
            return 100*max(batch_idx)
        return 50

    def training_step(self, batch, batch_idx):
        image_dict, object_list = batch
        for obj in object_list:
            obj.category = obj.category.cuda()
            obj.position = obj.position.cuda()
            obj.los = obj.los.cuda()
        image_dict = {
            key: item.cuda()
            for key, item in image_dict.items()
        }
        
        num_query = 200
        output_list, points = self(num_query, image_dict)

        index_list = self._match(
            (output_list, points),
            object_list
        )

        # _device = pred_cls.device
        # B, L, P, C = pred_cls.shape
        
        loss = 0
        for idx, (output_dict, indices) in enumerate(zip(output_list, index_list)):
            coeff = self.n_layers - idx

            pred_cls = output_dict['classification']
            # pred_pos = points + output_dict['position'].sigmoid()*self.voxel_size*coeff
            pred_pos = norm2absolute(output_dict['position'], self.space)
            pred_link = output_dict['link']

            _device = pred_cls.device
            B, P, C = pred_cls.shape
            
            target_classes = torch.full((B, P, C), 0, dtype=torch.float, device=_device)
            for i, (pred_ids, label_ids) in enumerate(indices):
                target_classes[i, pred_ids] = object_list[i].category[label_ids]

            idx = self._get_src_permutation_idx(indices)

            target_classes = torch.full((B, P, C), 0, dtype=torch.float, device=_device)
            for i, (pred_ids, label_ids) in enumerate(indices):
                target_classes[i, pred_ids] = object_list[i].category[label_ids]#.long()
            
            loss_ce = sigmoid_focal_loss(
                pred_cls, target_classes,
                alpha=0.25, gamma=2, reduction='mean'
            )

            target_boxes = torch.cat([obj.position[i] for obj, (_, i) in zip(object_list, indices)], dim=0)
            loss_pos = F.l1_loss(pred_pos[idx], target_boxes, reduction='mean')

            target_link = torch.cat([obj.los[i] for obj, (_, i) in zip(object_list, indices)], dim=0)
            loss_link = sigmoid_focal_loss(pred_link[idx], target_link, alpha=0.75, gamma=2, reduction='mean')

            loss += loss_ce + loss_link + loss_pos
            
        # acc_cls = ((pred_cls[:, -1]>0.5)*1. == target_classes).float().mean()
        # acc_link = ((pred_link[:, -1][idx]>0.5)*1. == target_link).float().mean()
        # print(loss_pos.item())

        # self.log("train_loss", loss, prog_bar=True)
        # self.log("train_ce", loss_ce, prog_bar=True)
        # self.log("train_pos", loss_pos, prog_bar=True)
        # self.log("train_link", loss_link, prog_bar=True)

        # self.log("train_acc_cls", acc_cls, prog_bar=True)
        # self.log("train_acc_link", acc_link, prog_bar=True)

        return {
            "loss": loss,
            "loss_ce": loss_ce,
            "loss_pos": loss_pos,
            "loss_link": loss_link,
            # "acc_cls": acc_cls,
            # "acc_link": acc_link,
        }
