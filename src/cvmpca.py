import math
import copy
# from collections import OrderedDict
from dataclasses import dataclass, field

from typing import Dict, List, Tuple, Optional, Callable, Any, Union, AnyStr, Generic

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
            pixel_coor[..., 0] = 2*(pixel_coor[..., 0] / (H - 1)) - 1
            pixel_coor[..., 1] = 2*(pixel_coor[..., 1] / (W - 1)) - 1
            feature_phead = feature.view(B, self.n_heads, self._head_dim, H, W)

            camera_feature = F.grid_sample(
                feature_phead.flatten(0, 1),
                pixel_coor.transpose(1, 2).flatten(0, 1),
                align_corners=True, padding_mode="zeros"
            ).view(B, self.n_heads, self._head_dim, K, self.n_points)
            camera_feature = camera_feature.permute(0, 3, 1, 4, 2)# * bounding.unsqueeze(-1)

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

        self.proj_attn_norm = nn.LayerNorm(embed_dim)
        # self.mhca_kv_norm = nn.LayerNorm(embed_dim)
        self.proj_attn = DeformableMHA(
            cameras=cameras,
            embed_dim=embed_dim, n_heads=n_heads, n_points=4,
            attention_dropout=attention_dropout, dropout=dropout,
        )

        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, [2*embed_dim, embed_dim], activation_layer=nn.GELU, dropout=dropout)
        
        # self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row") 1- (1-l/L)*1 + l/L*stochastic_depth_prob
        # self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.stochastic_depth = nn.Identity()
        # dropout is included in each layer

    def forward(self, query, points: Tensor, value):
        x = query + self.stochastic_depth(self.proj_attn(self.proj_attn_norm(query), points, value))

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
            feature = {key: list(features.values())[~idx] for key, features in feature_dict.items()}
            x = block(x, points, feature)

            output_list.append({
                key: head(x)
                for key, head in head_dict.items()# if key != 'position'
            })

            # pos_emb = head_dict['position'](x)
            # points = (self.inverse_sigmoid(points) + pos_emb).sigmoid()
            # output_list[idx]['position'] = points#.detach()

            points = points + output_list[idx]['position'].sigmoid()*self.voxel_size

        return output_list


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
        self.coeff = 4
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

        self.n_layers = len(embed_dims) + 1  # decoder and voxels
        if isinstance(ratio, int): ratio = [ratio]*self.n_layers
        assert len(ratio) == self.n_layers, "The length of ratio must be equal to the number of layers"
        self.ratio = ratio

        self.fpn = FeaturePyramidNetwork(embed_dims, embed_dim)
        self.embedding = nn.Linear(1+embed_dim, embed_dim)

        _cls_head = nn.Linear(embed_dim, n_classes, bias=False)
        _pos_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 3)
        )
        _link_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
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
            for _ in range(self.n_layers-1)
        ])
        self.heads = nn.ModuleList([
            nn.ModuleDict({
                "classification": copy.deepcopy(_cls_head),
                "position": copy.deepcopy(_pos_head),
                "link": copy.deepcopy(_link_head),
            }) for _ in range(self.n_layers)
        ])

        empty_weight = torch.ones(self.n_classes)
        empty_weight[0] = 0.1
        self.register_buffer("empty_weight", empty_weight)

        nn.init.xavier_uniform_(self.embedding.weight.data)
        nn.init.constant_(self.embedding.bias.data, 0.)
        for head_dict in self.heads:
            for layer in head_dict['position'].children():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight.data)
                    nn.init.constant_(layer.bias.data, 0.)
    
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
    
    def forward(self, ue_list: List[Dict], topk: int, features: List[Dict[str, Tensor]]):
        visual_features = {
            camera_key: self._process_view(feature_dict)
            for camera_key, feature_dict in features.items()
        }

        # aggregate visual features for each voxel
        voxels_ = self.voxels + self.voxel_size/2  # (N, 3)
        voxel_features = self._get_voxel_features(voxels_, visual_features)  # (B, N, D)
        
        # classify(/offset/ca) each voxel
        output_dict = {key: head(voxel_features) for key, head in self.heads[0].items()}
        coor = voxels_ + output_dict['position'].tanh()*self.voxel_size*3
        output_dict['position'] = coor
        output_list = [output_dict]

        # TODO: support multiple target ids
        # choose top K of target ids only (e.g., phone)
        logits = output_dict['classification']  # (B, N, C)
        B = logits.shape[0]
        # _, indices = logits.view(B, -1).topk(topk, dim=-1)
        _, indices = logits[..., self.target_ids].view(B, -1).topk(topk, dim=-1)
        top_proposals_index = indices // self.n_classes  # (B, K) which voxel
        top_proposals_class = indices % self.n_classes   # (B, K) which class

        b_ids = torch.arange(B)
        coor = coor[b_ids[..., None], top_proposals_index]  # (B, K, 3)
        selected_features = voxel_features[b_ids[..., None], top_proposals_index]  # (B, K, D)
        # (B, K) @ (K, D)
        cls_emb = logits[b_ids[..., None], top_proposals_index].softmax(-1) @ self.heads[0]['classification'].weight

        query = selected_features + cls_emb
        required_rates = []
        for b in range(B):
            temp_coor, ue = coor[b], ue_list[b]  # (\hat{K}, 3), (K, 3)
            # get indices of object with target ids
            filter_ids = [i for i, cat in enumerate(ue.category) if cat[self.target_ids].sum() > 0]
            ue = reader.Object(
                instanceId=ue.instanceId[filter_ids],
                category=ue.category[filter_ids],
                position=ue.position[filter_ids],
                los=ue.los[filter_ids],
                required_rate=ue.required_rate[filter_ids],
            )
            # print(ue)

            distance = torch.cdist(temp_coor, ue.position)
            best_match = distance.argmin(dim=1)
            required_rate = ue.required_rate[best_match]
            required_rates.append(required_rate)
        required_rates = torch.stack(required_rates, dim=0)
        # print(required_rates.shape, query.shape)
        query = torch.cat([required_rates[..., None], query], dim=-1)
        query = self.embedding(query)

        for idx, (block, head_dict) in enumerate(zip(self.blocks, self.heads[1:])):
            feature = {
                key: list(features.values())[~idx]
                for key, features in visual_features.items()
            }
            query = block(query, coor, feature)

            output_dict = {key: head(query) for key, head in head_dict.items()}
            coor = coor.detach() + output_dict['position'].tanh()*self.voxel_size*3
            output_dict['position'] = coor 
            # the idx-th embedding is train by and only by the idx-th layer
            
            output_list.append(output_dict)

        return {
            "output_list": output_list
        }
    
    # @torch.no_grad()
    # def _match(self, output_list, labels):
    #     output_list, points = outputs
    
    @torch.no_grad()
    def _match(self, output_list, labels):
        index_list = []
        
        tgt_ids = torch.cat([tgt.category for tgt in labels]).argmax(-1)
        tgt_pos = torch.cat([tgt.position for tgt in labels])
        tgt_link = torch.cat([tgt.los for tgt in labels]).sigmoid()

        # for each layer's output
        for idx, output_dict in enumerate(output_list):
            pred_cls = output_dict['classification']  # (B, K, C)
            pred_pos = output_dict['position']  # (B, K, 3)
            pred_link = output_dict['link']  # (B, K, M)

            # after the first layer, only train the target ids
            if idx > 0:
                filter_ids = [i for i, tgt in enumerate(tgt_ids) if tgt in [self.target_ids]]

                tgt_ids = tgt_ids[filter_ids]
                tgt_pos = tgt_pos[filter_ids]
                tgt_link = tgt_link[filter_ids]
            
            B, K, _ = pred_cls.shape

            out_prob = pred_cls.flatten(0, 1).softmax(-1)
            log_scores = -out_prob[:, tgt_ids]
            
            loss_pos = torch.cdist(pred_pos.flatten(0, 1), tgt_pos, p=2)

            loss_link = sigmoid_focal_loss(
                pred_link.flatten(0, 1).unsqueeze(1).expand(-1, tgt_link.shape[0], -1),
                tgt_link.unsqueeze(0).expand(pred_link.shape[0]*pred_link.shape[1], -1, -1),
                alpha=0.25, gamma=2, reduction='mean'
            )

            C = .5*log_scores + 1.*loss_pos + .4*loss_link
            C = C.view(B, K, -1).cpu()
            
            if idx == 0:
                sizes = [len(obj.category) for obj in labels]
            else:
                sizes = [
                    len([cat for cat in obj.category.argmax(-1) if cat in [self.target_ids]])
                    for obj in labels
                ]
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
            return 200
        return 50

    def training_step(self, batch, batch_idx):
        image_dict, object_list = batch
        
        num_query = self._get_num_query(batch_idx)
        output_dict = self(object_list, num_query, image_dict)

        index_list = self._match(output_dict['output_list'], object_list)
        
        loss = 0
        level = batch_idx // 196
        loss_ces, loss_poses, loss_links = [], [], []
        for l, (output, indices) in enumerate(zip(output_dict['output_list'], index_list)):
            pred_cls = output['classification']
            # pred_pos = points + output_dict['position'].sigmoid()*self.voxel_size*coeff
            pred_pos = output['position']
            pred_link = output['link']

            _device = pred_cls.device
            B, P, C = pred_cls.shape

            idx = self._get_src_permutation_idx(indices)

            if l > 0:
                for b in range(B):
                    ue = object_list[b]
                    filter_ids = [i for i, cat in enumerate(ue.category) if cat[self.target_ids].sum() == 1]
                    object_list[b] = reader.Object(
                        instanceId=ue.instanceId[filter_ids],
                        category=ue.category[filter_ids],
                        position=ue.position[filter_ids],
                        los=ue.los[filter_ids],
                        required_rate=ue.required_rate[filter_ids],
                    )
            target_classes = torch.full((B, P, C), 0, dtype=torch.float, device=_device)
            for i, (pred_ids, label_ids) in enumerate(indices):
                target_classes[i, pred_ids] = object_list[i].category[label_ids]#.long()
            
            loss_ce = sigmoid_focal_loss(
                pred_cls, target_classes,
                alpha=0.25, gamma=2,
                reduction='mean'
            )
            loss_ces.append(loss_ce.item())

            target_boxes = torch.cat([obj.position[i] for obj, (_, i) in zip(object_list, indices)], dim=0)
            loss_pos = F.l1_loss(pred_pos[idx], target_boxes, reduction='mean')
            loss_poses.append(loss_pos.item())

            target_link = torch.cat([obj.los[i] for obj, (_, i) in zip(object_list, indices)], dim=0)
            loss_link = sigmoid_focal_loss(pred_link[idx], target_link, alpha=0.75, gamma=2, reduction='mean')
            loss_links.append(loss_link.item())

            coeff = float(level >= l)
            loss += coeff*(loss_ce + loss_link + 0.6*loss_pos)
        # else:
        
        # if (batch_idx+1) % 100 == 0:
        #     print(batch_idx, output_dict['output_list'][0]['position'][self._get_src_permutation_idx(indices[0])])
        #     print(object_list[0].position)
        #     print(
        #         F.l1_loss(output_dict['output_list'][0]['position'][self._get_src_permutation_idx(indices[0])],
        #                   object_list[0].position, reduction='mean')
        #     )
        #     print('-'*30)

        # acc_cls = ((pred_cls>0.5)*1. == target_classes).float().mean()
        # acc_link = ((pred_link[idx]>0.5)*1. == target_link).float().mean()
        # print(loss_pos.item())
            
        # print(min(level, self.n_layers-1), loss_poses[min(level, self.n_layers-1)])

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_ce", loss_ces[min(level, self.n_layers-1)], prog_bar=True)
        self.log("train_pos", loss_poses[min(level, self.n_layers-1)], prog_bar=True)
        self.log("train_link", loss_links[min(level, self.n_layers-1)], prog_bar=True)

        # self.log("train_acc_cls", acc_cls, prog_bar=True)
        # self.log("train_acc_link", acc_link, prog_bar=True)

        return loss
        # return {
        #     "loss": loss,
        #     "loss_ce": loss_ce,
        #     "loss_pos": loss_pos,
        #     "loss_link": loss_link,
        #     # "acc_cls": acc_cls,
        #     # "acc_link": acc_link,
        # }
