from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass

import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
import torchmetrics as metrics

from scipy.optimize import linear_sum_assignment

from reader import Object


class SetCriterionMixin:

    @torch.no_grad()
    def _match(self, labels: List[Object], preds: List[Object]):
        C = np.zeros((len(labels), len(preds)))
        for i in range(len(labels)):
            for j in range(len(preds)):
                C[i, j] = labels[i].distance(preds[j])

        label_indinces, pred_indices = linear_sum_assignment(C)
        return label_indinces, pred_indices


class SetCriterion(nn.Module):
    
    def __init__(self,
        num_ue: int, num_sbs: int, num_classes: int,
        num_layers: int, pc_range,
        cost_class: float=1, cost_pos: float=1, cost_vis: float=1
    ):
        super().__init__()
        self.num_ue = num_ue
        self.num_sbs = num_sbs
        self._num_classes = num_classes
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))

        # self.auc = metrics.classification.MultilabelAUROC(
        #     num_labels=num_sbs, ignore_index=0, average="macro", thresholds=10
        # )
        self.acc = metrics.classification.MultilabelAccuracy(
            num_labels=num_sbs, ignore_index=0, average="macro", threshold=0.5
        )

    # @torch.no_grad()
    # def _match(self, labels, preds):
    #     pred_obj = [
    #         Object(
    #             instanceId=None,
    #             category=preds[0][0][i],
    #             position=preds[1][0][i]*(self.pc_range[:, 1] - self.pc_range[:, 0]) + self.pc_range[:, 0],
    #             los=preds[2][0][i]
    #         )
    #         for i in range(preds[0].shape[1])
    #     ]
    #     C = np.zeros((len(labels), len(pred_obj)))
    #     for i in range(len(labels)):
    #         for j in range(len(pred_obj)):
    #             C[i, j] = labels[i].distance(pred_obj[j])

    #     label_indinces, pred_indices = linear_sum_assignment(C)
    #     return label_indinces, pred_indices

    def forward(self, label_list: List[List[Object]], prediction: Tuple[Tensor, Tensor, Tensor]):
        cat_list, position_list, los_list = prediction
        num_layers, num_query = cat_list.shape[1], cat_list.shape[2]

        index_tuple_list = []
        with torch.no_grad():

            # for each sample in the batch
            for label, (cat, pos, los) in zip(label_list, zip(cat_list, position_list, los_list)):
                cat, pos, los = cat[0], pos[0], los[0]
                C = torch.zeros((label.category.shape[0], num_query), device=cat_list.device)  # (K, K_hat)

                # for each object in label list
                for i in range(label.category.shape[0]):
                    # cat_delta = F.binary_cross_entropy_with_logits(
                    cat_delta = sigmoid_focal_loss(
                        cat, label.category[i].unsqueeze(0).expand(num_query, -1), 
                        alpha=0.75, gamma=4, reduction='none'
                    ).mean(-1)
                    position_label = label.position[i].unsqueeze(0).expand(num_query, -1)
                    pos_delta = F.l1_loss(
                        pos,#*(self.pc_range[:, 1] - self.pc_range[:, 0]) + self.pc_range[:, 0],
                        # position_label,
                        (position_label - self.pc_range[:, 0]) / (self.pc_range[:, 1] - self.pc_range[:, 0]),
                        reduction='none'
                    ).mean(-1)
                    # los_delta = F.binary_cross_entropy_with_logits(
                    los_delta = sigmoid_focal_loss(
                        los, label.los[i].unsqueeze(0).expand(num_query, -1), alpha=0.75, reduction='none'
                    ).mean(-1)

                    C[i] = 0.005*cat_delta + 0.005*los_delta + 2*pos_delta

                label_indices, pred_indices = linear_sum_assignment(C.cpu().numpy())
                index_tuple_list.append((label_indices, pred_indices))

        class_label = torch.zeros_like(cat_list, requires_grad=False)  # (B, L, num_query, C)
        class_label[:, :, :, 0] = 1  # background
        for i, indices in enumerate(index_tuple_list):
            label_indices, pred_indices = indices
            class_label[i, :, pred_indices] = label_list[i].category[label_indices].unsqueeze(0).expand(num_layers, -1, -1).to(cat_list.dtype)

        losses = []
        acc = []
        for i, indices in enumerate(index_tuple_list):
            label_indices, pred_indices = indices
            
            # cls_loss = F.binary_cross_entropy_with_logits(cat_list[i], class_label[i])
            cls_loss = sigmoid_focal_loss(cat_list[i], class_label[i], alpha=0.75, gamma=4, reduction='mean')
                # label_list[i].category[label_indices].unsqueeze(0).expand(num_layers, -1, -1)
                
            position_label = label_list[i].position[label_indices].unsqueeze(0).expand(num_layers, -1, -1)
            reg_loss = F.l1_loss(
                position_list[i, :, pred_indices, :],#*(self.pc_range[:, 1] - self.pc_range[:, 0]) + self.pc_range[:, 0],
                (position_label - self.pc_range[:, 0]) / (self.pc_range[:, 1] - self.pc_range[:, 0]),
                # position_label
            )
            reg_loss_2 = F.mse_loss(
                position_list[i, :, pred_indices, :]*(self.pc_range[:, 1] - self.pc_range[:, 0]) + self.pc_range[:, 0],
                # (position_label - self.pc_range[:, 0]) / (self.pc_range[:, 1] - self.pc_range[:, 0])
                position_label
            )
            # print(
            #     position_list[i, -1, pred_indices, :], 
            #     (position_label[-1] - self.pc_range[:, 0]) / (self.pc_range[:, 1] - self.pc_range[:, 0])
            # )
            # los_loss = F.binary_cross_entropy_with_logits(
            los_loss = sigmoid_focal_loss(
                los_list[i, :, pred_indices],
                label_list[i].los[label_indices].unsqueeze(0).expand(num_layers, -1, -1),
                alpha=0.75, reduction='mean'
            )
            acc = self.acc(los_list[i, -1, pred_indices], label_list[i].los[label_indices].long())

            losses.append({
                'cls': cls_loss,
                'reg': reg_loss,
                'reg_2': reg_loss_2,
                'los': los_loss,
                'acc': acc
            })
        return losses

