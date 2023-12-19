import math
from dataclasses import dataclass, field
from typing import Union, Tuple, List

import torch
from torch import nn, Tensor, BoolTensor
import torch.nn.functional as F


class MHA(nn.Module): 
    
    def __init__(self, embed_dim, num_heads, attention_dropout, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.query = self.build_query()
        self.key = self.build_key()
        self.value = self.build_value()
        self.linear = self.build_linear()

        self.attention_dropout = nn.Dropout(self.attention_dropout)
        self.dropout = nn.Dropout(self.dropout)

    def build_query(self):
        return nn.Linear(self.embed_dim, self.embed_dim)

    def build_key(self):
        return nn.Linear(self.embed_dim, self.embed_dim)
    
    def build_value(self):
        return nn.Linear(self.embed_dim, self.embed_dim)
    
    def build_linear(self):
        return nn.Linear(self.embed_dim, self.embed_dim)
    
    def _headify(self, x: Tensor):
        B, T = x.shape[:2]
        return x.view(B, T, self.num_heads, self._head_dim).transpose(1, 2)
    
    def _deheadify(self, x: Tensor):
        B, H, T, D = x.shape
        return x.transpose(1, 2).reshape(B, T, H*D)
    
    def _calculate_logits(self, q: Tensor, k: Tensor):
        return (q @ k.mT) / math.sqrt(self.embed_dim)
    
    def _calculate_attention(self, logits: Tensor, mask: BoolTensor):
        if mask is not None:
            logits = logits.masked_fill(mask, float('-inf'))
        return F.softmax(logits, dim=-1)
    
    def _aggregate(self, scores: Tensor, v: Tensor):
        return scores @ v
    
    def forward(self,  # (B, T, D) and (B, S, D)
        query: Tensor, key: Tensor=None, value: Tensor=None,
        mask: Tensor=None
    ):
        if key is None: key = query
        if value is None: value = key
        
        q, k, v = self.query(query), self.key(key), self.value(value)
        q, k, v = self._headify(q), self._headify(k), self._headify(v)
        # (B, H, T, D') and (B, H, S, D')

        logits = self._calculate_logits(q, k)  # (B, H, T, S)
        scores = self._calculate_attention(logits, mask)
        scores = self.attention_dropout(scores)

        o = self._aggregate(scores, v)  # (B, H, T, S) @ # (B, H, S, D') -> (B, H, T, D')
        o = self._deheadify(o)
        z = self.linear(o)

        return self.dropout(z), scores
   