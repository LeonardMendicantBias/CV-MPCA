import math
from typing import Union, Tuple, List

import torch
from torch import nn, Tensor, BoolTensor
import torch.nn.functional as F



class MHA(nn.Module):

    def __init__(self,
        embed_dim: int,
        num_heads: int,
        attention_dropout: float,
        dropout: float,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self._head_dim = embed_dim//num_heads

        self.query = self.build_query()
        self.key = self.build_key()
        self.value = self.build_value()
        self.linear = self.build_linear()

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(dropout)

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
    
    def forward(self, 
        query: Tensor, key: Tensor=None, value: Tensor=None,
        mask: Tensor=None
    ):
        if key is None: key = query
        if value is None: value = key
        
        q, k, v = self.query(query), self.key(key), self.value(value)
        q, k, v = self._headify(q), self._headify(k), self._headify(v)

        logits = self._calculate_logits(q, k)
        scores = self._calculate_attention(logits, mask)
        scores = self.attention_dropout(scores)

        o = self._aggregate(scores, v)
        o = self._deheadify(o)
        z = self.linear(o)
        return self.dropout(z)
    
class RelativeMHA(MHA):

    def __init__(self,
        *args, 
        rel,
        **kwargs
    ):
        self.rel = rel

        super().__init__(*args, **kwargs)

    def _get_relative_distance(self, s, t):
        coords_h, coords_w = torch.arange(s), torch.arange(t)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww

    def forward(self, x: Tensor, mask: Tensor=None):

        pass

