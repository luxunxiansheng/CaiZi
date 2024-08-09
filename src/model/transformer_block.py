""" 
###########################################
#
# Author: Bin.Li                          
# Email:  ornot2008@yahoo.com
# MIT License
# Copyright (c) 2025 debutpark.com 
#
###########################################
"""

import torch.nn as nn
import torch


from model.layer_norm import LayerNorm
from model.mlp import MLP
from model.multihead_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dimension_embedding: int,
        block_size: int,
        num_heads: int = 1,
        drop_rate: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.layernorm_1 = LayerNorm(dimension_embedding, bias)
        self.attention = MultiHeadAttention(
            dimension_embedding,
            block_size,
            num_heads,
            drop_rate,
            bias,
        )
        self.layernorm_2 = LayerNorm(dimension_embedding,bias)
        self.mlp = MLP(dimension_embedding, drop_rate, bias)
      
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layernorm_1(x))
        x = x + self.mlp(self.layernorm_2(x))
        return x

