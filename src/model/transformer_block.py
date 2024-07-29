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
    def __init__(self, dimension_input: int, dimension_embedding: int, block_size: int, num_heads: int = 1, drop_rate: float = 0.0, qkv_bias: bool = False):
        super().__init__()
        self.layernorm_1 = LayerNorm(dimension_embedding)
        self.attention = MultiHeadAttention(dimension_input, dimension_embedding, block_size, num_heads, drop_rate, qkv_bias)
        self.layernorm_2 = LayerNorm(dimension_embedding)
        self.mlp = MLP(dimension_embedding)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shortcut connection for attention block
        shortcut = x
        x = self.layernorm_1(x)
        x = self.attention(x)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.layernorm_2(x)
        x = self.mlp(x)
        x = x + shortcut  # Add the original input back

        return x