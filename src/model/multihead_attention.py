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

import math

import torch
import torch.nn as nn

from torch.nn import functional as F



class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dimension_embedding: int,
        block_size: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert (
            dimension_embedding % num_heads == 0
        ), "d_out must be divisible by n_heads"
        
        self.attention = nn.Linear(dimension_embedding, dimension_embedding * 3, bias=bias)
        self.out_proj = nn.Linear(dimension_embedding, dimension_embedding,bias=bias)  # Linear layer to combine head outputs
        
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        
        self.dropout = dropout

        self.num_heads = num_heads
        self.dimension_embedding = dimension_embedding
        
       
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )

            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x):
        batch_size, block_size, embedding_size = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.attention(x).split(self.dimension_embedding, dim=2)
        k = k.view(batch_size, block_size, self.num_heads, embedding_size // self.num_heads).transpose(1, 2) # (Batchs, number_head, block_size, dimension_head)
        q = q.view(batch_size, block_size, self.num_heads, embedding_size // self.num_heads).transpose(1, 2) # (Batchs, number_head, block_size, dimension_head)
        v = v.view(batch_size, block_size, self.num_heads, embedding_size // self.num_heads).transpose(1, 2) # (Batchs, number_head, block_size, dimension_head)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attention = attention.masked_fill(self.bias[:,:,:block_size,:block_size] == 0, float('-inf'))
            attention = F.softmax(attention, dim=-1)
            attention = self.attention_dropout(attention)
            y = attention @ v 
        y = y.transpose(1, 2).contiguous().view(batch_size, block_size, embedding_size) # re-assemble all head outputs side by side

        # output projection
        y = self.residual_dropout(self.out_proj(y))
        return y
