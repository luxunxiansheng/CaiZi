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

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dimension_input: int,
        dimension_embedding: int,
        block_size: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert (
            dimension_embedding % num_heads == 0
        ), "d_out must be divisible by n_heads"

        self.dimension_embedding = dimension_embedding
        self.num_heads = num_heads
        self.dimension_head = (
            dimension_embedding // num_heads
        )  # Reduce the projection dim to match desired output dim

        self.weights_query = nn.Linear(
            dimension_input, dimension_embedding, bias=qkv_bias
        )
        self.weights_key = nn.Linear(
            dimension_input, dimension_embedding, bias=qkv_bias
        )
        self.weights_value = nn.Linear(
            dimension_input, dimension_embedding, bias=qkv_bias
        )
        self.out_proj = nn.Linear(
            dimension_embedding, dimension_embedding
        )  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "mask", torch.triu(torch.ones(block_size, block_size), diagonal=1)
            )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.weights_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.weights_query(x)
        values = self.weights_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.dimension_head)
        values = values.view(b, num_tokens, self.num_heads, self.dimension_head)
        queries = queries.view(b, num_tokens, self.num_heads, self.dimension_head)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.flash:
            use_dropout = 0.0 if not self.training else self.dropout.p
            context_vec = nn.functional.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=None,
                dropout_p=use_dropout,
                is_causal=True,
            )
        else:

            # Compute scaled dot-product attention (aka self-attention) with a causal mask
            attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

            # Original mask truncated to the number of tokens and converted to boolean
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

            # Use the mask to fill attention scores
            attn_scores.masked_fill_(mask_bool, -torch.inf)

            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Shape: (b, num_tokens, num_heads, head_dim)
            context_vec = attn_weights @ values

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(b, num_tokens, self.dimension_embedding)
        )
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
