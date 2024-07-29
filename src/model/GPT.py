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
from model.transformer_block import TransformerBlock

class GPT(nn.Module):
    def __init__(self, vocab_size: int, dimension_embedding: int, block_size: int, n_layers: int, num_header: int, drop_rate: float, qkv_bias: bool = False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dimension_embedding)
        self.position_embedding = nn.Embedding(block_size, dimension_embedding)
        self.drop = nn.Dropout(drop_rate)
        
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(dimension_embedding, dimension_embedding, block_size, num_header, drop_rate, qkv_bias) for _ in range(n_layers)])
        
        self.final_layernorm = LayerNorm(dimension_embedding)
        self.out_head = nn.Linear(
            dimension_embedding, vocab_size, bias=False
        )

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        token_embeds = self.token_embedding(in_idx)
        position_embeds = self.position_embedding(torch.arange(seq_len, device=in_idx.device))
        x = token_embeds + position_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop(x)
        x = self.transformer_blocks(x)
        x = self.final_layernorm(x)
        logits = self.out_head(x)
        return logits