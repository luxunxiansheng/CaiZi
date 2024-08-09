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
        
        self.final_layernorm = LayerNorm(dimension_embedding, qkv_bias)
        self.out_head = nn.Linear(
            dimension_embedding, vocab_size, bias=False
        )
        
        # Tie weights
        self.token_embedding.weight= self.out_head.weight
        
        self.apply(self._init_weights)
        
        # apply special scaled init to the residual projections, per GPT-2 paper
        for name, parameter in self.named_parameters():
            if name.endswith('projection.weight'):
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02/math.sqrt(2 * n_layers))


                # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding.weight.numel()
        return n_params    
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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