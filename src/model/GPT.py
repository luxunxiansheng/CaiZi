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
    def __init__(
        self,
        vocab_size: int,
        dimension_embedding: int,
        block_size: int,
        n_layers: int,
        num_header: int,
        drop_rate: float,
        bias: bool = False,
    ):
        super().__init__()
        
        self.num_layers = n_layers
        self.num_head = num_header
        self.dimension_embedding = dimension_embedding
        self.block_size = block_size

        self.transformer = nn.ModuleDict(
            dict(
                token_embedding=nn.Embedding(vocab_size, dimension_embedding),
                position_embedding=nn.Embedding(block_size, dimension_embedding),
                drop=nn.Dropout(drop_rate),
                transformer_blocks=nn.ModuleList(
                    [
                        TransformerBlock(
                            dimension_embedding,
                            block_size,
                            num_header,
                            drop_rate,
                            bias,
                        )
                        for _ in range(n_layers)
                    ]
                ),
                final_layernorm=LayerNorm(dimension_embedding, bias),
            )
        )

        self.out_head = nn.Linear(dimension_embedding, vocab_size, bias=False)

        self.transformer.token_embedding.weight = (
            self.out_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for name, parameter in self.named_parameters():
            if name.endswith("out_proj.weight"):
                torch.nn.init.normal_(
                    parameter, mean=0.0, std=0.02 / math.sqrt(2 * n_layers)
                )

                # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            n_params -= self.transformer.position_embedding.weight.numel()
        return n_params
    
    def estimate_mfu(self, num_tokens_per_second, flops_promised=125e12):
        """ estimate model flops utilization (MFU) in units 4090 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        n_parameters = self.get_num_params()
        
        Q = self.dimension_embedding//self.num_head
        flops_per_token = 6*n_parameters + 12*self.num_layers*self.num_head*Q*self.block_size
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_token * num_tokens_per_second 
       
        mfu = flops_achieved / flops_promised
        return mfu

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        device = idx.device
        _, block_size = idx.size()

        position = torch.arange(
            0, block_size, dtype=torch.long, device=device
        )  # shape (block_size)

        # forward the GPT model itself
        token_embedding = self.transformer.token_embedding(
            idx
        )  # token embeddings of shape (batch_size, block_size, embedding_size)
        position_embedding = self.transformer.position_embedding(
            position
        )  # position embeddings of shape (block_size, embedding_size)
        x = self.transformer.drop(token_embedding + position_embedding)
        for transformer_block in self.transformer.transformer_blocks:
            x = transformer_block(x)
        x = self.transformer.final_layernorm(x)
        logits = self.out_head(x)  
        return logits
