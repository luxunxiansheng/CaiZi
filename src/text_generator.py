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
from torch.nn import functional as F

from model.GPT import GPT
from token_processor import TokenProcessor


class TextGenerator:
    def __init__(self, model: GPT, tokenproessor:TokenProcessor,device: torch.device):
        self.model = model.to(device)
        self.device = device

        self.tokenizer = tokenproessor

    def __call__(
        self,
        idx,
        max_new_tokens,
        block_size,
        temperature=1.0,
        top_k=None,        
    ) -> str:
        for _ in range(max_new_tokens):
            idx = idx.to(self.device)
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self.model(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        
        flat = idx.squeeze(0)
        decoded_text = self.tokenizer.decode(flat.tolist())
        return decoded_text
