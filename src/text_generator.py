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

from model.GPT import GPT
from token_processor import TikTokenizer


class TextGenerator:
    def __init__(self,model:GPT):
        self.model = model
        self.tokenizer = TikTokenizer()
        
        
    def __call__(self, idx, max_new_tokens, context_size) -> str:
        self.model.eval()
        for _ in range(max_new_tokens):
            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = idx[:, -context_size:]
            
            # Get the predictions
            with torch.no_grad():
                logits = self.model(idx_cond)
            
            # Focus only on the last time step
            # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]  

            # Apply softmax to get probabilities
            probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

            # Get the idx of the vocab entry with the highest probability value
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)
        
        decoded_text = self.tokenizer.decode(idx[0].tolist())
        decoded_text = decoded_text.replace("\n", " ")
        return decoded_text

        
       