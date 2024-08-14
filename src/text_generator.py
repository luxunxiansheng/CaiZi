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
    def __init__(self, model: GPT, device: torch.device):
        self.model = model.to(device)
        self.device = device

        self.tokenizer = TikTokenizer()

    def __call__(
        self,
        idx,
        max_new_tokens,
        context_size,
        temperature=0.0,
        top_k=None,
        eos_id=None,
    ) -> str:
        
        self.model.eval()
        for _ in range(max_new_tokens):
            idx = idx.to(self.device)
            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = idx[:, -context_size:]

            # Get the predictions
            with torch.no_grad():
                with torch.autocast(device_type=idx_cond.device.type, dtype=torch.bfloat16):
                    logits = self.model(idx_cond)

            # Focus only on the last time step
            # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]

            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf")).to(logits.device),
                    logits,
                )

            #  Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # Otherwise same as before: get idx of the vocab entry with the highest logits value
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if (
                idx_next == eos_id
            ):  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

            idx = idx.cpu()
        
        flat = idx.squeeze(0)
        decoded_text = self.tokenizer.decode(flat.tolist())
        return decoded_text
