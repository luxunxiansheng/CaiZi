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

from typing import List
import math

from torch.optim.lr_scheduler import _LRScheduler


class GPTLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float=3e-4,
        min_lr: float=3e-5,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(GPTLRScheduler, self).__init__(optimizer)

    def get_lr(self)->List[float]:
        # 1) linear warmup for warmup_iters steps
        if self.last_epoch < self.warmup_steps:
            return [self.max_lr * (self.last_epoch+1) / self.warmup_steps for _ in self.optimizer.param_groups]
        # 2) if it > lr_decay_iters, return min learning rate
        if self.last_epoch > self.max_steps :
            return [self.min_lr for _ in self.optimizer.param_groups]
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return [self.min_lr + coeff * (self.max_lr -self.min_lr) for _ in self.optimizer.param_groups]

        
        

