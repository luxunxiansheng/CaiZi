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
from math import e
import os

import torch
from torch.nn import Module
from torch.optim import Optimizer
from ray.train import Checkpoint

def save_checkpoint(model: Module, 
                    optimizer: Optimizer, 
                    epoch: int, 
                    temp_checkpoint_dir: str) -> Checkpoint:
    torch.save(
        model.state_dict(),  # NOTE: Unwrap the model.
        os.path.join(temp_checkpoint_dir, "model.pt"),
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(temp_checkpoint_dir, "optimizer.pt"),
    )
    torch.save(
        {"epoch": epoch},
        os.path.join(temp_checkpoint_dir, "extra_state.pt"),
    )


def resume_checkpoint(model: Module, optimizer: Optimizer,checkpoint:Checkpoint) -> int:
    with checkpoint.as_directory() as checkpoint_dir:
        model_state_dict = torch.load(
            os.path.join(checkpoint_dir, "model.pt"),
            # map_location=...,  # Load onto a different device if needed.
        )
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        )
        epoch_start = (
            torch.load(os.path.join(checkpoint_dir, "extra_state.pt"))["epoch"] + 1
        )
            
    return epoch_start
