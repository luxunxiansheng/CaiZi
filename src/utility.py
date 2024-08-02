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


def save_checkpoint(
    model: Module,
    optimizer: Optimizer,
    epoch: int,
    perplexity: float,
    best_checkpoint_dir: str,
):

    torch.save(
        model.state_dict(),  # NOTE: Unwrap the model.
        os.path.join(best_checkpoint_dir, "model.pt"),
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(best_checkpoint_dir, "optimizer.pt"),
    )
    torch.save(
        {
            "epoch": epoch,
            "perplexity": perplexity,

        },
        os.path.join(best_checkpoint_dir, "extra_state.pt"),
    )


def resume_checkpoint(
    model: Module, optimizer: Optimizer, checkpoint: Checkpoint
) -> tuple[int, float]:
    with checkpoint.as_directory() as checkpoint_dir:
        model_state_dict = torch.load(
            os.path.join(checkpoint_dir, "model.pt"),
            # map_location=...,  # Load onto a different device if needed.
        )
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        )
        extra_state = torch.load(os.path.join(checkpoint_dir, "extra_state.pt"))

    return (
        extra_state["epoch"],
        extra_state["perplexity"],

    )
