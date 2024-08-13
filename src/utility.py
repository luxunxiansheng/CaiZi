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

from calendar import c
import os

import urllib.request
from tqdm import tqdm


import torch
from torch.nn import Module
from torch.optim import Optimizer
from ray.train import Checkpoint

from transformers import GPT2LMHeadModel

from model.GPT import GPT


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


def assign_check(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())


def load_hf_weights_into_gpt(gpt: GPT, model_type: str) -> None:
    assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
    # only dropout can be overridden see more notes below

    print("loading weights from pretrained gpt: %s" % model_type)

    # n_layer, n_head and n_embd are determined from model_type
    config = {
        "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
        "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
        "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    }[model_type]
    print("forcing vocab_size=50257, block_size=1024, bias=True")
    config["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
    config["block_size"] = 1024  # always 1024 for GPT model checkpoints
    config["bias"] = True  # always True for GPT model checkpoints
    config["dropout"] = 0.0  # always 0.1 for GPT model checkpoints

    gpt = GPT(
        config["vocab_size"],
        config["n_embd"],
        config["block_size"],
        config["n_layer"],
        config["n_head"],
        config["dropout"],
        config["bias"],
    )


    # init a huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    d = model_hf.state_dict()

    gpt.transformer.position_embedding.weight = assign_check(
        gpt.transformer.position_embedding.weight, d["transformer.wpe.weight"]
    )
    gpt.transformer.token_embedding.weight = assign_check(
        gpt.transformer.token_embedding.weight, d["transformer.wte.weight"]
    )

    for tb_index in range(config["n_layer"]):
        gpt.transformer.transformer_blocks[tb_index].attention.attention.weight = (
            assign_check(
                gpt.transformer.transformer_blocks[tb_index].attention.attention.weight,
                d[f"transformer.h.{tb_index}.attn.c_attn.weight"].T,
            )
        )
        gpt.transformer.transformer_blocks[tb_index].attention.attention.bias = (
            assign_check(
                gpt.transformer.transformer_blocks[tb_index].attention.attention.bias,
                d[f"transformer.h.{tb_index}.attn.c_attn.bias"],
            )
        )
        gpt.transformer.transformer_blocks[tb_index].attention.out_proj.weight = (
            assign_check(
                gpt.transformer.transformer_blocks[tb_index].attention.out_proj.weight,
                d[f"transformer.h.{tb_index}.attn.c_proj.weight"].T,
            )
        )
        gpt.transformer.transformer_blocks[tb_index].attention.out_proj.bias = (
            assign_check(
                gpt.transformer.transformer_blocks[tb_index].attention.out_proj.bias,
                d[f"transformer.h.{tb_index}.attn.c_proj.bias"],
            )
        )

        gpt.transformer.transformer_blocks[tb_index].mlp.fc.weight = assign_check(
            gpt.transformer.transformer_blocks[tb_index].mlp.fc.weight,
            d[f"transformer.h.{tb_index}.mlp.c_fc.weight"].T,
        )
        gpt.transformer.transformer_blocks[tb_index].mlp.fc.bias = assign_check(
            gpt.transformer.transformer_blocks[tb_index].mlp.fc.bias,
            d[f"transformer.h.{tb_index}.mlp.c_fc.bias"],
        )
        gpt.transformer.transformer_blocks[tb_index].mlp.projection.weight = (
            assign_check(
                gpt.transformer.transformer_blocks[tb_index].mlp.projection.weight,
                d[f"transformer.h.{tb_index}.mlp.c_proj.weight"].T,
            )
        )
        gpt.transformer.transformer_blocks[tb_index].mlp.projection.bias = assign_check(
            gpt.transformer.transformer_blocks[tb_index].mlp.projection.bias,
            d[f"transformer.h.{tb_index}.mlp.c_proj.bias"],
        )

        gpt.transformer.transformer_blocks[tb_index].layernorm_1.weight = assign_check(
            gpt.transformer.transformer_blocks[tb_index].layernorm_1.weight,
            d[f"transformer.h.{tb_index}.ln_1.weight"],
        )
        gpt.transformer.transformer_blocks[tb_index].layernorm_1.bias = assign_check(
            gpt.transformer.transformer_blocks[tb_index].layernorm_1.bias,
            d[f"transformer.h.{tb_index}.ln_1.bias"],
        )

        gpt.transformer.transformer_blocks[tb_index].layernorm_2.weight = assign_check(
            gpt.transformer.transformer_blocks[tb_index].layernorm_2.weight,
            d[f"transformer.h.{tb_index}.ln_2.weight"],
        )
        gpt.transformer.transformer_blocks[tb_index].layernorm_2.bias = assign_check(
            gpt.transformer.transformer_blocks[tb_index].layernorm_2.bias,
            d[f"transformer.h.{tb_index}.ln_2.bias"],
        )

    gpt.transformer.final_layernorm.weight = assign_check(
        gpt.transformer.final_layernorm.weight, d["transformer.ln_f.weight"]
    )
    gpt.transformer.final_layernorm.bias = assign_check(
        gpt.transformer.final_layernorm.bias, d["transformer.ln_f.bias"]
    )

    gpt.out_head.weight = assign_check(gpt.out_head.weight, d["transformer.wte.weight"])
