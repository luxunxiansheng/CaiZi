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

from collections import OrderedDict
import os


import torch
from torch.nn import Module
from torch.optim import Optimizer

from torch.amp import GradScaler

from ray.train import Checkpoint

from transformers import GPT2LMHeadModel

from model.GPT import GPT


def save_checkpoint(
    model: Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    epoch: int,
    perplexity: float,
    best_checkpoint_dir: str,
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "perplexity": perplexity,
    }

    torch.save(
        checkpoint,
        os.path.join(best_checkpoint_dir, "checkpoint.pt"),
    )


def load_checkpoint(
    model: Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    checkpoint_dir: str,
    device: str,
) -> tuple[int, float]:
    
    epoch = 0
    perplexity = float("inf")
 
    if os.path.exists(checkpoint_dir):
        checkpoint = Checkpoint.from_directory(checkpoint_dir)
    else:
        checkpoint = None

    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(
                os.path.join(checkpoint_dir, "checkpoint.pt"), map_location=device
            )

            model.load_state_dict(checkpoint_dict["model"])
            optimizer.load_state_dict(checkpoint_dict["optimizer"])
            scaler.load_state_dict(checkpoint_dict["scaler"])
            epoch = checkpoint_dict["epoch"] if "epoch" in checkpoint_dict else 0
            perplexity = (
                checkpoint_dict["perplexity"]
                if "perplexity" in checkpoint_dict
                else float("inf")
            )

    return (epoch,perplexity)




def load_model_from_checkpoint(
    model: Module,
    best_checkpoint_dir: str,
    device: str,
) -> bool:
    loaded=False
        
    if os.path.exists(best_checkpoint_dir):
        checkpoint = Checkpoint.from_directory(best_checkpoint_dir)
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"), map_location=device)["model"]
 
                                # Create a new state_dict without 'module.' prefix
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    name = k.replace('_orig_mod.', '')  # remove 'module.' prefix
                    new_state_dict[name] = v
                
                model.load_state_dict(new_state_dict)
        

                
                model.load_state_dict(new_state_dict)
                
                loaded=True
    
    return loaded
    



def _assign_check(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())


def load_hf_weights_into_gpt(gpt: GPT, model_type: str) -> None:
    assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
    # only dropout can be overridden see more notes below

    print("loading weights from pretrained gpt: %s" % model_type)

    num_layer = len(gpt.transformer.transformer_blocks)

    # init a huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    d = model_hf.state_dict()

    gpt.transformer.position_embedding.weight = _assign_check(
        gpt.transformer.position_embedding.weight, d["transformer.wpe.weight"]
    )
    gpt.transformer.token_embedding.weight = _assign_check(
        gpt.transformer.token_embedding.weight, d["transformer.wte.weight"]
    )

    for tb_index in range(num_layer):
        gpt.transformer.transformer_blocks[tb_index].attention.attention.weight = (
            _assign_check(
                gpt.transformer.transformer_blocks[tb_index].attention.attention.weight,
                d[f"transformer.h.{tb_index}.attn.c_attn.weight"].T,
            )
        )
        gpt.transformer.transformer_blocks[tb_index].attention.attention.bias = (
            _assign_check(
                gpt.transformer.transformer_blocks[tb_index].attention.attention.bias,
                d[f"transformer.h.{tb_index}.attn.c_attn.bias"],
            )
        )
        gpt.transformer.transformer_blocks[tb_index].attention.out_proj.weight = (
            _assign_check(
                gpt.transformer.transformer_blocks[tb_index].attention.out_proj.weight,
                d[f"transformer.h.{tb_index}.attn.c_proj.weight"].T,
            )
        )
        gpt.transformer.transformer_blocks[tb_index].attention.out_proj.bias = (
            _assign_check(
                gpt.transformer.transformer_blocks[tb_index].attention.out_proj.bias,
                d[f"transformer.h.{tb_index}.attn.c_proj.bias"],
            )
        )

        gpt.transformer.transformer_blocks[tb_index].mlp.fc.weight = _assign_check(
            gpt.transformer.transformer_blocks[tb_index].mlp.fc.weight,
            d[f"transformer.h.{tb_index}.mlp.c_fc.weight"].T,
        )
        gpt.transformer.transformer_blocks[tb_index].mlp.fc.bias = _assign_check(
            gpt.transformer.transformer_blocks[tb_index].mlp.fc.bias,
            d[f"transformer.h.{tb_index}.mlp.c_fc.bias"],
        )
        gpt.transformer.transformer_blocks[tb_index].mlp.projection.weight = (
            _assign_check(
                gpt.transformer.transformer_blocks[tb_index].mlp.projection.weight,
                d[f"transformer.h.{tb_index}.mlp.c_proj.weight"].T,
            )
        )
        gpt.transformer.transformer_blocks[tb_index].mlp.projection.bias = _assign_check(
            gpt.transformer.transformer_blocks[tb_index].mlp.projection.bias,
            d[f"transformer.h.{tb_index}.mlp.c_proj.bias"],
        )

        gpt.transformer.transformer_blocks[tb_index].layernorm_1.weight = _assign_check(
            gpt.transformer.transformer_blocks[tb_index].layernorm_1.weight,
            d[f"transformer.h.{tb_index}.ln_1.weight"],
        )
        gpt.transformer.transformer_blocks[tb_index].layernorm_1.bias = _assign_check(
            gpt.transformer.transformer_blocks[tb_index].layernorm_1.bias,
            d[f"transformer.h.{tb_index}.ln_1.bias"],
        )

        gpt.transformer.transformer_blocks[tb_index].layernorm_2.weight = _assign_check(
            gpt.transformer.transformer_blocks[tb_index].layernorm_2.weight,
            d[f"transformer.h.{tb_index}.ln_2.weight"],
        )
        gpt.transformer.transformer_blocks[tb_index].layernorm_2.bias = _assign_check(
            gpt.transformer.transformer_blocks[tb_index].layernorm_2.bias,
            d[f"transformer.h.{tb_index}.ln_2.bias"],
        )

    gpt.transformer.final_layernorm.weight = _assign_check(
        gpt.transformer.final_layernorm.weight, d["transformer.ln_f.weight"]
    )
    gpt.transformer.final_layernorm.bias = _assign_check(
        gpt.transformer.final_layernorm.bias, d["transformer.ln_f.bias"]
    )

    gpt.out_head.weight = _assign_check(gpt.out_head.weight, d["transformer.wte.weight"])
