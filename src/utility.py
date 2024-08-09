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

import os
import json

from typing import List
import urllib.request
from tqdm import tqdm

import numpy as np

import tensorflow as tf

import torch
from torch.nn import Module
from torch.optim import Optimizer
from ray.train import Checkpoint

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


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def _download_file(url, destination):
    # Send a GET request to download the file

    try:
        with urllib.request.urlopen(url) as response:
            # Get the total file size from headers, defaulting to 0 if not present
            file_size = int(response.headers.get("Content-Length", 0))

            # Check if file exists and has the same size
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return

            # Define the block size for reading the file
            block_size = 1024  # 1 Kilobyte

            # Initialize the progress bar with total file size
            progress_bar_description = os.path.basename(
                url
            )  # Extract filename from URL
            with tqdm(
                total=file_size,
                unit="iB",
                unit_scale=True,
                desc=progress_bar_description,
            ) as progress_bar:
                # Open the destination file in binary write mode
                with open(destination, "wb") as file:
                    # Read the file in chunks and write to destination
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))  # Update progress bar
    except urllib.error.HTTPError:
        s = (
            f"The specified URL ({url}) is incorrect, the internet connection cannot be established,"
            "\nor the requested file is temporarily unavailable.\nPlease visit the following website"
            " for help: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(s)


def download_gpt2_model(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        _download_file(file_url, file_path)



def load_gpt2_params(model_dir: str) -> dict:
    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)

    # Open and read the file using a context manager to ensure it is closed properly
    try:
        with open(os.path.join(model_dir, "hparams.json")) as file:
            settings: dict = json.load(file)
         
    except Exception as e:
        print(f'Error reading hparams.json: {e}')
    
    
    # Initialize parameters dictionary with empty blocks for each layer
    params: dict = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array: np.ndarray = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts: List[str] = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict: dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number: int = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key: str = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def load_weights_into_gpt(gpt: GPT, params: dict) -> None:
    gpt.position_embedding.weight = assign(gpt.position_embedding.weight, params["wpe"])
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params["wte"])

    for index in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][index]["attn"]["c_attn"])["w"],
            3,
            axis=-1,
        )
        gpt.transformer_blocks[index].attention.weights_query.weight = assign(
            gpt.transformer_blocks[index].attention.weights_query.weight, q_w.T
        )
        gpt.transformer_blocks[index].attention.weights_key.weight = assign(
            gpt.transformer_blocks[index].attention.weights_key.weight, k_w.T
        )
        gpt.transformer_blocks[index].attention.weights_value.weight = assign(
            gpt.transformer_blocks[index].attention.weights_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][index]["attn"]["c_attn"])["b"],
            3,
            axis=-1,
        )
        gpt.transformer_blocks[index].attention.weights_query.bias = assign(
            gpt.transformer_blocks[index].attention.weights_query.bias, q_b
        )
        gpt.transformer_blocks[index].attention.weights_key.bias = assign(
            gpt.transformer_blocks[index].attention.weights_key.bias, k_b
        )
        gpt.transformer_blocks[index].attention.weights_value.bias = assign(
            gpt.transformer_blocks[index].attention.weights_value.bias, v_b
        )

        gpt.transformer_blocks[index].attention.out_proj.weight = assign(
            gpt.transformer_blocks[index].attention.out_proj.weight,
            params["blocks"][index]["attn"]["c_proj"]["w"].T,
        )
        gpt.transformer_blocks[index].attention.out_proj.bias = assign(
            gpt.transformer_blocks[index].attention.out_proj.bias,
            params["blocks"][index]["attn"]["c_proj"]["b"],
        )

        gpt.transformer_blocks[index].mlp.fc.weight = assign(
            gpt.transformer_blocks[index].mlp.fc.weight,
            params["blocks"][index]["mlp"]["c_fc"]["w"].T,
        )
        gpt.transformer_blocks[index].mlp.fc.bias = assign(
            gpt.transformer_blocks[index].mlp.fc.bias,
            params["blocks"][index]["mlp"]["c_fc"]["b"],
        )
        gpt.transformer_blocks[index].mlp.projection.weight = assign(
            gpt.transformer_blocks[index].mlp.projection.weight,
            params["blocks"][index]["mlp"]["c_proj"]["w"].T,
        )
        gpt.transformer_blocks[index].mlp.projection.bias = assign(
            gpt.transformer_blocks[index].mlp.projection.bias,
            params["blocks"][index]["mlp"]["c_proj"]["b"],
        )

        gpt.transformer_blocks[index].layernorm_1.weight = assign(
            gpt.transformer_blocks[index].layernorm_1.weight,
            params["blocks"][index]["ln_1"]["g"],
        )
        gpt.transformer_blocks[index].layernorm_1.bias = assign(
            gpt.transformer_blocks[index].layernorm_1.bias,
            params["blocks"][index]["ln_1"]["b"],
        )
        gpt.transformer_blocks[index].layernorm_2.weight = assign(
            gpt.transformer_blocks[index].layernorm_2.weight,
            params["blocks"][index]["ln_2"]["g"],
        )
        gpt.transformer_blocks[index].layernorm_2.bias = assign(
            gpt.transformer_blocks[index].layernorm_2.bias,
            params["blocks"][index]["ln_2"]["b"],
        )

    gpt.final_layernorm.weight = assign(gpt.final_layernorm.weight, params["g"])
    gpt.final_layernorm.bias = assign(gpt.final_layernorm.bias, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
