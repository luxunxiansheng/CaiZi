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
from abc import ABC, abstractmethod
from pathlib import Path


from attr import validate
import ray.train
import ray.train.torch
import torch
import torchmetrics

import ray

from document_processor import TextDocumentProcessor
from text_generator import TextGenerator
from token_processor import TikTokenizer
from chunk_processor import ChunkProcessor
from model.GPT import GPT
from utility import resume_checkpoint, save_checkpoint


class FundationModelTrainer(ABC):
    @abstractmethod
    def self_train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class RayGPT2FundationModelTrainer(FundationModelTrainer):
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_chunked_tokens = None
        self.validate_chunked_tokens = None

    @staticmethod
    def data_preprocess(
        dataset,
        block_size,
        stride,
    ):
        data_sources = [Path(item["path"]) for item in dataset]
        text_document_paths = ray.data.from_items(data_sources)

        train_text_document_processor = TextDocumentProcessor(section="train")
        train_texts = text_document_paths.map(train_text_document_processor)

        validate_text_document_processor = TextDocumentProcessor(section="validate")
        validate_texts = text_document_paths.map(validate_text_document_processor)

        tokenizer = TikTokenizer()
        train_tokens = train_texts.map(tokenizer)
        validate_tokens = validate_texts.map(tokenizer)

        chunk_processor = ChunkProcessor(max_length=block_size, stride=stride)
        train_chunked_tokens = train_tokens.flat_map(chunk_processor)
        validate_chunked_tokens = validate_tokens.flat_map(chunk_processor)

        return train_chunked_tokens, validate_chunked_tokens

    def self_train(self):
        train_loop_config = {
            "vocab_size": self.cfg["124M"]["vocab_size"],
            "dimension_embedding": self.cfg["124M"]["dimension_embedding"],
            "block_size": self.cfg["124M"]["block_size"],
            "num_layers": self.cfg["124M"]["num_layers"],
            "num_headers": self.cfg["124M"]["num_headers"],
            "drop_rate": self.cfg["124M"]["drop_rate"],
            "qkv_bias": self.cfg["124M"]["qkv_bias"],
            "check_frequency": self.cfg["ray_train"]["check_frequency"],
            "batch_size_per_worker": self.cfg["ray_train"]["batch_size_per_worker"],
            "num_epoch_per_worker": self.cfg["ray_train"]["num_epoch_per_worker"],
            "resume_training": self.cfg["ray_train"]["resume_training"],
            "best_checkpoint_dir": self.cfg["ray_train"]["best_checkpoint_dir"],
            "start_context": self.cfg["ray_train"]["start_context"],
        }

        dataset = self.cfg["dataset"]
        block_size = self.cfg["124M"]["block_size"]
        stride = self.cfg["124M"]["stride"]

        train_chunked_tokens, validate_chunked_tokens = (
            RayGPT2FundationModelTrainer.data_preprocess(
                dataset=dataset, block_size=block_size, stride=stride
            )
        )

        trainer = ray.train.torch.TorchTrainer(
            train_loop_per_worker=RayGPT2FundationModelTrainer._train_loop_per_worker,
            train_loop_config=train_loop_config,
            datasets={
                "train": train_chunked_tokens,
                "validate": validate_chunked_tokens,
            },
            dataset_config=ray.train.DataConfig(
                datasets_to_split=["train"],
            ),
            scaling_config=ray.train.ScalingConfig(
                num_workers=self.cfg["ray_train"]["num_workers"],
                use_gpu=self.cfg["ray_train"]["use_gpu"],
                resources_per_worker={
                    "CPU": self.cfg["ray_train"]["num_cpus_per_worker"],
                    "GPU": self.cfg["ray_train"]["num_gpus_per_worker"],
                },
            ),
            run_config=ray.train.RunConfig(
                storage_path=self.cfg["ray_train"]["storage_path"],
                name=self.cfg["ray_train"]["name"],
            ),
        )
        result = trainer.fit()
        print(result)

    def evaluate(self):
        pass

    def _start_ray(self):
        os.environ["RAY_DEDUP_LOGS"] = "0"
        os.environ["RAY_COLOR_PREFIX"] = "1"

        if ray.is_initialized():
            ray.shutdown()

        ray.init(
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": "$PYTHONPATH:" + self.cfg.project_root + "/src",
                    "RAY_DATA_VERBOSE_PROGRESS": "1",
                    "RAY_DEBUG": "1",
                },
                "working_dir": self.cfg.project_root,
                "excludes": [
                    "/bazel-*",
                    ".git",
                    "*.pyc",
                    "/__pycache__",
                    "/outputs",
                    "/model",
                ],
            },
            ignore_reinit_error=True,
        )

        # convience for debugging
        ray.data.DataContext.get_current().execution_options.verbose_progress = True
        ray.data.DataContext.log_internal_stack_trace_to_stdout = True

    @staticmethod
    def _train_loop_per_worker(cfg):
        vocab_size = cfg["vocab_size"]
        dimension_embedding = cfg["dimension_embedding"]
        block_size = cfg["block_size"]
        num_layers = cfg["num_layers"]
        num_headers = cfg["num_headers"]
        drop_rate = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        check_frequency = cfg["check_frequency"]
        batch_size_per_worker = cfg["batch_size_per_worker"]
        num_epoch_per_worker = cfg["num_epoch_per_worker"]
        resume_training = cfg["resume_training"]
        best_checkpoint_dir = cfg["best_checkpoint_dir"]
        start_context = cfg["start_context"]

        # GPT model
        model = GPT(
            vocab_size,
            dimension_embedding,
            block_size,
            num_layers,
            num_headers,
            drop_rate,
            qkv_bias,
        )
        model = ray.train.torch.prepare_model(model)
        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

        # ====== Resume training state from the checkpoint. ======
        epoch_start = 0
        best_perplexity = float("inf")
        best_epoch = 0

        if resume_training:
            if os.path.exists(best_checkpoint_dir):
                checkpoint = ray.train.Checkpoint.from_directory(best_checkpoint_dir)
            else:
                checkpoint = None
            if checkpoint:
                best_epoch, best_perplexity = resume_checkpoint(
                    model, optimizer, checkpoint
                )
                epoch_start = best_epoch
                print(
                    f"Resumed training from best_epoch {best_epoch},best_perplexity {best_perplexity}"
                )
            else:
                print(f"Checkpoint not found, starting from epoch 0")

        # loss function
        loss_function = torch.nn.CrossEntropyLoss()

        rank = ray.train.get_context().get_world_rank()
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # metrics
        metric = torchmetrics.text.Perplexity().to(device)

        # data
        train_data_shard = ray.train.get_dataset_shard("train")
        validate_data_shard = ray.train.get_dataset_shard("validate")

        report_metrics = {
            "epoch": 0,
            "train_loss": 0.0,
            "validate_loss": 0.0,
            "perplexity": 0.0,
            "best_epoch": best_epoch,
            "best_perplexity": best_perplexity,
        }

        text_generator = TextGenerator(model, device=device)
        tokenizer = TikTokenizer()

        for epoch in range(epoch_start + 1, num_epoch_per_worker + 1):
            model.train()

            report_metrics["epoch"] = epoch

            train_loss = 0
            batch_count = 0
            for batch in train_data_shard.iter_torch_batches(batch_size=batch_size_per_worker,drop_last=True,local_shuffle_buffer_size=1000,):
                batch_count += 1
                input_ids = batch["input_ids"]
                logits = model(input_ids)
                target_ids = batch["target_ids"]
                loss = loss_function(logits.flatten(0, 1), target_ids.flatten())
                train_loss += loss.item()  # only for reporting
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss = train_loss / batch_count
            report_metrics["train_loss"] = train_loss

            validate_loss = 0
            if epoch % check_frequency == 0:

                model.eval()

                with torch.no_grad():
                    batch_count = 0
                    for batch in validate_data_shard.iter_torch_batches(
                        batch_size=1,
                        drop_last=False,
                    ):
                        batch_count += 1
                        input_ids = batch["input_ids"]
                        logits = model(input_ids)
                        target_ids = batch["target_ids"]
                        loss = loss_function(logits.flatten(0, 1), target_ids.flatten())
                        validate_loss += loss.item()  # only for reporting
                        metric.update(logits, target_ids)

                validate_loss = validate_loss / batch_count
                perplexity = metric.compute().item()
                metric.reset()

                report_metrics["validate_loss"] = validate_loss
                report_metrics["perplexity"] = perplexity

                #In standard DDP training, where the model is the same across all ranks,
                # only the global rank 0 worker needs to save and report the checkpoint
                if ray.train.get_context().get_world_rank() == 0:
                    if perplexity < best_perplexity:
                        best_perplexity = perplexity
                        best_epoch = epoch

                        report_metrics["best_epoch"] = best_epoch
                        report_metrics["best_perplexity"] = best_perplexity

                        # create the best_checkpoint_dir if it does not exist
                        if not os.path.exists(best_checkpoint_dir):
                            os.makedirs(best_checkpoint_dir)

                        save_checkpoint(
                            model,
                            optimizer,
                            epoch,
                            perplexity,
                            best_checkpoint_dir,
                        )

                ray.train.report(metrics=report_metrics)

            decoded = text_generator(
                tokenizer.encode(start_context),
                max_new_tokens=50,
                context_size=block_size,
            )
            print(f"\n epoch:{epoch}: {decoded}")
