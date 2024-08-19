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
import inspect
from pathlib import Path

import time

import ray.train
import ray.train.torch
import torch
import torchmetrics

import ray

from document_processor import TextDocumentProcessor
from chunk_processor import ChunkProcessor
from token_processor import TokenProcessor
from model.GPT import GPT
from model.gpt_lr_scheduler import GPTLRScheduler
import  utility 


class FundationModelTrainer(ABC):
    @abstractmethod
    def self_supervised_train(self):
        pass

class RayGPT2FundationModelTrainer(FundationModelTrainer):
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_chunked_tokens = None
        self.validate_chunked_tokens = None
 
    def start_ray(self):
        os.environ["RAY_DEDUP_LOGS"] = "1"
        os.environ["RAY_COLOR_PREFIX"] = "1"
        os.environ["RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING"] = "1"

        if ray.is_initialized():
            ray.shutdown()

        ray.init(
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": "$PYTHONPATH:" + self.cfg.project_root + "/src",
                    "RAY_DATA_VERBOSE_PROGRESS": "1",
                    #"RAY_DEBUG": "0",
                    "PYTHONMALLOC": "malloc",
                    
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
            _metrics_export_port=8080,
        )

        # convience for debugging
        ray.data.DataContext.get_current().execution_options.verbose_progress = False
        ray.data.DataContext.log_internal_stack_trace_to_stdout = False
    
    def stop_ray(self): 
        ray.shutdown()    
        
    def data_preprocess(self):
        dataset = self.cfg["dataset"]
        data_sources = [Path(item["path"]) for item in dataset]
        text_document_paths = ray.data.from_items(data_sources)
        train_ratio = self.cfg["ray_data"]["train_ratio"]

        train_text_document_processor = TextDocumentProcessor(section="train",train_ratio=train_ratio)
        train_texts = text_document_paths.map(train_text_document_processor)

        validate_text_document_processor = TextDocumentProcessor(section="validate",train_ratio=train_ratio)
        validate_texts = text_document_paths.map(validate_text_document_processor)

        tokenizer_class = TokenProcessor.create(self.cfg['ray_data']['tokenizer_class']['name'])
        tokenizer_args =  self.cfg['ray_data']['tokenizer_class']['args']
        tokenizer= tokenizer_class(**tokenizer_args)
        train_tokens = train_texts.map(tokenizer)
        validate_tokens = validate_texts.map(tokenizer)
        
        block_size = self.cfg["model"]["block_size"]
        stride = self.cfg["model"]["stride"]
        chunk_processor = ChunkProcessor(block_size=block_size, stride=stride)
        self.train_chunked_tokens = train_tokens.flat_map(chunk_processor)
        self.validate_chunked_tokens = validate_tokens.flat_map(chunk_processor)

     
    def self_supervised_train(self):
        train_loop_config = {
            "vocab_size": self.cfg["model"]["vocab_size"],
            "dimension_embedding": self.cfg["model"]["dimension_embedding"],
            "block_size": self.cfg["model"]["block_size"],
            "num_layers": self.cfg["model"]["num_layers"],
            "num_headers": self.cfg["model"]["num_headers"],
            "drop_rate": self.cfg["model"]["drop_rate"],
            "bias": self.cfg["model"]["bias"],
            "check_frequency": self.cfg["ray_train"]["check_frequency"],
            "gradient_accumulation_steps":self.cfg["ray_train"]["gradient_accumulation_steps"],
            "physical_training_batch_size_per_worker": self.cfg["ray_train"]["physical_training_batch_size_per_worker"],
            "physical_validate_batch_size_per_worker": self.cfg["ray_train"]["physical_validate_batch_size_per_worker"],
            "num_epoch_per_worker": self.cfg["ray_train"]["num_epoch_per_worker"],
            "resume_training": self.cfg["ray_train"]["resume_training"],
            "best_checkpoint_dir": self.cfg["ray_train"]["best_checkpoint_dir"],
            "latest_checkpoint_dir": self.cfg["ray_train"]["latest_checkpoint_dir"],
            "start_context": self.cfg["ray_train"]["start_context"],
            "warmup_steps": self.cfg["ray_train"]["warmup_steps"],
            "max_steps": self.cfg["ray_train"]["max_steps"],
            "max_lr": self.cfg["ray_train"]["max_lr"],
            "min_lr": self.cfg["ray_train"]["min_lr"],
            "beta1": self.cfg["ray_train"]["beta1"],
            "beta2": self.cfg["ray_train"]["beta2"],
            "decay_lr": self.cfg["ray_train"]["decay_lr"],
            "weight_decay": self.cfg["ray_train"]["weight_decay"],
            "data_type": self.cfg["ray_train"]["data_type"],
          
        }
        trainer = ray.train.torch.TorchTrainer(
            train_loop_per_worker=RayGPT2FundationModelTrainer._train_workload_per_worker,
            train_loop_config=train_loop_config,
            datasets={
                "train": self.train_chunked_tokens,
                "validate": self.validate_chunked_tokens,
            },
            dataset_config=ray.train.DataConfig(
                datasets_to_split=["train"], # only split the train dataset into shards
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



    @staticmethod
    def _train_workload_per_worker(cfg):
        vocab_size = cfg["vocab_size"]
        dimension_embedding = cfg["dimension_embedding"]
        block_size = cfg["block_size"]
        num_layers = cfg["num_layers"]
        num_headers = cfg["num_headers"]
        drop_rate = cfg["drop_rate"]
        bias = cfg["bias"]
        check_frequency = cfg["check_frequency"]
        gradient_accumulation_steps = cfg["gradient_accumulation_steps"]
        physical_training_batch_size_per_worker = cfg["physical_training_batch_size_per_worker"]
        physical_validate_batch_size_per_worker = cfg["physical_validate_batch_size_per_worker"]
        num_epoch_per_worker = cfg["num_epoch_per_worker"]
        resume_training = cfg["resume_training"]
        best_checkpoint_dir = cfg["best_checkpoint_dir"]
        latest_checkpoint_dir = cfg["latest_checkpoint_dir"]
        warmup_steps = cfg["warmup_steps"]
        max_steps = cfg["max_steps"]
        max_lr = cfg["max_lr"]
        min_lr = cfg["min_lr"]
        beta1 = cfg["beta1"]
        beta2 = cfg["beta2"]
        decay_lr = cfg["decay_lr"]
        weight_decay = cfg["weight_decay"]
        data_type = cfg["data_type"]
        

        floating_point_precision = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[data_type]

        rank = ray.train.get_context().get_world_rank()
        device =ray.train.torch.get_device()
        use_amp = (floating_point_precision=="float16")

        torch.manual_seed(1337 + rank)
        torch.set_float32_matmul_precision("high")

        # data
        train_data_shard, validate_data_shard = (
            RayGPT2FundationModelTrainer._prepare_data()
        )

        # GPT model
        model = RayGPT2FundationModelTrainer._prepare_model(
            vocab_size,
            dimension_embedding,
            block_size,
            num_layers,
            num_headers,
            drop_rate,
            bias,
        )

        # optimizer
        optimizer = RayGPT2FundationModelTrainer._prepare_optimizer(
            weight_decay, max_lr,beta1,beta2,model, device
        )

        # initialize a GradScaler. Enable AMP for float16.
        # According to the pytorch documentation:
        #     "When entering an autocast-enabled region, Tensors may be any type. 
        #     You should not call half() or bfloat16() on your model(s) or inputs 
        #     when using autocasting"
        scaler = RayGPT2FundationModelTrainer._prepare_gradient_scaler(use_amp=use_amp)

        # lr scheduler
        lr_scheduler = RayGPT2FundationModelTrainer._prepare_lr_scheduler(
            warmup_steps, max_steps, max_lr, min_lr, decay_lr,optimizer
        )

        # loss function
        loss_function = RayGPT2FundationModelTrainer._prepare_loss_function()

        # metrics
        metric = RayGPT2FundationModelTrainer._prepare_metric(device)

        # ====== Resume training state from the checkpoint. ======
        epoch_start = 0
        best_perplexity = float("inf")
        best_epoch = 0

        if resume_training:
            epoch,perplexity  = (
                RayGPT2FundationModelTrainer._resume_training(latest_checkpoint_dir, 
                                                              model, 
                                                              optimizer,
                                                              scaler,
                                                              device)
                )
            
            print(f"Resuming training from epoch {epoch} with  perplexity {perplexity}")
            
            epoch_start = epoch
            

        report_metrics = {
            
            "rank": rank,
            "global_step": 0,
            "epoch": epoch_start,
            
            "token_total": 0,  # total tokens processed
            "token_process_time_ms": 0.0, # time in ms
            "token_per_second": 0.0,    # speed 
        
            "epoch_train_loss": 0.0,
            "epoch_batch_count": 0,
        
            "norm": 0.0,
            "learning_rate": 0.0,
            
            "validate_batch_count": 0,
            "validate_loss": 0.0,
            "perplexity": 0.0,
            
            "best_epoch": best_epoch,
            "best_perplexity": best_perplexity,
            "validate_loss_at_best_perplexity": 0.0,
        }

        logical_batch_size_per_worker = physical_training_batch_size_per_worker * gradient_accumulation_steps
        
        print(f"total_tokens_per_logical_batch_per_worker: {logical_batch_size_per_worker*block_size}")
        
        global_step = 0
        
        for epoch in range(epoch_start + 1, num_epoch_per_worker + 1):
            current_rank = ray.train.get_context().get_world_rank()
            report_metrics["rank"] = current_rank
            report_metrics["epoch"] = epoch
        
            token_processed = 0
            epoch_train_loss = 0
            epoch_batch_count = 0
            t0 = time.time()

            model.train()
            for logical_batch in train_data_shard.iter_torch_batches(batch_size=logical_batch_size_per_worker,drop_last=False,local_shuffle_buffer_size=1000,):
             
                logical_input_ids = logical_batch["input_ids"]
                logical_target_ids = logical_batch["target_ids"]
                
                for step in range(gradient_accumulation_steps):
                    start_index = step * physical_training_batch_size_per_worker
                    end_index = start_index + physical_training_batch_size_per_worker
                    
                    physical_input_ids_in_current_step = logical_input_ids[start_index : end_index]
                    physical_target_ids_in_current_step = logical_target_ids[start_index : end_index]
                    
                    # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
                    with torch.autocast(device_type=device.type,dtype=floating_point_precision,enabled=use_amp,):
                        logits = model(physical_input_ids_in_current_step)
                        physical_loss = loss_function(logits.flatten(0, 1),physical_target_ids_in_current_step.flatten(),)
                        physical_loss = (physical_loss / gradient_accumulation_steps)  # normalize the loss to account for the gradient accumulation

                    
                    # require_backward_grad_sync is set to True for the last step in the gradient accumulation  
                    # to speed up the training process by reducing the synchronization overhead across workers.
                    model.require_backward_grad_sync = (step == gradient_accumulation_steps - 1)
                    
                    # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                    # Backward passes under autocast are not recommended.
                    # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                    scaler.scale(physical_loss).backward()

                    # for reporting
                    epoch_train_loss += (physical_loss.detach().item())
                    epoch_batch_count += 1
                    global_step += 1
                    token_processed += (physical_training_batch_size_per_worker * block_size)

                # Unscales the gradients of optimizer's assigned parameters in-place for clipping.
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped
                scaler.step(optimizer)
                
                # Updates the scale for next iteration.
                scaler.update()
                
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            assert epoch_batch_count > 0, "epoch_batch_count must be greater than 0"

            epoch_train_loss = epoch_train_loss / epoch_batch_count
            report_metrics["epoch_train_loss"] = epoch_train_loss
            report_metrics["epoch_batch_count"] = epoch_batch_count
            report_metrics["global_step"] = global_step

            t1 = time.time()
            dt = t1 - t0
            token_per_second = token_processed / dt            
            report_metrics["token_total"] = token_processed
            report_metrics["token_process_time_ms"] = dt * 1000
            report_metrics["token_per_second"] = token_per_second

            report_metrics["norm"] = norm.item()
            report_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]

            # Evaluate the model on the validation set only the current worker is the rank 0 worker
            if  ray.train.get_context().get_world_rank() == 0:
                validate_loss = 0
                validate_batch_count = 0
                model.eval()
                with torch.no_grad():
                    for batch in validate_data_shard.iter_torch_batches(batch_size=physical_validate_batch_size_per_worker,drop_last=False,):
                        validate_batch_count += 1
                        input_ids = batch["input_ids"]
                        target_ids = batch["target_ids"]

                        with torch.autocast(device_type=device.type,dtype=floating_point_precision,enabled=use_amp,):
                            logits = model(input_ids)
                            loss = loss_function(logits.flatten(0, 1), target_ids.flatten())

                        validate_loss += loss.item()  # only for reporting
                        metric.update(logits, target_ids)

                validate_loss = validate_loss / validate_batch_count
                perplexity = metric.compute().item()
                metric.reset()

                report_metrics["validate_loss"] = validate_loss
                report_metrics["validate_batch_count"] = validate_batch_count
                report_metrics["perplexity"] = perplexity


                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    best_epoch = epoch

                    report_metrics["best_epoch"] = best_epoch
                    report_metrics["best_perplexity"] = best_perplexity
                    report_metrics["validate_loss_at_best_perplexity"] = validate_loss

                    # In standard DDP training, where the model is the same across all ranks,
                    # so only the global rank 0 worker needs to save and report the checkpoint
                   
                    # create the best_checkpoint_dir if it does not exist
                    if not os.path.exists(best_checkpoint_dir):
                        os.makedirs(best_checkpoint_dir)

                    utility.save_checkpoint(
                        model,
                        optimizer,
                        scaler,
                        epoch,
                        perplexity,
                        best_checkpoint_dir,
                    )

                # save the latest checkpoint periodically
                if epoch % check_frequency == 0:
                    if not os.path.exists(latest_checkpoint_dir):
                        os.makedirs(latest_checkpoint_dir)

                    utility.save_checkpoint(
                        model,
                        optimizer,
                        scaler,
                        epoch,
                        perplexity,
                        latest_checkpoint_dir,
                    )

            ray.train.report(metrics=report_metrics)

    @staticmethod
    def _prepare_gradient_scaler(use_amp=True):
        return torch.amp.GradScaler(enabled=use_amp)

    @staticmethod
    def _resume_training(checkpoint_dir, model, optimizer,scaler, device):
        
        epoch, perplexity = utility.load_checkpoint(
            model, optimizer, scaler, checkpoint_dir,str(device)
        )
        
        return epoch,perplexity

    @staticmethod
    def _prepare_metric(device):
        metric = torchmetrics.text.Perplexity().to(device)
        return metric

    @staticmethod
    def _prepare_loss_function():
        loss_function = torch.nn.CrossEntropyLoss()
        return loss_function

    @staticmethod
    def _prepare_lr_scheduler(warmup_steps, max_steps, max_lr, min_lr,decay_lr,optimizer):
        scheduler = GPTLRScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            max_lr=max_lr,
            min_lr=min_lr,
            decay_lr=decay_lr,
        )

        return scheduler

    @staticmethod
    def _prepare_optimizer(weight_decay, max_lr,beta1,beta2, model, device):

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in str(device)
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=max_lr,
            betas=(beta1, beta2),
            eps=1e-8,
            **extra_args,
        )

        return optimizer

    @staticmethod
    def _prepare_data():
        train_data_shard = ray.train.get_dataset_shard("train")
        validate_data_shard = ray.train.get_dataset_shard("validate")
        return train_data_shard, validate_data_shard

    @staticmethod
    def _prepare_model(
        vocab_size,
        dimension_embedding,
        block_size,
        num_layers,
        num_headers,
        drop_rate,
        bias,
    ):
        model = GPT(
            vocab_size,
            dimension_embedding,
            block_size,
            num_layers,
            num_headers,
            drop_rate,
            bias,
        )
        model = torch.compile(model)
        model = ray.train.torch.prepare_model(model)

        return model
