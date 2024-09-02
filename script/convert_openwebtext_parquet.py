
import os

from fastapi import concurrency
project_root = os.path.dirname(os.getcwd())
import sys
# Add the directory to the Python path
sys.path.append(f"{project_root}/src")


from pathlib import Path

import ray
from datasets import load_dataset,load_from_disk

from config import gpt2_cfg as cfg 
from datasource_processor import DatasourceProcessor
from text_split_processor import TextSplitProcessor
from chunk_processor import ChunkProcessor
from token_processor import TokenProcessor

ray.data.DataContext.get_current().execution_options.verbose_progress = True
ray.data.DataContext.get_current().DEFAULT_ENABLE_PROGRESS_BAR_NAME_TRUNCATION = False



hf_dataset = load_from_disk(cfg["dataset"]["source"][0]["path"])
ray_ds = ray.data.from_huggingface(hf_dataset)

train_ratio = cfg["ray_data"]["train_ratio"]
text_split_processor = TextSplitProcessor(train_ratio=train_ratio)
texts = ray_ds.map(text_split_processor,
                          num_cpus=2,
                        concurrency= 2)

tokenizer_class = TokenProcessor.create(cfg['ray_data']['tokenizer_class']['name'])
tokenizer_args =  cfg['ray_data']['tokenizer_class']['args']
tokenizer= tokenizer_class(**tokenizer_args)
tokens = texts.map(tokenizer,
                num_cpus=4,
                concurrency= 4)

block_size = cfg["model"]["block_size"]
stride = cfg["model"]["stride"]
chunk_processor = ChunkProcessor(block_size=block_size, stride=stride)
chunked_tokens = tokens.map(chunk_processor,
                            num_cpus=4,
                            concurrency= 4)


chunked_tokens.write_parquet(cfg["dataset"]["chunked_tokens"],concurrency=10)

