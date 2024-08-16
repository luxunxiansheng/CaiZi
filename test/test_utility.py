
import unittest

import tempfile
from contextlib import nullcontext

import torch
import ray

from config import gpt2_cfg as cfg
from token_processor import TikTokenizer
from text_generator import TextGenerator
from utility import (
    load_hf_weights_into_gpt,
    save_checkpoint,
    resume_checkpoint,
)

from model.GPT import GPT
import unittest


@unittest.skip("skip")
class RayClassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start it once for the entire test suite/module
        ray.init(
            num_cpus=4,
            num_gpus=0,
        )

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_save_checkpoint(self):
        vocab_size = 50257
        dimension_embedding = 768
        block_size = 1024
        num_header = 12
        n_layers = 12
        drop_rate = 0.1
        bias = False

        model = GPT(
            vocab_size,
            dimension_embedding,
            block_size,
            n_layers,
            num_header,
            drop_rate,
            bias,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
        scaler = torch.amp.GradScaler()

        epoch = 5
        perplexity = 123.4

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            save_checkpoint(model, optimizer, scaler,epoch, perplexity, temp_checkpoint_dir)

            checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)

            self.assertTrue(checkpoint)

    def test_resume_checkpoint(self):

        vocab_size = 50257
        dimension_embedding = 768
        block_size = 1024
        num_header = 12
        n_layers = 12
        drop_rate = 0.1
        bias = False
        device = 'cpu'
        torch.set_default_device(device)

        model = GPT(
            vocab_size,
            dimension_embedding,
            block_size,
            n_layers,
            num_header,
            drop_rate,
            bias,
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
        scaler = torch.amp.GradScaler()

        epoch = 5
        perplexity = 123.4
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            save_checkpoint(model, optimizer, scaler, epoch, perplexity, temp_checkpoint_dir)

            # Create the checkpoint, which is a reference to the directory.
            checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)

            vocab_size = 50257
            dimension_embedding = 768
            block_size = 1024
            num_header = 12
            n_layers = 12
            drop_rate = 0.1
            bias = False

            model = GPT(
                vocab_size,
                dimension_embedding,
                block_size,
                n_layers,
                num_header,
                drop_rate,
                bias,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.0004, weight_decay=0.1
            )
            scaler = torch.amp.GradScaler()

            epoch_start, perplexity = resume_checkpoint(model, optimizer,scaler, checkpoint,device)

            self.assertEqual(epoch_start, 5)
            self.assertEqual(perplexity, 123.4)


class UtilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model_size = "124M"
        self.model_dir = cfg["124M"]["openai_gpt_dir"] + "/" + self.model_size

    #@unittest.skip("skip")
    def test_load_weights_to_gpt(self):
        seed = 1337
        torch.manual_seed(seed)
        torch.set_float32_matmul_precision('high')
  
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'

        device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        
        
        tokenizer = TikTokenizer()
        start_context = "\n"
        encoded_tensor = tokenizer.encode(start_context)
        
        model_type = "gpt2"
        
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

        model = GPT(
            config["vocab_size"],
            config["n_embd"],
            config["block_size"],
            config["n_layer"],
            config["n_head"],
            config["dropout"],
            config["bias"],
    )
   
        load_hf_weights_into_gpt(model, "gpt2")
        
       
        # Create a TextGenerator instance
        text_generator = TextGenerator(model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        with torch.no_grad():
            with ctx:
                for k in range(10):        
                    # Generate new text
                    decoded = text_generator(
                        encoded_tensor,
                        max_new_tokens=500,
                        block_size=1024,
                        temperature=0.8,
                        top_k=200
                    )
                    print(decoded)


if __name__ == "__main__":
    unittest.main()
