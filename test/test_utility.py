from ast import mod
from json import load

import unittest

import tempfile

import torch
import ray

from config import gpt2_cfg as cfg
from token_processor import TikTokenizer
from  text_generator import TextGenerator
from utility import (
    load_hf_weights_into_gpt,
    save_checkpoint,
    resume_checkpoint,
)

from model.GPT import GPT
import unittest



#@unittest.skip("skip")
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

        epoch = 5
        perplexity = 123.4

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            save_checkpoint(model, optimizer, epoch, perplexity, temp_checkpoint_dir)

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

        epoch = 5
        perplexity = 123.4
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            save_checkpoint(model, optimizer, epoch, perplexity, temp_checkpoint_dir)

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

            epoch_start, perplexity = resume_checkpoint(model, optimizer, checkpoint)

            self.assertEqual(epoch_start, 5)
            self.assertEqual(perplexity, 123.4)


class UtilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model_size = "124M"
        self.model_dir =  cfg["124M"]["openai_gpt_dir"] +"/"+self.model_size


    #@unittest.skip("skip")
    def test_load_weights_to_gpt(self):
        tokenizer = TikTokenizer()
        start_context = "Every effort moves you"
        encoded_tensor = tokenizer.encode(start_context)

        vocab_size = 50257
        dimension_embedding = 768
        block_size = 1024
        num_header = 12
        n_layers = 12
        drop_rate = 0.1
        bias = True

        model = GPT(
            vocab_size,
            dimension_embedding,
            block_size,
            n_layers,
            num_header,
            drop_rate,
            bias,
        )
        
        print("model:", model)

        load_hf_weights_into_gpt(model, "gpt2")
        

        torch.manual_seed(123)
        
        # Create a TextGenerator instance
        text_generator = TextGenerator(model, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # Generate new text
        decoded = text_generator(encoded_tensor, 
                                 max_new_tokens=25, 
                                 context_size=1024,
                                 temperature=1.5,
                                 top_k=50,)
        print("decoded:", decoded)
     

       
        
        

    
        
        
        


        
        
        
        
        


       


if __name__ == "__main__":
    unittest.main()
