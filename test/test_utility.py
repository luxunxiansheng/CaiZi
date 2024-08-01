import unittest

import torch
import ray

from config import gpt2_cfg as cfg 
from utility import save_checkpoint, resume_checkpoint

from model.GPT import GPT

class RayClassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start it once for the entire test suite/module
        ray.init(num_cpus=4, 
                num_gpus=0,)

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
        qkv_bias = False

        model = GPT(vocab_size, dimension_embedding, block_size,n_layers, num_header, drop_rate, qkv_bias)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
        

        epoch = 5
        temp_checkpoint_dir = cfg.output_dir
        checkpoint = save_checkpoint(model, optimizer, epoch, temp_checkpoint_dir)
        self.assertTrue(checkpoint)
        
    def test_resume_checkpoint(self):
        
        self.test_save_checkpoint()
        
        # Create the checkpoint, which is a reference to the directory.
        checkpoint = ray.train.Checkpoint.from_directory(cfg.output_dir)
        
        vocab_size = 50257
        dimension_embedding = 768
        block_size = 1024
        num_header = 12
        n_layers = 12
        drop_rate = 0.1
        qkv_bias = False
        
        model = GPT(vocab_size, dimension_embedding, block_size, n_layers, num_header, drop_rate, qkv_bias)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    
        epoch_start = resume_checkpoint(model, optimizer, checkpoint)
    
        self.assertEqual(epoch_start, 6)
        
        
        

if __name__ == "__main__":
    unittest.main()
