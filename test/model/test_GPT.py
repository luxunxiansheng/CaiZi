import unittest
import torch

import tiktoken
from model.GPT import GPT


class TestGPT(unittest.TestCase):
    def test_forward(self):
        

        tokenizer = tiktoken.get_encoding("gpt2")

        batch = []

        txt1 = "Every effort moves you"
        txt2 = "Every day holds a"

        batch.append(torch.tensor(tokenizer.encode(txt1)))
        batch.append(torch.tensor(tokenizer.encode(txt2)))
        batch = torch.stack(batch, dim=0)
        print(batch.shape)
        
        vocab_size = 50257
        dimension_embedding = 768
        block_size = 1024
        num_header = 12
        n_layers = 12
        drop_rate = 0.1
        qkv_bias = False

        model = GPT(vocab_size, dimension_embedding, block_size,n_layers, num_header, drop_rate, qkv_bias)
        
        out = model(batch)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params:,}")
        
        # parameters = model.state_dict()
        # for k,v in parameters.items():
        #     print(k,v.shape)
        

        # in_idx = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        # logits = model.forward(in_idx)

        # self.assertEqual(logits.shape, (2, 5, vocab_size))

if __name__ == "__main__":
    unittest.main()