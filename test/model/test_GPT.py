import unittest
import torch

from model.GPT import GPT


class TestGPT(unittest.TestCase):
    def test_forward(self):
        vocab_size = 50257
        dimension_embedding = 768
        block_size = 1024
        num_header = 12
        n_layers = 12
        drop_rate = 0.1
        qkv_bias = False

        model = GPT(vocab_size, dimension_embedding, block_size,n_layers, num_header, drop_rate, qkv_bias)
        
        parameters = model.state_dict()
        for k,v in parameters.items():
            print(k,v.shape)
        

        in_idx = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        logits = model.forward(in_idx)

        self.assertEqual(logits.shape, (2, 5, vocab_size))

if __name__ == "__main__":
    unittest.main()