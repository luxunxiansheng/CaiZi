import unittest

import torch

from model.transformer_block import TransformerBlock

from config import gpt2_cfg
class TestTransformerBlock(unittest.TestCase):
    def test_forward(self):
        torch.manual_seed(123)

        x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
        block = TransformerBlock(768, 1024,12,0.1,False)
        output = block(x)

        print("Input shape:", x.shape)
        print("Output shape:", output.shape)

if __name__ == "__main__":
    unittest.main()