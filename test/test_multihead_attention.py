import unittest
import torch
from multihead_attention import MultiHeadAttention

class TestMultiHeadAttention(unittest.TestCase):
    def test_forward(self):
        dimension_input = 64
        dimension_embedding = 128
        block_size = 10
        dropout = 0.1
        num_heads = 4

        attention = MultiHeadAttention(dimension_input, dimension_embedding, block_size, dropout, num_heads)

        x = torch.randn(2, 10, dimension_input)  # Random input tensor

        output = attention(x)

        self.assertEqual(output.shape, (2, 10, dimension_embedding))  # Check output shape


if __name__ == "__main__":
    unittest.main()