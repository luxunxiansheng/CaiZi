import unittest
import torch
from model.multihead_attention import MultiHeadAttention

class TestMultiHeadAttention(unittest.TestCase):
    def test_forward(self):
        
        torch.manual_seed(123)

        inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
        )

        batch = torch.stack((inputs, inputs), dim=0)
        block_size = batch.shape[1]
        dimension_input = inputs.shape[1]
        dimension_embedding = 2
        
        attention = MultiHeadAttention(dimension_input, dimension_embedding, block_size, 0.0, 2)



        with torch.no_grad():
            context_vecs = attention(batch)

        print(context_vecs)
 

        

if __name__ == "__main__":
    unittest.main()