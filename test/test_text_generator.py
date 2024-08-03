import unittest
from uu import decode
import torch
import tiktoken

from model.GPT import GPT
from text_generator import TextGenerator

class TestTextGenerator(unittest.TestCase):
    def test_text_generator(self):
        
        tokenizer = tiktoken.get_encoding("gpt2")
        
        start_context = "Hello, I am"

        encoded = tokenizer.encode(start_context)
        print("encoded:", encoded)

        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        print("encoded_tensor.shape:", encoded_tensor.shape)
        
        vocab_size = 50257
        dimension_embedding = 768
        block_size = 1024
        num_header = 12
        n_layers = 12
        drop_rate = 0.1
        qkv_bias = False

        model = GPT(vocab_size, dimension_embedding, block_size,n_layers, num_header, drop_rate, qkv_bias)
        model.eval()
        
        # Create a TextGenerator instance
        text_generator = TextGenerator(model)
        
        # Generate new text
        decoded = text_generator(encoded_tensor, max_new_tokens=6, context_size=1024)

        print("decoded:", decoded)
        
        
if __name__ == "__main__":
    unittest.main()