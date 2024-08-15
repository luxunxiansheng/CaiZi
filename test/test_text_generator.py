import unittest
from uu import decode
import torch


from model.GPT import GPT
from text_generator import TextGenerator
from token_processor import TikTokenizer

class TestTextGenerator(unittest.TestCase):
    def test_text_generator(self):
        
        tokenizer = TikTokenizer()
        
        start_context = "Hello, I am"

        encoded_tensor = tokenizer.encode(start_context)
        print("encoded:", encoded_tensor)

        
        print("encoded_tensor.shape:", encoded_tensor.shape)
        
        vocab_size = 50257
        dimension_embedding = 768
        block_size = 1024
        num_header = 12
        n_layers = 12
        drop_rate = 0.1
        bias = True

        model = GPT(vocab_size, dimension_embedding, block_size,n_layers, num_header, drop_rate, bias)
        
        model.eval()
        
        # Create a TextGenerator instance
        text_generator = TextGenerator(model, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Generate new text
        decoded = text_generator(encoded_tensor, max_new_tokens=6, block_size=1024)

        print("decoded:", decoded)
        
        
if __name__ == "__main__":
    unittest.main()