import unittest


import torch


from model.GPT import GPT
from preprocessor.token_processor import CharTokenizer
from text_generator import TextGenerator


from config import gpt2_nano_cfg
from utility import load_model_from_checkpoint

class TestTextGenerator(unittest.TestCase):
    def test_text_generator(self):
        
        token_processor_args = gpt2_nano_cfg["ray_data"]['tokenizer_class']['args']

        token_processor = CharTokenizer(**token_processor_args)
        
        start_context = "hello world"

        encoded = token_processor({"train": start_context, "validate": start_context})
        encoded_tensor = torch.tensor(encoded["train"]).unsqueeze(0)
        
        print("encoded:", encoded_tensor)

        
        vocab_size = 65
        dimension_embedding = 384
        block_size = 256
        num_header = 6
        n_layers = 6
        drop_rate = 0.2
        bias = False

        model = GPT(vocab_size, dimension_embedding, block_size,n_layers, num_header, drop_rate, bias)
        
        load_model_from_checkpoint(model, "/workspaces/CaiZi/model_weights/best_checkpoint", device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        
        model.eval()
        
        # Create a TextGenerator instance
        text_generator = TextGenerator(model,token_processor, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Generate new text
        decoded = text_generator(encoded_tensor, max_new_tokens=2000, block_size=256)

        print(f"\n\n --------------decoded------------:\n\n{decoded}")
        
        
if __name__ == "__main__":
    unittest.main()