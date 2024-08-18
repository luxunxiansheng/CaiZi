import unittest
from uu import decode
import torch


from model.GPT import GPT
from text_generator import TextGenerator
from token_processor import CharTokenizer, TikTokenizer
from collections import OrderedDict
from config import gpt2_cfg

class TestTextGenerator(unittest.TestCase):
    def test_text_generator(self):
        
        token_processor_args = gpt2_cfg["ray_data"]['tokenizer_class']['args']

        token_processor = CharTokenizer(**token_processor_args)
        
        start_context = "\n"

        encoded = token_processor({"text": start_context})
        encoded_tensor = torch.tensor(encoded["ids"]).unsqueeze(0)
        
        print("encoded:", encoded_tensor)

        
        
        
        vocab_size = 65
        dimension_embedding = 384
        block_size = 256
        num_header = 6
        n_layers = 6
        drop_rate = 0.2
        bias = False

        model = GPT(vocab_size, dimension_embedding, block_size,n_layers, num_header, drop_rate, bias)
        
        checkpoint = torch.load("/workspaces/CaiZi/model_weights/best_checkpoint/checkpoint.pt")["model"]
        
        

        # Create a new state_dict without 'module.' prefix
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace('_orig_mod.', '')  # remove 'module.' prefix
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        
        
        
        model.eval()
        
        # Create a TextGenerator instance
        text_generator = TextGenerator(model,token_processor, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Generate new text
        decoded = text_generator(encoded_tensor, max_new_tokens=2000, block_size=256)

        print("decoded:", decoded)
        
        
if __name__ == "__main__":
    unittest.main()