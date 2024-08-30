from enum import unique
from math import e
import unittest

from text_split_processor import TextSplitProcessor
from token_processor import CharTokenizer, TikTokenizer, TokenProcessor
from config import gpt2_cfg

#@unittest.skip("Skip this test")
class TestTokenProcessor(unittest.TestCase):


    def test_create(self):
        
        token_processor_class_name = gpt2_cfg["ray_data"]['tokenizer_class']['name']
        token_processor_args = gpt2_cfg["ray_data"]['tokenizer_class']['args']
        token_processor_class= TokenProcessor.create(token_processor_class_name)
        token_processor= token_processor_class(**token_processor_args)
        
        raw_text= {"train": "Hello,In the sunlit terraces of someunknownPlace.",
                   "validate": "Hell,world."}
        
        encoded_text = token_processor(raw_text)
        print((encoded_text["train"]))
        
 

#@unittest.skip("Skip this test")
class TestCharTokenizer(unittest.TestCase):
    def test_encode_and_decoder(self):
        
        doc_processor = TextSplitProcessor()
        raw_text = doc_processor({"item":gpt2_cfg.text_dataset[0]["path"]})
        
        token_processor_args = gpt2_cfg["ray_data"]['tokenizer_class']['args']

        token_processor = CharTokenizer(**token_processor_args)
        encoded_text = token_processor(raw_text)
        print(len(encoded_text["train"]))
        
        print(token_processor.decode(encoded_text["train"]))



if __name__ == "__main__":
    unittest.main()