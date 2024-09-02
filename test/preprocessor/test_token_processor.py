import unittest

from preprocessor.text_split_processor import TextSplitProcessor
from preprocessor.token_processor import CharTokenizer, TokenProcessor
from config import gpt2_cfg, gpt2_nano_cfg

#@unittest.skip("Skip this test")
class TestTokenProcessor(unittest.TestCase):


    def test_create(self):
        
        token_processor_class_name = gpt2_cfg["ray_data"]['tokenizer_class']['name']
        token_processor_args = gpt2_cfg["ray_data"]['tokenizer_class']['args']
        token_processor_class= TokenProcessor.create(token_processor_class_name)
        token_processor= token_processor_class(**token_processor_args)
        
        raw_text= {"text": "Hello,In the sunlit terraces of someunknownPlace."}
        
        encoded_text = token_processor(raw_text)
        print((encoded_text["token"]))
        
 

#@unittest.skip("Skip this test")
class TestCharTokenizer(unittest.TestCase):
    def test_encode_and_decoder(self):
        
        doc_processor = TextSplitProcessor()
        raw_text = "hello wold"
        
        token_processor_args = gpt2_nano_cfg["ray_data"]['tokenizer_class']['args']

        token_processor = CharTokenizer(**token_processor_args)
        encoded_text = token_processor({"text":raw_text})
        print(len(encoded_text["token"]))
        
        print(token_processor.decode(encoded_text["token"]))



if __name__ == "__main__":
    unittest.main()