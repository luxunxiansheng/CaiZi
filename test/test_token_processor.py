from math import e
import unittest

from document_processor import TextDocumentProcessor
from token_processor import CharTokenizer, TikTokenizer
from config import gpt2_cfg

@unittest.skip("Skip this test")
class TestTokenProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.token_processor = TikTokenizer()
        
    def test_encode_and_decoder(self):
        
        raw_text= {"text": "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."}
        
        encoded_text = self.token_processor(raw_text)
        print(len(encoded_text["ids"]))
        
        
        strings = self.token_processor.decode(encoded_text["ids"])

        print(strings)
        
    def test_decode(self):
        pass

class TestCharTokenizer(unittest.TestCase):
    def setUp(self) -> None:
        

        self.token_processor = CharTokenizer()
        
    def test_encode_and_decoder(self):
        
        doc_processor = TextDocumentProcessor(section="train")

        train_raw_text = doc_processor({"item":gpt2_cfg.dataset[0]["path"]})
        print(f"train_raw_text length: {len(train_raw_text['text'])}")
        
        
        encoded_text = self.token_processor(train_raw_text)
        print(len(encoded_text["ids"]))
        
        # print(self.token_processor.decode(encoded_text["ids"]))

class TestTokenProcessor(unittest.TestCase):
    def setUp(self) -> None:
        pass
    
    def test_create(self):
        token_processor_class= CharTokenizer.create("CharTokenizer")
        char_token_processor = token_processor_class()
        
        
   
        
    def test_call(self):
        pass
    
    def test_decode(self):
        pass

if __name__ == "__main__":
    unittest.main()