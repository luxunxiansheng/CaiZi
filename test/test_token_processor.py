from math import e
import unittest

from token_processor import CharTokenizer, TikTokenizer


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
        
        raw_text= {"text": "Hello, do you like tea?]"}
        encoded_text = self.token_processor(raw_text)
        print(encoded_text)
        
        print(self.token_processor.decode(encoded_text["ids"]))


if __name__ == "__main__":
    unittest.main()