import unittest

from token_processor import TikTokenizer

class TestTokenProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.token_processor = TikTokenizer()
        
    def test_encode_and_decoder(self):
        
        raw_text= {"text": "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."}
        
        encoded_text = self.token_processor(raw_text)
        print(encoded_text["ids"])
        
        
        strings = self.token_processor.decode(encoded_text)

        print(strings)
        
    def test_decode(self):
        pass
        


if __name__ == "__main__":
    unittest.main()