import unittest

from config import gpt2_cfg
from document_processor import TextDocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    def test_document_processor(self):
        doc_processor = TextDocumentProcessor()

        raw_text = doc_processor(gpt2_cfg.dataset["xjtu_1"]['path'])
        print(raw_text[:100])
   

    

if __name__ == "__main__":
    unittest.main()