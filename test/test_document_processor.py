"""
Author: Bin.Li
Email:  ornot2008@yahoo.com
MIT License
Copyright (c) 2025 debutpark.com
"""

import unittest

from config import gpt2_cfg
from document_processor import TextDocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    def test_document_processor(self):
        doc_processor = TextDocumentProcessor(section="validate")

        raw_text = doc_processor({"item":gpt2_cfg.dataset[0]["path"]})
        print(len(raw_text))
   

    

if __name__ == "__main__":
    unittest.main()