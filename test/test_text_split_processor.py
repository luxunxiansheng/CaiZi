"""
Author: Bin.Li
Email:  ornot2008@yahoo.com
MIT License
Copyright (c) 2025 debutpark.com
"""

import unittest

from config import gpt2_cfg
from text_split_processor import  TextSplitProcessor

class TestTextSplitProcessor(unittest.TestCase):
    @unittest.skip("skip test_text_document_processor")
    def test_text_split_processor(self):
        doc_processor = TextSplitProcessor()

        with open(gpt2_cfg.text_dataset[0]["path"], "r", encoding="utf-8") as f:
             raw_text = f.read()   

        train_raw_text = doc_processor({"text":raw_text})
        print(f"train_raw_text length: {len(train_raw_text['train'])}")
   
    

    

if __name__ == "__main__":
    unittest.main()