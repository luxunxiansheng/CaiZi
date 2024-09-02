"""
Author: Bin.Li
Email:  ornot2008@yahoo.com
MIT License
Copyright (c) 2025 debutpark.com
"""


import unittest

from preprocessor.chunk_processor import ChunkProcessor


class TestChunkProcessor(unittest.TestCase):
    def test_chunk_processor(self):
        
        chunk_processor = ChunkProcessor()
        data = {"token": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
              

        data_items = chunk_processor(data)
        
        print(data_items[0]["input_ids"])
        print(data_items[0]["target_ids"])
    
        print("*********************************")
   
        print(data_items[1]["input_ids"])
        print(data_items[1]["target_ids"])
        

if __name__ == "__main__":
    unittest.main()
        