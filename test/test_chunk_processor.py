"""
Author: Bin.Li
Email:  ornot2008@yahoo.com
MIT License
Copyright (c) 2025 debutpark.com
"""


import unittest

from chunk_processor import ChunkProcessor


class TestChunkProcessor(unittest.TestCase):
    def test_chunk_processor(self):
        
        chunk_processor = ChunkProcessor()
        
        data = {"ids": [1,2,3,4,5,6,7,8,9,10]}
        
        data_items = chunk_processor(data)
        
        print(data_items)
        

if __name__ == "__main__":
    unittest.main()
        