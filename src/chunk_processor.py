""" 
###########################################
#
# Author: Bin.Li                          
# Email:  ornot2008@yahoo.com
# MIT License
# Copyright (c) 2025 debutpark.com 
#
###########################################
"""

from typing import Dict, List
class ChunkProcessor:
    def __init__(self, max_length:int=4, stride:int=1):
        self.max_length = max_length
        self.stride = stride


    def __call__(self, data: Dict[str,List[int]])->List[Dict[str, List[int]]]:
        
        token_ids = data['ids']
        
        data_items = []
        
        for i in range(0, len(token_ids) - self.max_length, self.stride):
            input_chunk = token_ids[i:i + self.max_length]
            target_chunk = token_ids[i + 1: i + self.max_length + 1]
            data_items.append({"input_ids": input_chunk, "target_ids": target_chunk})
        
        return data_items
            

        
        
        
      


