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

from typing import  Dict
import pyarrow.parquet as pq

class DatasourceProcessor():
    def __init__(self, source_format: str = "text"):
        self.source_format = source_format

    def __call__(self, path: Dict[str,str]) -> Dict[str, str]:
        if self.source_format == "text":
            with open(path['item'], "r", encoding="utf-8") as f:
                raw_text = f.read()   
            return {"text": raw_text}
        elif self.source_format == "parquet":
            table = pq.read_table(path['item'])
            texts = table['text']
            
            raw_text = ""
            for text in texts:
                raw_text += text.as_py() + "<|endoftext|>"            
            return {"text": raw_text}
        else:
            return {"text": ""}
  
            
            
            

            



            

