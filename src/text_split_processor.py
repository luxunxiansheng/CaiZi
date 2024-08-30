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


class TextSplitProcessor():
    def __init__(self,train_ratio:float=0.9) -> None:
        self.train_ratio = train_ratio
       
            
    def __call__(self, text: Dict[str,str]) -> Dict[str, str]:
    
        
        raw_text = text["text"]

        raw_length = len(raw_text) 
        train_text = raw_text[:int(raw_length * self.train_ratio)]
        validate_text = raw_text[int(raw_length * self.train_ratio):]
        return {"train": train_text, "validate": validate_text} 









        
    