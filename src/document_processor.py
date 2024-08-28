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


from abc import ABC, abstractmethod
from pathlib import Path
from typing import  Dict

from datasets import load_from_disk

class DocumentProcessor(ABC):
    @abstractmethod
    def __call__(self, doc_path: Dict[str,str]) -> Dict[str, str]:
        pass

class TextDocumentProcessor(DocumentProcessor):
    def __init__(self,train_ratio:float=0.9) -> None:
        self.train_ratio = train_ratio
       
            
    def __call__(self, doc_path: Dict[str,str]) -> Dict[str, str]:
        with open(doc_path['item'], "r", encoding="utf-8") as f:
            raw_text = f.read()        

        raw_length = len(raw_text) 
        train_text = raw_text[:int(raw_length * self.train_ratio)]
        validate_text = raw_text[int(raw_length * self.train_ratio):]
        return {"train": train_text, "validate": validate_text}

# class HuggingFaceDocumentProcessor(DocumentProcessor):
#     def __init__(self,train_ratio:float=0.99995) -> None:
#         self.train_ratio = train_ratio


#     def __call__(self, doc_path: Dict[str,str]) -> Dict[str, str]:
#         path = Path(doc_path['item'])
#         dataset = load_from_disk(path)
#         # owt by default only contains the 'train' split, so create a test split
#         split_dataset = dataset["train"].train_test_split(test_size=1-self.train_ratio, seed=2357, shuffle=True)
#         split_dataset['val'] = split_dataset.pop('test') # rename the test split to val




        
    