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


class DocumentProcessor(ABC):
    @abstractmethod
    def __call__(self, doc_path: Path) -> Dict[str, str]:
        pass

class TextDocumentProcessor(DocumentProcessor):
    def __init__(self,section:str="train",train_ratio:float=0.9) -> None:
        self.train_ratio = train_ratio
        self.section = section
        
    
    def __call__(self, doc_path: Dict[str,str]) -> Dict[str, str]:
        with open(doc_path['item'], "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        if self.section == "train":
            text = raw_text[:int(len(raw_text) * self.train_ratio)]
        elif self.section == "validate":
            text = raw_text[int(len(raw_text) * self.train_ratio):]
        else:
            raise ValueError("section must be either 'train' or 'validate'")
        
        
        return {"text": text}