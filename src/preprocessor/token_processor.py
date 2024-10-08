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


from abc import ABC
from abc import abstractmethod

from typing import Dict, List

import torch

import tiktoken
    
class TokenProcessor(ABC):
    @abstractmethod
    def __call__(self, text:Dict[str,str]) -> Dict[str, List[int]]:
        raise(NotImplementedError)
    
    @abstractmethod
    def decode(self, tokens:List[int]) -> str:
        raise(NotImplementedError)
    
    @staticmethod
    def create(name: str) -> "TokenProcessor":
        if name == "CharTokenizer":
            return CharTokenizer
        elif name == "TikTokenizer":
            return TikTokenizer
        else:
            raise ValueError("Unknown TokenProcessor")
    

class TikTokenizer(TokenProcessor):
    def __init__(self,name:str="gpt2"):
        self.tokenizer = tiktoken.get_encoding(name)
  
    def __call__(self, input_text:Dict[str,str]) -> Dict[str, List[int]]:        
        text = input_text['text']        
        integers = self.tokenizer.encode(text,allowed_special={"<|endoftext|>"})
        return {"token": integers}
        
    def decode(self, input_ids:List[int]) -> str:       
        return self.tokenizer.decode(input_ids)
        
    def encode(self, text:str) -> torch.Tensor:
        encoded = self.tokenizer.encode(text,allowed_special={"<|endoftext|>"})
        return  torch.tensor(encoded).unsqueeze(0)


class CharTokenizer(TokenProcessor):
    def __init__(self,unique_chars:List[str]):
        assert len(unique_chars) > 0, "Ensure you have at least one unique character"
        self.char2idx = {char:idx for idx,char in enumerate(unique_chars)}
        self.idx2char = {idx:char for idx,char in enumerate(unique_chars)}
    
    def __call__(self, input_text:Dict[str,str]) -> Dict[str, List[int]]:
        text = input_text['text']     
        assert self.char2idx is not None, "Ensure you have at least one unique character"   
        integers = [self.char2idx[char] for char in text]

        return {"token": integers}
        
    def decode(self, input_ids:List[int]) -> str:
        assert self.idx2char is not None, "Ensure you have at least one unique character"
        return "".join([self.idx2char[idx] for idx in input_ids])
    
    def encode(self, text:str) -> torch.Tensor:
        encoded = [self.char2idx[char] for char in text]
        return  torch.tensor(encoded).unsqueeze(0)
   
    @staticmethod
    def detect_unique_chars(text)->List[str]:
        unique_chars = set(text)
        return sorted(list(unique_chars))

