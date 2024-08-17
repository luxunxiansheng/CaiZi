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
    

class CharTokenizer(TokenProcessor):
    def __call__(self, input_text:Dict[str,str]) -> Dict[str, List[int]]:
        text = input_text['text']
        
        unique_chars = self.detect_unique_chars(text)
        self.char2idx = {char:idx for idx,char in enumerate(unique_chars)}
        self.idx2char = {idx:char for idx,char in enumerate(unique_chars)}
        
        integers = [self.char2idx[char] for char in text]
        return {"ids": integers}
        
    def decode(self, input_ids:List[int]) -> str:
        return "".join([self.idx2char[idx] for idx in input_ids])
        
    def encode(self, text:str) -> torch.Tensor:
        encoded = [self.char2idx[char] for char in text]
        return  torch.tensor(encoded).unsqueeze(0)
    
    def detect_unique_chars(self,text)->List[str]:
        unique_chars = set(text)
        return list(unique_chars)

class TikTokenizer(TokenProcessor):
    def __init__(self,name:str="gpt2"):
        self.tokenizer = tiktoken.get_encoding(name)
  
    def __call__(self, input_text:Dict[str,str]) -> Dict[str, List[int]]:        
        text = input_text['text']        
        integers = self.tokenizer.encode(text,allowed_special={"<|endoftext|>"})
        integers.append(self.tokenizer.eot_token)
        return {"ids": integers}
        
    def decode(self, input_ids:List[int]) -> str:       
        return self.tokenizer.decode(input_ids)
        
    def encode(self, text:str) -> torch.Tensor:
        encoded = self.tokenizer.encode(text,allowed_special={"<|endoftext|>"})
        return  torch.tensor(encoded).unsqueeze(0)
