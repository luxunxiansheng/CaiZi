
from abc import ABC
from abc import abstractmethod
from typing import Dict, List

import tiktoken
    
class TokenProcessor(ABC):
    @abstractmethod
    def __call__(self, text:Dict[str,str]) -> Dict[str, List[int]]:
        raise(NotImplementedError)
    
    @abstractmethod
    def decode(self, tokens:Dict[str, int]) -> Dict[str, str]:
        raise(NotImplementedError)
    

class TikTokenizer(TokenProcessor):
    def __init__(self,name:str="gpt2"):
        self.tokenizer = tiktoken.get_encoding(name)
    
    def __call__(self, input_text:Dict[str,str]) -> Dict[str, List[int]]:        
        text = input_text['text']        
        integers = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        return {"ids": integers}
        
    def decode(self, input_ids:Dict[str, List[int]]) -> Dict[str, str]:
        ids = input_ids['ids']
        text = self.tokenizer.decode(ids)
        return {"text": text}
        
