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
    def __call__(self, doc_path: Dict[str,str]) -> Dict[str, str]:
        with open(doc_path['item'], "r", encoding="utf-8") as f:
            raw_text = f.read()
        return {"text": raw_text}