
from abc import ABC, abstractmethod
from pathlib import Path


class DocumentProcessorBase(ABC):
    @abstractmethod
    def __call__(self, doc_path: Path) -> str:
        pass

class TextDocumentProcessor(DocumentProcessorBase):
    def __call__(self, doc_path: Path) -> str:
        with open(doc_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        return raw_text