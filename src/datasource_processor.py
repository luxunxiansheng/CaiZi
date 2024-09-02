from typing import  Dict, List

import pyarrow.parquet as pq

class DatasourceProcessor():
    def __init__(self, source_format: str = "text"):
        self.source_format = source_format

    def __call__(self, path: Dict[str,str]) -> List[Dict[str, str]]:
        if self.source_format == "text":
            with open(path['item'], "r", encoding="utf-8") as f:
                raw_text = f.read()   
            return [{"text": raw_text}]
        elif self.source_format == "parquet":
            table = pq.read_table(path['item'])
            texts = table['text']
            return [{"text": text.as_py()} for text in texts]
    


            

