from typing import  Dict


class DatasourceProcessor():
    def __call__(self, path: Dict[str,str]) -> Dict[str, str]:
        with open(path['item'], "r", encoding="utf-8") as f:
            raw_text = f.read()   
        return {"text": raw_text}

