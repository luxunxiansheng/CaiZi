import unittest


from config import gpt2_cfg as cfg

from preprocessor.datasource_processor import DatasourceProcessor

class TestDatasourceProcessor(unittest.TestCase):
    def test_parquet_format(self):
        processor = DatasourceProcessor(source_format="parquet")
        path = {"item": cfg.project_root+"/dataset/openwebtext/shards/shard_0.parquet"}
        result = processor(path)
        print(len(result["text"]))
      

if __name__ == "__main__":
    unittest.main()