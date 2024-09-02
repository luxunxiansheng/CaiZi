import unittest

from datasource_processor import DatasourceProcessor

class TestDatasourceProcessor(unittest.TestCase):
    def test_parquet_format(self):
        processor = DatasourceProcessor(source_format="parquet")
        path = {"item": "/workspaces/CaiZi/dataset/openwebtext/shards/shard_0.parquet"}
        result = processor(path)
        print(result[0]["text"])
      

if __name__ == "__main__":
    unittest.main()