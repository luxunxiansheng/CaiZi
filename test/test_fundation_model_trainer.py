import unittest

from config import gpt2_cfg as cfg 
from fundation_model_trainer import RayGPT2FundationModelTrainer

class TestGPT2FundationModelTrainer(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = RayGPT2FundationModelTrainer(cfg)
    
    
    #@unittest.skip("skip training test")
    def test_data_process(self):
        self.trainer.start_ray()
        self.trainer.data_preprocess()
       
        validate_dataset = self.trainer.validate_chunked_tokens
        
        validate_token_count = 0
        for row in validate_dataset.iter_rows():
            validate_token_count = validate_token_count + len(row["input_ids"])
            
        
            
        train_dataset = self.trainer.train_chunked_tokens
        train_token_count = 0
        for row in train_dataset.iter_rows():
            train_token_count = train_token_count + len(row["input_ids"])
        
        print(f"tokens: {train_token_count}")
        print(f"tokens: {validate_token_count}")
        
        print(f"total tokens: {train_token_count + validate_token_count}")
        
        
        self.trainer.stop_ray()

    @unittest.skip("skip training test")
    def test_train(self):
        self.trainer.start_ray()
        self.trainer.data_preprocess()
        
        self.trainer.self_supervised_train()
        
        self.trainer.stop_ray()
        
        

    


if __name__ == "__main__":
    unittest.main()