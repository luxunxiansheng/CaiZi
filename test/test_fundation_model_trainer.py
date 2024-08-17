import unittest

from config import gpt2_cfg as cfg 
from fundation_model_trainer import RayGPT2FundationModelTrainer

class TestGPT2FundationModelTrainer(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = RayGPT2FundationModelTrainer(cfg)
    
    
    @unittest.skip("skip training test")
    def test_data_process(self):
        self.trainer.start_ray()
        self.trainer.data_preprocess()
       
        print(f"train_chunked_tokens: {len(self.trainer.train_chunked_tokens.take(1)[0]['input_ids'])}")
        
        self.trainer.stop_ray()

    def test_train(self):
        self.trainer.start_ray()
        self.trainer.data_preprocess()
        
        self.trainer.self_supervised_train()
        
        self.trainer.stop_ray()
        
        

    


if __name__ == "__main__":
    unittest.main()