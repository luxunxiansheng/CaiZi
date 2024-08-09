import unittest

from config import gpt2_cfg as cfg 
from fundation_model_trainer import RayGPT2FundationModelTrainer

class TestGPT2FundationModelTrainer(unittest.TestCase):
    def setUp(self) -> None:
        
        cfg["ray_train"]["num_epoch_per_worker"] = 5
        
        self.trainer = RayGPT2FundationModelTrainer(cfg)
    
    
    def test_train(self):
        self.trainer.self_supervised_train()

    


if __name__ == "__main__":
    unittest.main()