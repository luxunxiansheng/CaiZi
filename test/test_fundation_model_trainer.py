import unittest



from config import gpt2_cfg as cfg 
from fundation_model_trainer import RayGPT2FundationModelTrainer
from token_processor import TokenProcessor

class TestGPT2FundationModelTrainer(unittest.TestCase):
    def setUp(self) -> None:
        
        self.trainer = RayGPT2FundationModelTrainer(cfg)
        self.trainer.start_ray()
        self.trainer.data_preprocess()
    
    def tearDown(self) -> None:
        self.trainer.stop_ray()
    
    
    @unittest.skip("skip training test")
    def test_data_process(self):
       
        tokenizer_class = TokenProcessor.create(cfg['ray_data']['tokenizer_class']['name'])
        tokenizer_args =  cfg['ray_data']['tokenizer_class']['args']
        tokenizer= tokenizer_class(**tokenizer_args)
    
        validate_dataset = self.trainer.validate_chunked_tokens
    
    
        for row in validate_dataset.iter_rows():
            input_ids = row["input_ids"]
            target_ids = row["target_ids"]
            
            print("-----------------------------")
            print(tokenizer.decode(input_ids))
            print("..............................")
            print(tokenizer.decode(target_ids))
            print("*******************************")

    #@unittest.skip("skip training test")
    def test_train(self):

        self.trainer.self_supervised_train()
        

        

    


if __name__ == "__main__":
    unittest.main()