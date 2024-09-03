import unittest


from config import gpt2_cfg as cfg
from fundation_model_trainer import RayGPT2FundationModelTrainer
from preprocessor.token_processor import TikTokenizer

@unittest.skip("skip training test")
class TestGPT2FundationModelTrainer(unittest.TestCase):
    def setUp(self) -> None:

        self.trainer = RayGPT2FundationModelTrainer(cfg)

    @unittest.skip("skip training test")
    def test_load_data(self):

        self.trainer.load_data()

        validate_dataset = self.trainer.validate_chunked_tokens

        tokenizer = TikTokenizer()
        for row in validate_dataset.iter_rows():
            
            input_ids = row["input_ids"]
            target_ids = row["target_ids"]

            print("-----------------------------")
            print(tokenizer.decode(input_ids))
            print("..............................")
            print(tokenizer.decode(target_ids))
            print("*******************************")

            break

    @unittest.skip("skip training test")
    def test_train_with_plain_text(self):

        self.trainer.self_supervised_train()


if __name__ == "__main__":
    unittest.main()
