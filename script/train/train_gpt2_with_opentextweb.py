""" 
###########################################
#
# Author: Bin.Li                          
# Email:  ornot2008@yahoo.com
# MIT License
# Copyright (c) 2025 debutpark.com 
#
###########################################
"""

from config import gpt2_cfg as cfg
from fundation_model_trainer import RayGPT2FundationModelTrainer

if __name__ == "__main__":
    trainer = RayGPT2FundationModelTrainer(cfg)
    trainer.load_data()
    trainer.self_supervised_train()
