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



import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dimension_embedding,dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimension_embedding, 4 * dimension_embedding),
            nn.GELU(),
            nn.Linear(4 * dimension_embedding, dimension_embedding),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)
