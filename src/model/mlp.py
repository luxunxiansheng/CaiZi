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
    def __init__(self, dimension_embedding: int):
        super().__init__()
      
        self.fc=nn.Linear(dimension_embedding, 4 * dimension_embedding)
        self.gelu=nn.GELU()
        self.projection=nn.Linear(4 * dimension_embedding, dimension_embedding)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.projection(x)
        return x
