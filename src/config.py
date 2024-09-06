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

import os
from hydra import compose, initialize,core

core.global_hydra.GlobalHydra.instance().clear()
initialize(config_path="../config",version_base=None)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

gpt2_cfg = compose(config_name="openwebtext")
gpt2_cfg.project_root = project_root


gpt2_nano_cfg = compose(config_name="shakespeare")
gpt2_nano_cfg.project_root = project_root