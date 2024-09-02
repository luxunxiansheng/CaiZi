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

from typing import Dict, List


class ChunkProcessor:
    def __init__(self, block_size: int = 4, stride: int = 4):
        self.block_size = block_size
        self.stride = stride

    def __call__(
        self, data: Dict[str, List[int]]
    ) -> Dict[str, List[Dict[str, List[int]]]]:
        train_ids = data["train"]
        train_data_items = []

        if len(train_ids) >= self.block_size:

            for i in range(0, len(train_ids) - self.block_size, self.stride):
                input_chunk = train_ids[i : i + self.block_size]
                target_chunk = train_ids[i + 1 : i + self.block_size + 1]
                train_data_items.append(
                    {"input_ids": input_chunk, "target_ids": target_chunk}
                )

        else:
            input_chunk = train_ids[: self.block_size]
            target_chunk = train_ids[1 : self.block_size + 1]
            train_data_items.append(
                {"input_ids": input_chunk, "target_ids": target_chunk}
            )

        validate = data["validate"]
        validate_data_items = []
        if len(validate) >= self.block_size:
            for i in range(0, len(validate) - self.block_size, self.stride):
                input_chunk = validate[i : i + self.block_size]
                target_chunk = validate[i + 1 : i + self.block_size + 1]
                validate_data_items.append(
                    {"input_ids": input_chunk, "target_ids": target_chunk}
                )

        else:
            input_chunk = validate[: self.block_size]
            target_chunk = validate[1 : self.block_size + 1]
            validate_data_items = [
                {"input_ids": input_chunk, "target_ids": target_chunk}
            ]

        return {"train": train_data_items, "validate": validate_data_items}
