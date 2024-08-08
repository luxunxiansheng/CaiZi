import unittest
import math
import torch

from torch.optim import SGD
from model.gpt_lr_scheduler import GPTLRScheduler

class TestGPTLRScheduler(unittest.TestCase):
    def test_get_lr(self):
        
        model = torch.nn.Linear(10, 10)
        optimizer = SGD(model.parameters(), lr=0.1)
        
        scheduler = GPTLRScheduler(optimizer, warmup_steps=100, max_steps=1000, max_lr=0.1, min_lr=0.01)
        
        # Test warmup phase
        for epoch in range(100):
            lrs = scheduler.get_lr()
            expected_lr = [0.1 * (epoch + 1) / 100]
            self.assertEqual(lrs, expected_lr)
            scheduler.step()
        
        # Test decay phase
        for epoch in range(100, 1000):
            lrs = scheduler.get_lr()
            decay_ratio = (epoch - 100) / (1000 - 100)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            expected_lr = [0.01 + coeff * (0.1 - 0.01)]
            self.assertEqual(lrs, expected_lr)
            scheduler.step()
        
        # Test after max_steps
        lrs = scheduler.get_lr()
        expected_lr = [0.01]
        self.assertEqual(lrs, expected_lr)

if __name__ == '__main__':
    unittest.main()