import torch
import sys
sys.path.append('../')
from model import STR_Model

import unittest

class TestSTRModel(unittest.TestCase):
    def setUp(self):
        self.model = STR_Model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss = lambda x, y: torch.sum(torch.abs(x - y))

    def test_singular_forward(self):
        x = torch.randn(1, 1, 60, 100)
        x = x.to(self.device)
        y = self.model(x)
        self.assertEqual(y.size(), torch.Size([26, 1, 4]))
        
    def test_batch_same_width_forward(self):
        x = torch.randn(32, 1, 60, 100)
        x = x.to(self.device)
        y = self.model(x)
        self.assertEqual(y.size(), torch.Size([26, 32, 4]))
        
    def test_batch_diff_width_forward(self):
        x1 = torch.randn(3, 1, 60, 100)
        x2 = torch.randn(5, 1, 60, 150)

        # Pad x1 to match x2
        x1 = torch.cat((x1, torch.zeros(3, 1, 60, 50)), dim=3)
        x = torch.cat((x1, x2), dim=0)
        x = x.to(self.device)
        
        y = self.model(x)
        self.assertEqual(y.size(), torch.Size([150//4 + 1, 8, 4]))
        
    def test_singular_backward(self):
        x = torch.randn(1, 1, 60, 100)
        x = x.to(self.device)
        y = self.model(x)
        loss = self.loss(y, torch.zeros_like(y))
        loss.backward()
        
    def test_batch_backward(self):
        x = torch.randn(32, 1, 60, 100)
        x = x.to(self.device)
        y = self.model(x)
        loss = self.loss(y, torch.zeros_like(y))
        loss.backward()

if __name__ == '__main__':
    unittest.main(verbosity=2)