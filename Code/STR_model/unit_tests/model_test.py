import torch
import sys
sys.path.append('../')
from model import STR_Model, STR_Model_Longer_512, STR_Model_Longer_1024

import unittest

class TestSTR_Model(unittest.TestCase):
    def setUp(self):
        self.model = STR_Model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss = lambda x, y: torch.sum(torch.abs(x - y))
        self.width_factor = 4
        self.width_extra = 1

    def test_singular_forward(self):
        test_widths = [100, 102, 104, 106, 108] 
        for width in test_widths:
            x = torch.randn(1, 1, 60, width)
            x = x.to(self.device)
            y = self.model(x)
            self.assertEqual(y.size(), torch.Size([width // self.width_factor + self.width_extra, 1, 4]))
        
    def test_batch_same_width_forward(self):
        width = 100
        out_width = width // self.width_factor + self.width_extra
        x = torch.randn(32, 1, 60, width)
        x = x.to(self.device)
        y = self.model(x)
        self.assertEqual(y.size(), torch.Size([out_width, 32, 4]))
        
    def test_batch_diff_width_forward(self):
        final_width = 150
        out_width = final_width // self.width_factor + self.width_extra
        x1 = torch.randn(3, 1, 60, 100)
        x2 = torch.randn(5, 1, 60, 150)

        # Pad x1 to match x2
        x1 = torch.cat((x1, torch.zeros(3, 1, 60, 50)), dim=3)
        x = torch.cat((x1, x2), dim=0)
        x = x.to(self.device)
        
        y = self.model(x)
        self.assertEqual(y.size(), torch.Size([out_width, 8, 4]))
        
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

class TestSTR_Model_Longer_512(TestSTR_Model):
    def setUp(self):
        self.model = STR_Model_Longer_512()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss = lambda x, y: torch.sum(torch.abs(x - y))
        self.width_factor = 1
        self.width_extra = 0

class TestSTR_Model_Longer_1024(TestSTR_Model):
    def setUp(self):
        self.model = STR_Model_Longer_1024()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss = lambda x, y: torch.sum(torch.abs(x - y))
        self.width_factor = 1
        self.width_extra = 0
    
if __name__ == '__main__':
    unittest.main(verbosity=2)