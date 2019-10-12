import torch
import torch.nn as nn


class poly_regression(nn.Module):
    def __init__(self):
        super(poly_regression,self).__init__()
        self.poly=nn.Linear(3,1)

    def forward(self, x):
        out=self.poly(x)
        return out
