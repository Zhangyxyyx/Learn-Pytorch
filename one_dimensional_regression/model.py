import torch
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.linear=nn.Linear(1,1)


    def forward(self, x):
        out=self.linear(x)

        return out

