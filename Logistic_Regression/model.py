import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.lr=nn.Linear(2,1)
        self.sm=nn.Sigmoid()

    def forward(self, x):
        x=self.lr(x)
        out=self.sm(x)
        return out