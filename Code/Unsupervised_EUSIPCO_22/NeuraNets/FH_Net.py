import filing_paths
import torch
import torch.nn as nn
import torch.nn.functional as func


class FH_Net(nn.Module):

    def __init__(self):
        super(FH_Net, self).__init__()

    def Build(self,m,n):

        self.l1 = nn.Linear(n,10)
        self.l2 = nn.Linear(10,30)
        self.out = nn.Linear(30,m)

        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.out(x)

        return x

    def InitSequence(self,*args):
        pass
    def init_hidden(self):
        pass