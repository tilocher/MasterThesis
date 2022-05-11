import filing_paths
import torch
import torch.nn as nn
import torch.nn.functional as func
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

@variational_estimator
class FH_Net(nn.Module):

    def __init__(self):
        super(FH_Net, self).__init__()

    def Build(self,m,n):

        self.l1 = BayesianLinear(n,10)
        self.l2 = BayesianLinear(10,100)
        self.out = BayesianLinear(100,m+n)

        self.relu = nn.LeakyReLU(0.01)

    def forward(self,x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.out(x)

        return x

    def InitSequence(self,*args):
        pass
    def init_hidden(self):
        pass