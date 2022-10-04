import torch
from torch.utils.data.dataloader import DataLoader, Dataset
from PriorModels.BasePrior import BasePrior
from SystemModels.Extended_sysmdl import SystemModel


class RikPrior(BasePrior):

    def __init__(self,**kwargs):

        super(RikPrior, self).__init__()

    def fit(self,data):

        Data = DataLoader(data, batch_size=len(data))
        # obs,states = next(iter(Data))

        obs, states = next(iter(Data))

        self.ObservationCovariace = []

        for channel in range(obs.shape[-1]):
            # Y_minus_i = torch.cat((obs[..., :channel], obs[..., channel + 1:]), dim=-1).squeeze()
            # y = obs[..., channel].squeeze()
            #
            # gamma = torch.linalg.pinv(Y_minus_i.mT.bmm(Y_minus_i)).bmm(Y_minus_i.mT).bmm(y.unsqueeze(-1))
            #
            # w = y.unsqueeze(-1) - Y_minus_i.bmm(gamma)
            #
            # self.ObservationCovariace.append(w.std())

            loss_fn = torch.nn.MSELoss(reduction='mean')
            w = loss_fn(obs[...,channel],states[...,channel])
            self.ObservationCovariace.append(torch.sqrt(w))

        self.PriorLearnSetsize = len(data)

        return self.ObservationCovariace



    def getSysModel(self):

        sysModel = SystemModel('Identity',1, 'Identity',1,self.T,self.T,)