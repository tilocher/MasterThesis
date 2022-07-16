# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
from datetime import datetime as dt
import os

import torch
import tqdm
import wandb
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data.dataloader import DataLoader
from log.BaseLogger import WandbLogger,LocalLogger
from tqdm import trange
import numpy as np
from SystemModels.Taylor_model import Taylor_model
from Filters.EM import EM_algorithm

class EM_Taylor_Pipeline():

    def __init__(self,taylor_model,Logger, parameters = ['R','Q', 'mu','Sigma']):


        self.Logger = Logger


        self.TaylorModel  = taylor_model

        self.HyperParams = {'Window':taylor_model.window,
                            'WindowSize': taylor_model.window_size,
                            'WindowParameter': taylor_model.window_parameters,
                            'TaylorOrder': taylor_model.taylor_order

        }

        self.Logger.SaveConfig(self.HyperParams)

        self.wandb = isinstance(Logger, WandbLogger)



        self.parameters = parameters



    def Test(self, TestLoader,itts, q_2, r_2):

        DataSet_length = len(TestLoader)

        init_values = { 'TestSamples':DataSet_length,
                        'Parameters': self.parameters,
                        'OptimizationIts': itts,
                        'q_2': q_2,
                       'r_2': r_2}

        self.Logger.SaveConfig(init_values)

        CV_Dataloader = DataLoader(TestLoader, shuffle=False, batch_size=1)

        total_loss = []

        train_input, train_target = next(iter(CV_Dataloader))

        # train_input_0 = train_input.squeeze()[:, 0]
        # train_input_1 = train_input.squeeze()[:, 1]
        #
        # train_target_0 = train_target.squeeze()[:, 0]
        # train_target_1 = train_target.squeeze()[:, 1]

        states_0, losses_0 = self.EM.EM(train_input.squeeze().T, train_target.squeeze().T, num_itts=itts, r_2=r_2, q_2=q_2, Plot='Plot')
        # states_1, losses_1 = self.EM.EM(train_input_1, train_target_1, num_itts=itts, r_2=r_2, q_2=q_2,
        #                                 wandb_true=self.wandb, Plot='Plot')

        # if losses != None and j == DataSet_length -1 :
        plt.plot(losses_0, '*g', label='loss per iteration')
        plt.grid()
        plt.xlabel('Iteration')
        plt.ylabel('Loss [dB]')
        plt.title('EM optimization convergence ')
        plt.legend()

        if self.wandb:
            wandb.log({'IterationLoss': plt})


        for j,(train_input, train_target) in enumerate(tqdm.tqdm(CV_Dataloader)):
            plot = 'Plot' if j >= DataSet_length - 5 else ''
            train_input_0 = train_input.squeeze()[:,0]
            train_input_1 = train_input.squeeze()[:, 1]

            train_target_0 = train_target.squeeze()[:, 0]
            train_target_1 = train_target.squeeze()[:, 1]

            states_0,_,_, losses_0 = self.EM.FilterEstimatedModel(train_input_0,train_target_0, Plot = plot)
            states_1,_,_, losses_1 = self.EM.FilterEstimatedModel(train_input_1,train_target_1, Plot = plot)


            total_loss.append([losses_0, losses_1])





        print('Average Test Loss: {}'.format(10*torch.log10(torch.tensor(total_loss).mean()).item()))

        if self.wandb:
            wandb.log({'Test Loss [dB]':10*torch.log10(torch.tensor(total_loss).mean()).item()})

    def TrainTaylor(self,TrainLoader):

        DataSet_length = len(TrainLoader)

        self.Logger.SaveConfig({'TrainSamples':DataSet_length})

        TrainDataset = DataLoader(TrainLoader, shuffle=False, batch_size=DataSet_length)

        train_inputs,_ = next(iter(TrainDataset))

        self.TaylorModel.fit(train_inputs.squeeze().mT)

        ssModel = self.TaylorModel.GetSysModel(train_inputs.shape[-1])
        ssModel.InitSequence(torch.zeros((2,1)), torch.eye(2))


        self.EM = EM_algorithm(ssModel, units= 'mV', parameters=self.parameters)




