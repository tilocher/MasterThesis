# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
from datetime import datetime as dt
import os

import numpy as np
import torch
import tqdm
import wandb
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data.dataloader import DataLoader
from log.BaseLogger import WandbLogger,LocalLogger
from SystemModels.Taylor_model import Taylor_model
from Filters.KalmanSmoother import KalmanSmoother
from DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH

class EM_Taylor_Pipeline():

    def __init__(self,taylor_model: Taylor_model, Logger: LocalLogger, em_parameters = ['R','Q', 'Mu','Sigma']):

        self.Logs = {'EM_Iter_Loss':'.npy',
                'EM_Sample_Plot': '.pdf',
                'EM_Convergence':'.pdf'}

        self.Logger = Logger

        self.Logger.AddLocalLogs(self.Logs)



        self.TaylorModel  = taylor_model

        self.HyperParams = {'Window':taylor_model.window,
                            'WindowSize': taylor_model.window_size,
                            'WindowParameter': taylor_model.window_parameters,
                            'TaylorOrder': taylor_model.taylor_order

        }

        self.Logger.SaveConfig(self.HyperParams)

        self.wandb = isinstance(Logger, WandbLogger)

        self.em_parameters = em_parameters


    def TrainTaylor(self,TrainLoader):

        DataSet_length = len(TrainLoader)

        self.Logger.SaveConfig({'TrainSamples': DataSet_length})

        TrainDataset = DataLoader(TrainLoader, shuffle=False, batch_size=DataSet_length)

        train_inputs,_ = next(iter(TrainDataset))

        self.TaylorModel.fit(train_inputs.squeeze().mT)

        self.ssModel = self.TaylorModel.GetSysModel(train_inputs.shape[-1])
        self.ssModel.InitSequence(torch.zeros((2,1)), torch.eye(2))

        self.KalmanSmoother = KalmanSmoother(ssModel= self.ssModel, em_vars= self.em_parameters)
        self.KalmanSmoother.InitSequence()


    def TestTaylorEM(self, TestLoader, em_its = 10, Num_Plot_Samples = 10):

        self.TestLoader = TestLoader

        DataSet_length = len(TestLoader)

        self.Logger.SaveConfig({'TestSamples': DataSet_length,
                                'EM_Parameters': self.em_parameters})


        TestDataset = DataLoader(TestLoader, shuffle=False, batch_size=DataSet_length)

        Test_Inputs, Test_Targets = next(iter(TestDataset))

        Initial_r_2 = np.random.random()

        Initial_q_2 = np.random.random()

        self.Logger.SaveConfig({'TestSamples': DataSet_length,
                                'EM_Parameters': self.em_parameters,
                                'Initial_r_2': Initial_r_2,
                                'Initial_q_2': Initial_q_2,
                                'EM_Iterations': em_its})

        self.EM_losses = self.KalmanSmoother.em(num_itts= em_its, Observations= Test_Inputs.squeeze(), T = self.ssModel.T,
                               q_2= Initial_q_2, r_2= Initial_r_2, states= Test_Targets.squeeze())

        np.save(self.Logger.GetLocalSaveName('EM_Iter_Loss'),self.EM_losses.numpy())

        if self.wandb:
            wandb.log({'EM Iteration Losses': self.EM_losses})

        self.PlotEMResults(Test_Inputs,Test_Targets, Num_Plot_Samples= Num_Plot_Samples)

    def PlotEMResults(self,Observations, States, Num_Plot_Samples = 10):


        plt.plot(self.EM_losses, '*', color = 'g', label = 'EM Iteration Loss')
        plt.grid()
        plt.legend()
        plt.title('EM MSE Convergence')
        plt.xlabel('Iteration Step')
        plt.ylabel('MSE Loss [dB]')
        plt.savefig(self.Logger.GetLocalSaveName('EM_Convergence'))
        plt.show()


        for i in range(Num_Plot_Samples):

            index = np.random.randint(0,Observations.shape[0])
            channel = np.random.randint(0,Observations.shape[-1])

            plt.plot(Observations.squeeze()[index,:,channel], label = 'Observations', alpha = 0.4, color  = 'r')

            plt.plot(States.squeeze()[index,:,channel], label = 'Ground Truth', color = 'g')

            plt.plot(self.KalmanSmoother.Smoothed_State_Means[index,:,channel,0], label = 'EM Smoothed States', color = 'b')

            plt.legend()

            plt.xlabel('Time Steps')

            plt.ylabel('Amplitude [mV]')

            plt.title('Sample of EM filtered Observations \n'
                      f'SNR: {self.TestLoader.dataset.SNR_dB} [dB], Em Iterations: {self.Logger.GetConfig()["EM_Iterations"]},'
                      f'Channel: {channel}')

            plt.savefig(self.Logger.GetLocalSaveName('EM_Sample_Plot',prefix= f'{i}_'))

            if self.wandb:
                wandb.log({'chart':plt})

            plt.show()












