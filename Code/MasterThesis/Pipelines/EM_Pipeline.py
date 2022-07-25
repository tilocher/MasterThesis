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

class EM_Taylor_Pipeline(nn.Module):

    def __init__(self,PriorModel, Logger: LocalLogger, em_parameters = ['R','Q', 'Mu','Sigma']):
        super(EM_Taylor_Pipeline, self).__init__()
        self.Logs = {'EM_Iter_Loss':'.npy',
                'EM_Sample_Plot': '.pdf',
                'EM_Convergence':'.pdf',
                'KGain':'.npy',
                'Pipelines':'.pt',
                'Prior_Plot':'.pdf'}

        self.Logger = Logger

        self.Logger.AddLocalLogs(self.Logs)



        self.PriorModel  = PriorModel

        self.HyperParams = {'Window':PriorModel.window,
                            'WindowSize': PriorModel.window_size,
                            'WindowParameter': PriorModel.window_parameters,
                            'TaylorOrder': PriorModel.taylor_order

        }

        self.Logger.SaveConfig(self.HyperParams)

        self.wandb = isinstance(Logger, WandbLogger)

        self.em_parameters = em_parameters

    def save(self):
        torch.save(self, self.Logger.GetLocalSaveName('Pipelines'))

    def TrainPrior(self,TrainLoader):

        try:
            self._TrainPrior(TrainLoader)
            self.PlotPrior()

        except:
            self.Logger.ForceClose()
            raise

    def _TrainPrior(self,TrainLoader):

        DataSet_length = len(TrainLoader)

        self.Logger.SaveConfig({'TrainSamples': DataSet_length})

        TrainDataset = DataLoader(TrainLoader, shuffle=False, batch_size=DataSet_length)

        train_inputs,_ = next(iter(TrainDataset))

        self.PriorModel.fit(train_inputs.squeeze().mT)

        self.ssModel = self.PriorModel.GetSysModel(train_inputs.shape[-1])
        # self.ssModel.f = lambda x,t: x
        # self.ssModel.f = self.proxy
        self.ssModel.InitSequence(torch.zeros((self.ssModel.m,1)), torch.eye(self.ssModel.m))

        self.KalmanSmoother = KalmanSmoother(ssModel= self.ssModel, em_vars= self.em_parameters)
        self.KalmanSmoother.InitSequence()

        self.PlotPrior()



    def PlotPrior(self):

        self.ssModel.GenerateSequence(self.ssModel.Q, self.ssModel.R, self.ssModel.T)


        sample = self.ssModel.x.T[:,0]

        # plt.plot(self.ssModel.x.T, label = 'Learned Prior')
        t = np.arange(start=0, stop=sample.shape[0] / (360), step=1 / (360))
        fig, ax = plt.subplots(dpi=200)

        ax.plot(t,sample, label = 'Learned Prior', color = 'g')
        # ax.plot(t, noise[:, 0], label='Noisy observation', color='r', alpha=0.4)
        ax.grid()
        ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude [mV]')
        ax.set_title('Prior learned by windowed Taylor Approximation \n'
                     'Window size: {}, Window type: {}'.format(self.PriorModel.window_size,self.PriorModel.window))

        axins = ax.inset_axes([0.05, 0.5, 0.4, 0.4])
        axins.plot(t, sample, color='g')
        axins.get_xaxis().set_visible(False)
        axins.get_yaxis().set_visible(False)

        x1, x2, y1, y2 = 0.4, 0.6, torch.min(sample).item(), torch.max(sample).item()
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axins.grid()

        ax.indicate_inset_zoom(axins, edgecolor="black")

        plt.savefig(self.Logger.GetLocalSaveName('Prior_Plot'))
        plt.show()


    def TestEM(self, TestLoader, em_its = 10, Num_Plot_Samples = 10):

        try:
            self._TestEM(TestLoader,em_its,Num_Plot_Samples)

        except:
            self.Logger.ForceClose()
            raise

    def proxy(self,x,t):
        return x

    def _TestEM(self, TestLoader, em_its = 10, Num_Plot_Samples = 10):

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

        self.KalmanSmoother.InitMean(Test_Inputs[:,0,0].unsqueeze(-1))

        # init_state = torch.zeros_like(Test_Inputs)
        # init_state = init_state[...,:(self.ssModel.m-2)]
        # init_state = torch.concat((Test_Targets, init_state), -1)
        # self.KalmanSmoother.InitMean(init_state.unsqueeze(-1))



        self.EM_losses = self.KalmanSmoother.em(num_itts= em_its, Observations= Test_Inputs.squeeze(), T = self.ssModel.T,
                               q_2= Initial_q_2, r_2= Initial_r_2, states= Test_Targets.squeeze())

        np.save(self.Logger.GetLocalSaveName('EM_Iter_Loss'),self.EM_losses.numpy())
        np.save(self.Logger.GetLocalSaveName('KGain'), self.KalmanSmoother.Kalman_Gains.numpy())

        if self.wandb:
            wandb.log({'EM Iteration Losses': self.EM_losses})
            wandb.log({'Final Loss [dB]': self.EM_losses[-1]})

        self.PlotEMResults(Test_Inputs,Test_Targets, Num_Plot_Samples= Num_Plot_Samples)

        self.save()


    def PlotEMResults(self,Observations, States, Num_Plot_Samples = 10):


        plt.plot(self.EM_losses, '*', color = 'g', label = 'EM Iteration Loss')
        plt.grid()
        plt.legend()
        plt.title('EM MSE Convergence')
        plt.xlabel('Iteration Step')
        plt.ylabel('MSE Loss [dB]')
        plt.savefig(self.Logger.GetLocalSaveName('EM_Convergence'))
        if self.wandb:
            wandb.log({'chart':plt})
        else:
            plt.show()

        t = np.arange(start=0, stop=self.TestLoader.dataset.fs / (360), step=1 / (360))

        for i in range(Num_Plot_Samples):

            # index = np.random.randint(0,Observations.shape[0])
            # channel = np.random.randint(0,Observations.shape[-1])
            index = i
            channel = 0

            fig, ax = plt.subplots(dpi=200)

            observation = Observations.squeeze()[index,:,channel]
            state = States.squeeze()[index,:,channel]
            smoothed_states = self.KalmanSmoother.Smoothed_State_Means[index,:,channel,0]

            ax.plot(t,observation, label = 'Observations', alpha = 0.4, color  = 'r')

            ax.plot(t,state, label = 'Ground Truth', color = 'g')

            # ax.plot(t,smoothed_states, label = 'EM Smoothed States', color = 'b')
            ax.plot(t,smoothed_states, label = 'Fitted prior', color = 'b')

            ax.legend()

            ax.set_xlabel('Time Steps')

            ax.set_ylabel('Amplitude [mV]')

            ax.set_title('Sample of EM filtered Observations \n'
                      f'SNR: {self.TestLoader.dataset.SNR_dB} [dB], Em Iterations: {self.Logger.GetConfig()["EM_Iterations"]},'
                      f'Channel: {channel}')

            axins = ax.inset_axes([0.05, 0.5, 0.4, 0.4])


            axins.plot(t,state,color = 'g')
            axins.plot(t, smoothed_states, color='b')
            axins.get_xaxis().set_visible(False)
            axins.get_yaxis().set_visible(False)

            x1, x2, y1, y2 = 0.4, 0.6, torch.min(torch.min(state),torch.min(smoothed_states)).item(), \
                             torch.max(torch.max(state),torch.max(smoothed_states)).item()
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticklabels([])
            axins.set_yticklabels([])
            axins.grid()


            ax.indicate_inset_zoom(axins, edgecolor="black")

            plt.savefig(self.Logger.GetLocalSaveName('EM_Sample_Plot',prefix= f'{i}_'))

            if self.wandb:
                wandb.log({'chart':plt})
            else:
                plt.show()












