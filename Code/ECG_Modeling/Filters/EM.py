# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import datetime
import time

import matplotlib.pyplot as plt
import torch
from tqdm import trange

from Code.ECG_Modeling.SystemModels.ECG_model import ECG_signal, GetNextState, pi
import numpy as np
from Code.ECG_Modeling.Filters.Extended_RTS_Smoother import Extended_rts_smoother
from Code.ECG_Modeling.SystemModels.Extended_sysmdl import SystemModel
from Code.ECG_Modeling.Filters.EKF import ExtendedKalmanFilter

class EM_algorithm():

    def __init__(self,ssModel: SystemModel, parameters: tuple = ('R'), **kwargs):

        self.ssModel = ssModel
        self.m = ssModel.m
        self.n = ssModel.n


        self.parameters = parameters

        if 'fs' in kwargs.keys(): self.fs = kwargs['fs']


        if 'deltaT' in  kwargs.keys(): self.fs = 1/kwargs['deltaT']


        self.plot_title = kwargs['Plot_title'] if 'Plot_title' in kwargs.keys() else ''


        self.units = kwargs['units'] if 'units' in kwargs.keys() else ''



    def FilterEstimatedModel(self, observations: torch.Tensor, states: torch.Tensor, Plot=''):



        KF = ExtendedKalmanFilter(self.ssModel)
        KF.InitSequence(self.ssModel.m1x_0, self.ssModel.m2x_0)
        RTS = Extended_rts_smoother(self.ssModel)

        MSE_RTS_linear_arr = torch.empty((self.batch_size,self.channels)) if states != None else None
        loss_rts = torch.nn.MSELoss(reduction='mean')

        filtered_states = torch.empty((self.batch_size, self.channels, self.ssModel.m, self.T))
        error_cov = torch.empty((self.batch_size,self.channels, self.ssModel.m, self.ssModel.m, self.T))
        KGs = torch.empty((self.batch_size, self.channels, self.ssModel.m, self.ssModel.m, self.T))
        # Run Loop

        with torch.no_grad():
            for j in range(self.batch_size):
                for i in range(self.channels):
                    KF.GenerateSequence(observations[j,i].reshape((self.n,-1)), KF.T_test)
                    RTS.GenerateSequence(KF.x, KF.sigma, RTS.T_test)

                    if states != None:

                        MSE_RTS_linear_arr[j,i] = loss_rts(self.ssModel.h(RTS.s_x,0).squeeze(), states[j,i].squeeze()).item()


                    filtered_states[j,i] = RTS.s_x
                    error_cov[j,i] = RTS.s_sigma
                    KGs[j,i] = RTS.SG_array.reshape(self.ssModel.m , self.ssModel.m , self.T)

            if states != None:
                print('Mean RTS loss Estimated Model: {} [dB]'.format(
                10 * torch.log10(MSE_RTS_linear_arr.mean()).item()))

        self.EstRTS = RTS

        if Plot != '':

            if 'fs' in self.__dict__:
                t = np.arange(start=0, stop=self.T / (self.fs), step=1 / (self.fs))
            else:
                t = np.linspace(0,self.T, self.T)

            rand_sample = np.random.randint(0,self.batch_size)
            rand_channel = np.random.randint(0,self.channels)

            if states != None:
                plt.plot(t,states[rand_sample,rand_channel].squeeze(), label='Noiseless data', alpha=0.8, color='g')
            plt.plot(t,observations[rand_sample,rand_channel].squeeze(), label='noisy data', alpha=0.3, color='r')

            plt.plot(t,filtered_states[rand_sample,rand_channel,0].squeeze(), label='Estimated State', color='b')

            if 'plot_title' in self.__dict__:
                title_string = 'Filtered Signal Sample, {}'.format(self.plot_title)
            else:
                title_string = 'Filtered Signal Sample'

            plt.title(title_string)

            xlabel_string = 'Time steps' if not 'fs' in self.__dict__ else 'Time [s]'
            ylabel_string = 'Amplitude [{}]'.format(self.units)

            plt.xlabel(xlabel_string)
            plt.ylabel(ylabel_string)
            plt.legend()
            if Plot != 'Plot':
                plt.savefig('..\\Plots\\EM_filtered_Sample_{}_{}.pdf'.format(str(datetime.datetime.now().date()),Plot))
            plt.show()
            plt.clf()

        return filtered_states, error_cov, KGs, MSE_RTS_linear_arr


    def EM(self,observation: torch.Tensor, state: torch.Tensor = None, q_2 = 1., r_2 = 1. ,
           Plot = 'Plot', num_itts = 10, Q = None):


        Q = q_2 * torch.eye(self.m) if Q == None else Q
        R = r_2 * torch.eye(self.n)

        losses = []

        with torch.no_grad():

            batch_size = self.batch_size  = observation.shape[0]
            T = self.T = observation.shape[-1]
            channels = self.channels = 1 if len(observation.shape) <= 2 else observation.shape[1]

            self.ssModel.T = self.ssModel.T_test = T

            observation = observation.reshape((batch_size,channels,self.n,T))

            if state != None:
                state = state.reshape((batch_size,channels, self.n, T))

            for i in range(num_itts):

                # Create the system model
                if 'R' in self.parameters:
                    self.ssModel.UpdateR(R)

                if 'Q' in self.parameters:
                    self.ssModel.UpdateQ(Q)

                filtered_states,error_cov,SGs,loss = self.FilterEstimatedModel(observation,state , Plot= Plot)

                error_cov = error_cov.mean(0).reshape((channels, self.m,self.m,-1))
                SGs = SGs.mean(0).reshape((channels, self.m,self.m,-1))
                Smoothed_Covariance = torch.einsum('cnnt,cmmt->cnmt', (error_cov[:,:,:,1:], SGs.transpose(1, 2)[:,:,:,:-1])) / batch_size
                E_xx = torch.einsum('bcmt,bcnt->cmnt',(filtered_states,filtered_states))/batch_size

                U_xx = (E_xx + error_cov).mean(0).mean(-1)
                U_yx = (torch.einsum('bcnt,bcmt->cnmt',(observation,filtered_states))/batch_size).mean(-1).mean(0)

                U_yy = (torch.einsum('bcnt,bcmt->cnmt',(observation,observation))/batch_size).mean(0).mean(-1)
                C = U_yx @ torch.inverse(U_xx)

                if 'C' in self.parameters:
                    self.ssModel.setHJac(lambda x,t: C)

                if 'R' in self.parameters:
                    R = U_yy - C @ U_yx.T

                V_xx = (E_xx[:,:,:,:-1] + error_cov[:,:,:,:-1]).mean(0)
                V_x1x1 = (E_xx[:,:,:,1:] + error_cov[:,:,:,1:]).mean(0)
                V_x1x = (((torch.einsum('bcnt,bcmt->cnmt', (filtered_states[:,:,:,1:], filtered_states[:,:,:,:-1]))/batch_size)
                          + Smoothed_Covariance).reshape((self.channels,self.m,self.m,-1))).mean(0)

                if 'A' in self.parameters or 'Q' in self.parameters:


                    if 'A' in self.parameters:
                        A = torch.bmm(V_x1x.reshape(-1, self.m, self.m),
                                      torch.inverse(V_xx.reshape(-1, self.m, self.m))).reshape((self.m, self.m, -1))

                        # self.ssModel.setFJac(lambda x,t: A[:,:,t-1] if t > 0 else torch.eye(self.n) )
                        A_mean = A.mean(-1)
                        self.ssModel.setFJac(lambda x,t: A_mean)

                    if 'Q' in self.parameters and 'A' not in self.parameters:
                        Q = (V_x1x1.mean(-1) - V_x1x.mean(-1))# - V_x1x.mean(-1).T + V_xx.mean(-1)) #(V_x1x1 - torch.bmm(A.T,V_x1x.T).T).mean()
                    elif 'Q' in self.parameters and 'A' in self.parameters:
                        Q = (V_x1x1.mean(-1) - V_x1x.mean(-1) - V_x1x.mean(-1).T + V_xx.mean(-1))
                if torch.any(torch.abs(torch.linalg.eigvals(Q)) < 0 ):
                    print('noonononono')

                self.ssModel.InitSequence(filtered_states.mean((0,1))[:,0], error_cov.mean(0)[:,:,0])

                if loss != None:
                    losses.append(10*torch.log10(loss.mean()).item())


            if loss != None:

                plt.plot(losses,'*g', label = 'loss per iteration')
                plt.grid()
                plt.xlabel('Iteration')
                plt.ylabel('Loss [dB]')
                plt.title('EM optimization convergence '+ self.plot_title)
                plt.legend()
                plt.savefig('..\\Plots\\EM_convergence_plot.pdf')
                plt.show()


if __name__ == '__main__':

    def f(x,t):
        return x
    def h(x,t):
        return x[0]

    ToyModel = SystemModel(f,1,h,1,100,100,1,1)
    ToyModel.InitSequence(torch.randn((1)) , torch.randn((1,1))**2)

    # Load dataset
    ecg_signal = torch.load('..\\Datasets\\Synthetic\\17.06.22--14.32.pt')

    # Get noisy data and noiseless data
    data_noisy = ecg_signal.traj_noisy[:, 2, :]
    data_noiseless = ecg_signal.traj[:, 2, :]

    from Code.ECG_Modeling.DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH

    snr = 20

    loader = PhyioNetLoader_MIT_NIH(1,1,SNR_dB=snr,random_sample=False)



    obs,state = loader.GetData(2)


    EM = EM_algorithm(ToyModel,parameters = ('R','A'), Plot_title= 'SNR: {} [dB]'.format(snr), units= 'mV')
    EM.EM(obs,state,num_itts=20, q_2= 0.002,Plot= 'SNR_{}'.format(snr))