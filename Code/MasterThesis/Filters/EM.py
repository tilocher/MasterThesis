# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import datetime
import os
import time

import matplotlib.pyplot as plt
import torch
import wandb
from tqdm import trange

# from Code.ECG_Modeling.SystemModels.ECG_model import ECG_signal, GetNextState, pi
import numpy as np
from Extended_RTS_Smoother import Extended_rts_smoother
from SystemModels.Extended_sysmdl import SystemModel
from EKF import ExtendedKalmanFilter
from log.BaseLogger import WandbLogger,LocalLogger
from utils import reshape
import pykalman

class EM_algorithm():

    def __init__(self,ssModel: SystemModel, parameters: list = ('R','mu','Sigma'), **kwargs):


        self.ssModel = ssModel
        self.m = ssModel.m
        self.n = ssModel.n


        self.parameters = parameters


        if 'fs' in kwargs.keys(): self.fs = kwargs['fs']


        if 'deltaT' in  kwargs.keys(): self.fs = 1/kwargs['deltaT']


        self.plot_title = kwargs['Plot_title'] if 'Plot_title' in kwargs.keys() else ''


        self.units = kwargs['units'] if 'units' in kwargs.keys() else ''

        self.random_plot = False

    def SetRandomPlot(self,value: bool):
        self.random_plot = value



    def FilterEstimatedModel(self, observations: torch.Tensor, states: torch.Tensor, Plot=''):



        KF = ExtendedKalmanFilter(self.ssModel)
        KF.InitSequence(self.ssModel.m1x_0, self.ssModel.m2x_0)
        RTS = Extended_rts_smoother(self.ssModel)

        MSE_RTS = torch.empty(1) if states != None else None
        loss_rts = torch.nn.MSELoss(reduction='mean')


        # Run Loop

        with torch.no_grad():
            observations = observations.reshape((self.n,-1))

            KF.GenerateSequence(observations, KF.T_test)
            RTS.GenerateSequence(KF, RTS.T_test)

            if states != None:

                MSE_RTS = loss_rts(self.ssModel.h(RTS.s_x,0).squeeze(), states.squeeze())


            filtered_states = RTS.s_x
            error_cov = RTS.s_sigma
            # KGs = RTS.SG_array.reshape(self.ssModel.m , self.ssModel.m , self.T)
            Smoothed_cov = RTS.s_smooth_prior

            if states != None:
                print('Mean RTS loss Estimated Model: {} [dB]'.format(
                10 * torch.log10(MSE_RTS).item()))

        self.EstRTS = RTS

        if Plot != '':

            if 'fs' in self.__dict__:
                t = np.arange(start=0, stop=self.T / (self.fs), step=1 / (self.fs))
            else:
                t = np.linspace(0,self.T, self.T)

            for channel in range(observations.shape[0]):
                # if states != None:
                    # plt.plot(t,states[channel].squeeze(), label='Noiseless data', alpha=0.8, color='g')
                # plt.plot(t,observations[channel].squeeze(), label='noisy data', alpha=0.3, color='r')

                plt.plot(t,self.ssModel.h(filtered_states,t)[channel].squeeze(), label='Estimated State', color='b')

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
                # if Plot == 'Plot':
                #     wandb.log({'TestSample': plt})

                #     plt.savefig(os.path.dirname(os.path.realpath(__file__)) +
                #                 '\\..\\Plots\\EM_filtered_sample\\EM_filtered_Sample_{}_{}.pdf'.format(str(datetime.datetime.now().date()),Plot))
                plt.show()
                plt.clf()

        return filtered_states, error_cov, Smoothed_cov, MSE_RTS


    def EM(self,observation: torch.Tensor, state: torch.Tensor = None, q_2 = 1., r_2 = 1. ,
           Plot = 'Plot', num_itts = 10, Q = None, window_size = 20):


        Q = q_2 * torch.eye(self.m) if Q == None else Q
        R = r_2 * torch.eye(self.n)

        self.ssModel.UpdateCovariance_Matrix(Q, R)

        KF = ExtendedKalmanFilter(self.ssModel)
        KF.InitSequence(self.ssModel.m1x_0, self.ssModel.m2x_0)

        KF.GenerateSequence(observation, self.ssModel.T)

        # Compare to pykalman
        kf = pykalman.KalmanFilter(
            transition_matrices=self.ssModel.getFJacobian(0,0).numpy(), observation_matrices=self.ssModel.getHJacobian(0,0).numpy(),
            transition_covariance=Q.numpy(), observation_covariance=R.numpy(),
            transition_offsets=None, observation_offsets=None,
            initial_state_mean=self.ssModel.m1x_0.squeeze().numpy().T, initial_state_covariance=self.ssModel.m2x_0.numpy(),
            em_vars=['transition_covariance', 'observation_covariance',
                     'initial_state_mean', 'initial_state_covariance']
        )
        # fs = kf.filter(observation.numpy().T)
        kf = kf.em(observation.numpy().T,n_iter= 10)
        print(kf.observation_covariance)

        fs = kf.smooth(observation.numpy().T)





        # plt.plot(KF.x[0], label='Mine')
        plt.plot(fs[0][:,0], label = 'pykalman')
        # plt.legend()
        # plt.show()

        losses = []

        with torch.no_grad():

            T = self.T = observation.shape[-1]

            self.ssModel.T = self.ssModel.T_test = T


            observation = reshape(observation.squeeze(),(self.n, T))

            # observation = observation.reshape((self.n,T))

            if state != None:
                state = state.reshape((self.n, T))

            if 'Q' in self.parameters:
                self.ssModel.UpdateQ(Q)

            for i in range(num_itts):

                if i == 8:
                    print('dfsdf')
                # Create the system model
                if 'R' in self.parameters:
                    self.ssModel.UpdateR(R)


                if i == num_itts -1:
                    Plot_ = Plot
                else:
                    Plot_ = ''

                filtered_states,error_cov,pairwise_cov,loss = self.FilterEstimatedModel(observation,state , Plot= Plot_)

                observation = observation.reshape((-1,self.n,1))
                filtered_states = filtered_states.T.unsqueeze(-1)
                error_cov = error_cov.permute((2,0,1))
                pairwise_cov = pairwise_cov.permute((2,0,1))


                # Smoothed_Covariance = torch.bmm(error_cov[1:], SGs.mT[:-1])
                E_xx = torch.bmm(filtered_states,filtered_states.mT)

                U_xx = E_xx + error_cov
                U_yx = torch.bmm(observation,filtered_states.mT)

                U_yy = torch.bmm(observation,observation.mT)
                C = U_yx @ torch.linalg.pinv(U_xx)

                if 'C' in self.parameters:
                    self.ssModel.setHJac(lambda x,t: C[t])

                if 'R' in self.parameters and 'C' in self.parameters:
                    R = (U_yy - torch.bmm(C , U_yx.mT)).mean(0)
                elif 'R' in self.parameters and 'C' not in self.parameters:
                    C = self.ssModel.HJac(filtered_states,0).repeat(T, 1, 1)
                    R = (U_yy - torch.bmm(C, U_yx.mT) - torch.bmm( U_yx, C.mT) + torch.bmm(torch.bmm(C,U_xx),C.mT)).mean(0)

                err = torch.empty((T-1,self.m,1))
                for t in range(T-1):
                    F = self.ssModel.getFJacobian(filtered_states[t],t)
                    err[t] = (filtered_states[t+1] - torch.mm(F,filtered_states[t]))

                # F = self.ssModel.getFJacobian(0,0)
                # err = (filtered_states[1:] - torch.mm(F,filtered_states[:-1]))

                V_xx = (E_xx[:-1,:,:] + error_cov[:-1,:,:])
                V_x1x1 = (E_xx[1:,:,:] + error_cov[1:,:,:])
                # V_x1x = torch.bmm(filtered_states[1:],filtered_states[:-1].mT) + pairwise_cov[1:]
                V_x1x =  pairwise_cov[1:]

                if 'A' in self.parameters or 'Q' in self.parameters:


                    if 'A' in self.parameters:
                        A = torch.bmm(V_x1x,torch.linalg.pinv(V_xx))

                        # self.ssModel.setFJac(lambda x,t: A[:,:,t-1] if t > 0 else torch.eye(self.n) )
                        A_mean = A.mean(0)
                        self.ssModel.setFJac(lambda x,t: A_mean)

                    if 'Q' in self.parameters and 'A' in self.parameters:
                        Q = (V_x1x1 - torch.bmm(A,V_x1x.mT)).mean(0)# - V_x1x.mean(-1).T + V_xx.mean(-1)) #(V_x1x1 - torch.bmm(A.T,V_x1x.T).T).mean()

                    elif 'Q' in self.parameters and 'A' not in self.parameters:
                        A = self.ssModel.FJac(filtered_states, 0).repeat(T-1, 1, 1)
                        # Q = (V_x1x1 - torch.bmm(A,V_x1x.mT) -  torch.bmm(V_x1x,A.mT) + torch.bmm(torch.bmm(A,V_xx),A.mT))
                        Vx1A = torch.bmm(V_x1x,A.mT)
                        Q = torch.bmm(err,err.mT) + torch.bmm(torch.bmm(A,error_cov[:-1]),A.mT) + error_cov[1:] -Vx1A - Vx1A.mT
                        self.ssModel.UpdateQ(Q.mean(0))
                        # Inter = Q.unfold(0, window_size, 1).mean(-1)
                        # Q = torch.cat((Inter, Q[-window_size:]), dim=0)
                        # self.ssModel.SetQ_array(Q)


                if 'Sigma' in self.parameters:
                    next_init_cov = error_cov[0]
                else:
                    next_init_cov = self.ssModel.m2x_0

                if 'mu' in self.parameters:
                    next_init_mean = filtered_states[0]
                else:
                    next_init_mean = self.ssModel.m1x_0

                self.ssModel.InitSequence(next_init_mean, next_init_cov)

                if loss != None :
                    itt_loss = 10*torch.log10(loss.mean()).item()
                    losses.append(itt_loss)


                # if wandb_true and Plot == 'Plot':
                #     wandb.log({'IterationLoss [dB]':itt_loss})


                print(self.ssModel.Q)
                print('---------------------------------')

            # if loss != None:
            #     return filtered_states,losses
            # else:
            #     losses = loss

            return filtered_states, losses


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