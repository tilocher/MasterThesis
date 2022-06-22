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

    def __init__(self,ssModel: SystemModel, parameters: tuple = ('R')):

        self.ssModel = ssModel
        self.m = ssModel.m
        self.n = ssModel.n


        self.parameters = parameters


    def FilterEstimatedModel(self, observations: torch.Tensor, states: torch.Tensor, Plot=''):



        KF = ExtendedKalmanFilter(self.ssModel)
        KF.InitSequence(self.ssModel.m1x_0, self.ssModel.m2x_0)
        RTS = Extended_rts_smoother(self.ssModel)

        MSE_RTS_linear_arr = torch.empty(self.batch_size) if states != None else None
        loss_rts = torch.nn.MSELoss(reduction='mean')

        filtered_states = torch.empty((self.batch_size, self.channels, self.ssModel.m, self.T))
        error_cov = torch.empty((self.batch_size, self.ssModel.m, self.ssModel.m, self.T))
        error_cov_prior = torch.empty((self.batch_size, self.ssModel.m, self.ssModel.m, self.T))
        # Run Loop

        with torch.no_grad():
            for j in trange(self.batch_size):
                KF.GenerateSequence(observations[j].reshape((self.channels,self.m,-1)), KF.T_test)
                RTS.GenerateSequence(KF.x, KF.sigma, RTS.T_test)

                if states != None:

                    MSE_RTS_linear_arr[j] = loss_rts(RTS.s_x, states[j].squeeze()).item()
                    print('Mean RTS loss Estimated Model: {} [dB]'.format(
                        10 * torch.log10(MSE_RTS_linear_arr.mean()).item()))

                filtered_states[j] = RTS.s_x
                error_cov[j] = RTS.s_sigma
                error_cov_prior[j] = RTS.s_sigma_prior

        self.EstRTS = RTS

        if Plot == 'Plot' or Plot == 'Save':

            rand_sample = np.random.randint(0,self.batch_size)
            if states != None:
                plt.plot(states[rand_sample].squeeze(), label='Noiseless data', alpha=0.8, color='g')
            plt.plot(observations[rand_sample].squeeze(), label='noisy data', alpha=0.3, color='r')

            plt.plot(filtered_states[rand_sample].squeeze(), label='Filtered data, evolution unknown', color='b')

            plt.legend()
            if Plot == 'Save':
                plt.savefig('..\\Plots\\EM_filtered_Sample_{}.pdf'.format(str(datetime.datetime.now().date())))
            plt.show()

        return filtered_states, error_cov, error_cov_prior, MSE_RTS_linear_arr


    def EM(self,observation: torch.Tensor, state: torch.Tensor = None, q_2 = 1, r_2 = 1 , Plot = 'Plot', num_itts = 10):

        Q = q_2 * torch.eye(self.m)
        R = r_2 * torch.eye(self.n)

        losses = []

        with torch.no_grad():

            batch_size = self.batch_size  = observation.shape[0]
            T = self.T = observation.shape[-1]
            channels = self.channels = 1 if len(observation.shape) <= 2 else observation.shape[1]

            self.ssModel.T = self.ssModel.T_test = T

            observation = observation.reshape((batch_size,channels,self.n,T))

            if state != None:
                state = state.reshape((batch_size,channels, self.m, T))

            for i in range(num_itts):

                # Create the system model
                self.ssModel.UpdateCovariance_Matrix(Q,R)

                filtered_states,error_cov,error_cov_prior,loss = self.FilterEstimatedModel(observation,state , Plot= Plot)

                error_cov = error_cov.mean(0).reshape((self.m,self.m,-1))
                error_cov_prior = error_cov_prior.mean(0).reshape((self.m,self.m,-1))
                E_xx = torch.einsum('bqmt,tnqb->mnt',(filtered_states,filtered_states.T))/batch_size


                U_xx = (E_xx + error_cov).mean(-1)
                U_yx = (torch.einsum('bqnt,tmqb->nmt',(observation,filtered_states.T))/batch_size).mean(-1)
                U_yy = (torch.einsum('bqnt,tqmb->nmt',(observation,observation.T))/batch_size).mean(-1)

                C = U_yx @ torch.inverse(U_xx)

                if 'C' in self.parameters:
                    self.ssModel.setHJac(C)

                if 'R' in self.parameters:
                    R = U_yy - C @ U_yx.T

                V_xx = (E_xx[:,:,:-1] + error_cov[:,:,:-1])
                V_x1x1 = (E_xx[:,:,1:] + error_cov[:,:,1:])
                V_x1x = (((torch.einsum('bmqt,tqnb->mnt', (filtered_states[:,:,:,1:], filtered_states[:,:,:,:-1].T))/batch_size)
                          + error_cov_prior[:,:,:-1]).reshape((self.m,self.m,-1)))

                A = torch.bmm(V_x1x.T, torch.inverse(V_xx.T)).T

                if 'A' in self.parameters:
                    self.ssModel.setFJac(lambda x,t: A[:,:,t-1] if t>0 else torch.eye(self.n) )

                if 'Q' in self.parameters:
                    Q = (V_x1x1 - torch.bmm(A.T,V_x1x.T).T).mean()



                print('q: {}'.format(Q.item()))
                print('r^2: {}'.format(R.item()))

                if loss != None:
                    losses.append(10*torch.log10(loss.mean()).item())

            if loss != None:

                plt.plot(losses,'*g', label = 'loss per iteration')
                plt.grid()
                plt.xlabel('Iteration')
                plt.ylabel('Loss [dB]')
                plt.title('EM optimization convergence')
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

    # from Code.ECG_Modeling.DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH
    #
    # loader = PhyioNetLoader_MIT_NIH(1,1,10)
    #
    # obs,state = loader.GetData()

    obs = torch.randn((3,2,360))
    state = torch.randn((3,2,360))


    EM = EM_algorithm(ToyModel,parameters = ('R','A'))
    EM.EM(obs,state, q_2= 5e-3**2)