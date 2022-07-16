# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import datetime
import os

import matplotlib.pyplot as plt
import numpy
import scipy.special
import torch
import numpy as np
import wandb
from tqdm import trange
from scipy.optimize import minimize
from SystemModels.Extended_sysmdl import SystemModel


class Taylor_model():

    def __init__(self, taylor_order = 1, deltaT = 1,channels= 2, **kwargs):

        assert taylor_order >= 1, 'Taylor order must be at least 1'
        self.taylor_order = taylor_order
        self.deltaT = deltaT

        self.channels = channels

        self.basis_functions = torch.from_numpy(np.array([[deltaT**k/scipy.special.factorial(k)] for k in range(1,taylor_order + 1)])).float()
        self.basis_functions = self.basis_functions.reshape(( 1, -1)).repeat(( channels, 1))

        # Get window parameters
        if 'window' in kwargs.keys():
            self.window = kwargs['window']

            if 'window_parameters' in kwargs.keys():
                self.window_parameters = kwargs['window_parameters']

            else:
                self.window_parameters = None

            if 'window_size' in kwargs.keys():
                self.window_size = kwargs['window_size']

            else:
                self.window_size = 5
        elif 'window_size' in kwargs.keys():

            self.window,self.window_parameters = 'rectangular', 1.

            self.window_size = kwargs['window_size']

        else:
            self.window = ''
            self.window_size = 1
            self.window_parameters = 1

        self.CreateWindow()

    def CreateWindow(self):

        if self.window == '':
            return
        else:

            if self.window == 'rectangular':

                if self.window_parameters == None:
                    self.window_parameters = 1.

                self.window_weights = torch.from_numpy(np.array([self.window_parameters for w in range(self.window_size)]))



            elif self.window == 'exponential':

                if self.window_parameters == None:
                    self.window_parameters = 0.9

                self.window_weights = torch.from_numpy(
                    np.array([self.window_parameters**(np.abs(w - int(self.window_size/2))) for w in range(self.window_size)]))

            elif self.window == 'gaussian':

                if self.window_parameters == None:
                    self.window_parameters = 1

                self.window_weights = torch.from_numpy(
                    np.array([ self.window_parameters * np.exp(- (w - int(self.window_size/2))**2 /(2*self.window_parameters))
                              for w in range(self.window_size)]))

            elif self.window == 'linear':

                if self.window_parameters == None:
                    self.window_parameters = 1.

                slope = 1 / self.window_size

                self.window_weights = torch.from_numpy(
                    np.array([ -slope * np.abs(w - int(self.window_size/2)) + self.window_parameters
                              for w in range(self.window_size)]))

            else:
                raise ValueError('Window not supported')

            self.window_weights = torch.diag(self.window_weights).float()

            self.window_sum = torch.sum(self.window_weights)
            self.n_prediction_weight = torch.arange(-int(self.window_size/2)+1,int(self.window_size/2)+2).reshape(-1,1).float()




    def fit(self,data):

        self.T = data.shape[-1]

        if self.window == '':
            return self._FitWithoutWindow(data)

        else:
            return self._FitWithWindow(data)

    def _FitWithWindow(self,data):


        # time_steps = self.time_steps =  data.shape[-1]

        batch_size, channels, time_steps = data.shape

        self.time_steps = time_steps

        # basis_functions = self.n_prediction_weight.reshape(-1,1).float() @ self.basis_functions.T
        # basis_functions = basis_functions.repeat((batch_size,1,1))
        # basis_functions = self.basis_functions.reshape((1,1,-1)).repeat((batch_size,self.window_size,1))
        basis_functions = self.basis_functions.reshape((1, self.channels, -1)).repeat((batch_size, 1, 1))

        # weights = self.window_weights.repeat((batch_size,1,1))

        weights = torch.diag(self.window_weights)

        # basis_functions_w = basis_functions.mT.bmm(torch.sqrt(weights))



        coefficients = torch.zeros((self.taylor_order, time_steps))
        half_window = int(self.window_size/2)

        lower_bound = - half_window
        upper_bound = half_window + 1

        padded_data = torch.nn.functional.pad(data, (-lower_bound, upper_bound), 'replicate')

        def FuncToMin(params, obs, weights, basis):


            estimate = np.matmul(basis,params)


            loss = 0

            for j in range(len(weights)):

                vector = (obs[...,j] - estimate).unsqueeze(-1)
                loss += weights[j]   * torch.bmm(vector.mT, vector)

            return loss.mean().item()

        for t in trange(0, time_steps, desc = 'Calculating prior'):

            current_state = padded_data[:,:,t -lower_bound-half_window :t+half_window - lower_bound + 1]#.reshape((-1,1))
            observations = padded_data[:,:,t -lower_bound-half_window+1 :t+half_window - lower_bound + 2]

            target_tensor = (observations - current_state).reshape((batch_size,channels,-1))


            phi = minimize(FuncToMin,torch.zeros(self.taylor_order),args= (target_tensor,weights,basis_functions)).x


            # target_tensor_W = target_tensor.mT.bmm(torch.sqrt(weights))

            # c = torch.linalg.lstsq(basis_functions_w.mT,target_tensor_W.mT).solution

            coefficients[:,t] = torch.from_numpy(phi)



        self.coefficients = coefficients

        return coefficients



    def _FitWithoutWindow(self, data: torch.tensor):


        time_steps = self.time_steps =  data.shape[-1]

        data = data.reshape((-1,time_steps))

        batch_size = data.shape[0]

        basis_functions = self.basis_functions.repeat((1,batch_size)).T


        coefficients = torch.zeros((self.taylor_order, time_steps))


        for t in range(1,time_steps):
            target_tensor = (data[:,t] - data[:,t-1]).reshape((batch_size,-1))


            c = torch.linalg.lstsq(basis_functions,target_tensor).solution
            coefficients[:,t-1] = c.T

        self.coefficients = coefficients

        return coefficients

    def f(self,x,t):
        #TODO: change back
        # return (x.squeeze()+ self.basis_functions @ self.coefficients[:, t]).reshape(-1,1)
        return (x.squeeze()).reshape(-1, 1)

    @property
    def gradients(self):
        basis_functions = self.basis_functions.reshape((1,-1,1))
        basis_functions = basis_functions.repeat((self.T,1,1))


        return torch.bmm(torch.transpose(self.coefficients.unsqueeze(2),0,1).mT, basis_functions)


    def Jacobian(self,x,t):
        # return torch.atleast_2d(self.coefficients[0,t])


        # return (1 + self.coefficients[:,t] @ self.basis_functions).reshape((1,1))
        return torch.eye(self.channels)
    #
    def GetSysModel(self,channels):


        self.ssModel = SystemModel(self.f, 0, lambda x,t: x, 0,self.time_steps, self.time_steps,channels,channels)
        self.ssModel.setFJac(self.Jacobian)
        self.ssModel.setHJac(lambda x,t: torch.eye(self.channels))



        return self.ssModel


if __name__ == '__main__':

    wandb.login()
    wandb.init(project='MasterThesis',
               name= datetime.datetime.today().strftime('%d_%m___%H_%M'),
               group= 'TaylorModelEM')

    print(os.getcwd())
    from Code.ECG_Modeling.Filters.EM import EM_algorithm


    from Code.ECG_Modeling.DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH

    snr = 6

    ts = 360

    loader = PhyioNetLoader_MIT_NIH(1,1, 1*ts,SNR_dB=snr,random_sample=False)

    num_batches = 1000

    obs,state = loader.GetData(num_batches)



    taylor_model = Taylor_model(taylor_order= 5,  window= 'gaussian', window_size= 5 ,window_parameters = 5)

    shift_obs = torch.empty((num_batches,obs.shape[-1]))

    for t in range(num_batches):
        shift_obs[t] = torch.roll(obs[t,0],0)

    # taylor_model.fit(obs[:num_batches,0,:])
    taylor_model.fit(shift_obs[:-5])

    shift = 0
    shift_obs = torch.roll(obs[-2, 0], shift)
    shift_state = torch.roll(state[-2, 0], shift)

    #
    ssModel = taylor_model.GetSysModel()
    ssModel.InitSequence(shift_obs[0], torch.eye(1))
    ssModel.GenerateSequence(ssModel.Q,ssModel.R, ssModel.T)

    # ###############################################################

    EM = EM_algorithm(ssModel,Plot_title= 'SNR: {} [dB]'.format(snr), units= 'mV', parameters=['R','Q', 'mu','Sigma'])


    r_2 = np.random.random()
    q_2 = np.random.random()

    filtered_states, loss = EM.EM(shift_obs,shift_state,num_itts=100, r_2=r_2, q_2= q_2,Plot='_SNR_{}_Taylor'.format(snr))


    last_loss = round(loss[-1],2)
    #
    lower_window = 0
    upper_window = -1

    upper_window = state.shape[-1] + upper_window if upper_window < 0 else upper_window

    plt.plot(filtered_states.squeeze()[lower_window:upper_window], label= 'Predicted signal {} window'.format(taylor_model.window) , color = 'b')
    # plt.plot(ssModel1.x.squeeze()[lower_window:upper_window], label= 'Predicted signal rectangular', color = 'r')
    # plt.plot(ssModel2.x.squeeze()[lower_window:upper_window], label= 'Predicted signal exponential', color = 'g')
    plt.plot(state[-3,0,lower_window:upper_window].squeeze(), label = 'Ground truth',color= 'g')
    plt.plot(obs[-3,0,lower_window:upper_window].squeeze(), label = 'Observation, SNR: {}[dB]'.format(snr), color = 'r',alpha=0.3)
    plt.ylabel('Amplitude [mV]')
    plt.xlabel('Time-steps')
    plt.title('Zoomed from time-step: {}-{}'.format(lower_window, upper_window))
    plt.legend()
    wandb.log({'chart':plt})
    plt.savefig('..\\Plots\\Taylor_models\\sample_prior_batches_{}_windowType_{}_window_size_{}_window_param_{}_loss_{}.pdf'
                .format(num_batches,taylor_model.window,taylor_model.window_size , taylor_model.window_parameters,last_loss))
    plt.show()


