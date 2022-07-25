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
        # self.basis_functions = self.basis_functions.reshape(( 1, -1)).repeat(( channels, 1))

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

        data = data.detach().cpu()

        with torch.no_grad():

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
        basis_functions = self.basis_functions.reshape((1, 1, -1)).repeat((batch_size, 1, 1))

        # weights = self.window_weights.repeat((batch_size,1,1))

        weights = torch.diag(self.window_weights)

        # basis_functions_w = basis_functions.mT.bmm(torch.sqrt(weights))



        coefficients = torch.zeros((self.taylor_order,channels, time_steps))
        half_window = int(self.window_size/2)

        lower_bound = - half_window
        upper_bound = half_window + 1

        padded_data = torch.nn.functional.pad(data, (-lower_bound, upper_bound), 'replicate')

        def FuncToMin(params, obs, weights, basis):


            estimate = np.matmul(basis,params.reshape(basis.shape[-1],-1)).squeeze()


            loss = 0

            for j in range(len(weights)):

                vector = (obs[...,j] - estimate).unsqueeze(-1)
                loss += weights[j]   * torch.bmm(vector.mT, vector)

            return loss.mean().item()

        for t in trange(0, time_steps, desc = 'Calculating prior'):

            current_state = padded_data[:,:,t -lower_bound-half_window :t+half_window - lower_bound + 1]#.reshape((-1,1))
            observations = padded_data[:,:,t -lower_bound-half_window+1 :t+half_window - lower_bound + 2]

            target_tensor = (observations - current_state)


            phi = minimize(FuncToMin,np.zeros((self.taylor_order,channels)),args= (target_tensor,weights,basis_functions)).x


            # target_tensor_W = target_tensor.mT.bmm(torch.sqrt(weights))

            # c = torch.linalg.lstsq(basis_functions_w.mT,target_tensor_W.mT).solution

            coefficients[...,t] = torch.from_numpy(phi.reshape(-1,channels))



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
        return (x.squeeze() + (self.basis_functions.T @ self.coefficients[..., t]).squeeze()).unsqueeze(-1)
        # return (x.squeeze()).reshape(-1, 1)

    @property
    def gradients(self):
        basis_functions = self.basis_functions.reshape((1,-1,1))
        basis_functions = basis_functions.repeat((self.T,1,1))


        return torch.bmm(torch.transpose(self.coefficients.unsqueeze(2),0,1).mT, basis_functions)


    def FJacobian(self,x,t):
        return torch.eye(self.channels)

    def h(self,x, t):
        return x

    def HJacobian(self,x, t):
        return torch.eye(self.channels)

    def GetSysModel(self,channels,gpu = False):


        if gpu:
            dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.ssModel = SystemModel(self.f, 0, self.h, 0, self.time_steps, self.time_steps, channels,
                                       channels)

            self.basis_functions = self.basis_functions.to(dev)
            self.ssModel.setFJac(self.FJacobian)
            self.ssModel.setHJac(self.HJacobian)


        self.ssModel = SystemModel(self.f, 0, self.h, 0,self.time_steps, self.time_steps,channels,channels)
        self.ssModel.setFJac(self.FJacobian)
        self.ssModel.setHJac(self.HJacobian)



        return self.ssModel

