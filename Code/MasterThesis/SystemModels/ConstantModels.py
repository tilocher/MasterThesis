# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.special import factorial
from SystemModels.Extended_sysmdl import SystemModel


class ConstantModel():

    def __init__(self,state_order,observed_states,q_2,r_2,T = 472, deltaT = 1.):

        # Get the order of the systems
        self.state_order = self.m = state_order
        self.observed_states = self.n = observed_states
        self.observation_order = len(self.observed_states) if isinstance(observed_states, tuple) else 1

        # Set noise statistics
        self.q_2 = q_2
        self.r_2 = r_2

        # Set Time
        self.T = T

        # Calculate state evolution matrix
        self.F = torch.eye(state_order)
        for i in range(1,state_order):
            self.F += torch.diag_embed(torch.ones(state_order - i), offset=i) * (deltaT)**(i)/(factorial(i))

        # Calculate the Observation matrix
        if isinstance(self.observed_states,int):
            self.H = torch.zeros((1,self.state_order))
            self.H[0,observed_states] = 1
        else:
            self.H = torch.zeros((len(self.observed_states), self.state_order))
            for i, state in enumerate(self.observed_states):
                self.H[i,state] = 1


        # Get State Space Noise
        self.base_Q = torch.zeros_like(self.F)


        for i in range(0,state_order):
            for j in range(0,state_order):
                i_prim = (state_order - i)
                j_prim = (state_order - j)
                exponent = (i_prim -1 + j_prim)
                self.base_Q[i,j] = deltaT**exponent / (exponent * (factorial(j_prim-1) * factorial(i_prim-1)))

        self.Q = q_2 * self.base_Q
        self.R = r_2 * torch.eye(self.observation_order)



        self.ssModel = SystemModel(self.f,np.sqrt(q_2),self.h, np.sqrt(r_2), T,T,state_order,self.observation_order)
        self.ssModel.setFJac(self.df)
        self.ssModel.setHJac(self.dh)
        self.ssModel.UpdateCovariance_Matrix(self.Q,self.R)

    def f(self,x, t):
        return (self.F @ x)

    def h(self,x, t):
        return self.H @ x

    def df(self,x,t):
        return self.F

    def dh(self,x,t):
        return self.H
    def GetSysModel(self,T,timesteps = None):
        return self.ssModel

    def fit(self,x):
        return self.GetSysModel(None)


    def UpdateGain(self,q_2,r_2):

        self.q_2 = q_2
        self.r_2 = r_2
        self.Q = q_2 * self.base_Q
        self.R = r_2 * torch.eye(self.observation_order)
        self.ssModel.UpdateCovariance_Matrix(self.Q , self.R)


