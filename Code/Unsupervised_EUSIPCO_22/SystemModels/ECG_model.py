# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import numpy as np
import torch
import scipy.special
from scipy.integrate import odeint
from matplotlib import pyplot as plt

# System model based on https://ieeexplore.ieee.org/abstract/document/1186732

pi = np.pi


def RateOfChange(state, t ,stheta,a ,b, f2):

    x = state[0]
    y = state[1]
    z = state[2]
    w = state[3]

    A = torch.zeros((4,4))


    alpha = 1 - torch.sqrt(x ** 2 + y ** 2)
    theta = torch.atan2(x, y)

    A[0,0] = alpha
    A[0,1] = -w
    A[1,0] = w
    A[1,1] = alpha

    A[2,2] = -1

    B = torch.zeros((4,1))
    dz = 0
    for i, ai in enumerate(a):
        dtheta = (theta - stheta[i])
        dz += ai * dtheta * torch.exp(-dtheta ** 2 / (2 * b[i] ** 2))

    # dz -= (z - A * torch.sin(2 * pi * f2 * t))
    B[2] = dz

    return A,B



class ECG_signal():

    def __init__(self):

        # Parameters for Matrix exponential
        self.deltaT = 1e-3
        self.order = 3

        # Parameters: P,Q,R,S,T
        self.theta = torch.tensor([-pi/3, -pi/12, 0, pi/12, pi/2])
        # self.theta = torch.tensor([-1, -pi/12, 0, pi/12, pi/2])

        self.a = torch.tensor([1.2, -5.,30. ,-7.5, 0.75])
        # self.a = torch.tensor([1.2])
        self.b = torch.tensor([0.25, 0.1, 0.1, 0.1, 0.4])
        self.w = 10
        self.f2 = 0
        self.A = 0.15

    def ChangeParameters(self,a,w,b):

        self.a = a
        self.w = w
        self.b = b


    def InitSequence(self,m1_x0):

        self.m1 = m1_x0
        self.state = self.m1

        self.time = 0

    def step(self):
        A,B = RateOfChange(self.state,self.time, self.theta,self.a,self.b,self.f2)
        F = torch.eye(4)
        for j in range(1,self.order):
            F += torch.matrix_power(A * self.deltaT,j) / scipy.special.factorial(j)

        self.state = F @ self.state + self.deltaT * B



    def GenerateSequence(self,num_steps = 100):

        self.InitSequence(torch.tensor([[1.],[0],[0.],[self.w]]))

        traj = torch.empty((4,num_steps))

        for i in range(num_steps):
            traj[:,i] = self.state.squeeze()
            self.step()

        return traj
