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


def RateOfChange(state ,stheta,a ,b,A_sin,f2,t):

    x = state[:,0]
    y = state[:,1]
    z = state[:,2]
    w = state[:,3]

    A = torch.zeros((x.shape[0],4,4))


    alpha = 1 - torch.sqrt(x ** 2 + y ** 2)
    theta = torch.atan2(x, y)

    A[:,0,0] = alpha
    A[:,0,1] = -w
    A[:,1,0] = w
    A[:,1,1] = alpha

    A[:,2,2] = -1

    B = torch.zeros((x.shape[0],4))
    dz = 0
    for i, ai in enumerate(a):
        dtheta = (theta - stheta[i])
        std = dtheta + 2*b[i]
        dz += ai * dtheta * torch.exp(-dtheta ** 2 / (2 * b[i] ** 2))

    dz -= (z - A_sin * torch.sin(2 * pi * f2 * t))
    B[:,2] = dz

    return A,B



class ECG_signal():

    def __init__(self, batches):

        # Parameters for Matrix exponential
        self.deltaT = 1e-3
        self.order = 3

        # Parameters: P,Q,R,S,T
        self.theta = torch.tensor(batches*[[-pi/3, -pi/12, 0, pi/12, pi/2]]).T
        # self.theta = torch.tensor([-1, -pi/12, 0, pi/12, pi/2])

        self.a = torch.tensor(batches*[[1.2, -5.,30. ,-7.5, 0.75]]).T
        # self.a = torch.tensor([1.2])
        self.b = torch.tensor(batches*[[0.25, 0.1, 0.1, 0.1, 0.4]]).T
        self.w = torch.tensor(batches*[[10]])



        self.f2 = 0
        self.A_sin = 0.15

    def ChangeParameters(self,a,b, theta,w):

        self.a =  a
        self.b = b
        self.w = w.unsqueeze(-1)
        self.theta = theta

    def InitSequence(self,m1_x0):

        self.m1 = m1_x0
        self.state = self.m1

        self.time = 0

    def step(self):
        A,B = RateOfChange(self.state, self.theta,self.a,self.b,self.A_sin, self.f2, self.time)
        F = torch.empty(self.state.shape[0],4,4)
        F[:] = torch.eye(4)
        for j in range(1,self.order):
            F += torch.matrix_power(A * self.deltaT,j) / scipy.special.factorial(j)

        self.state  =  torch.einsum('bij,bj->bi',(F,self.state)) + self.deltaT * B
        # checksum = F[0] @ self.state[0] + self.deltaT * B[0]
        self.time +=1
        1


    def GenerateSequence(self,num_steps = 100, batches = 1, init_vec = None):

        x_state = torch.ones(batches,1)
        y_state = torch.zeros(batches,1)
        if not init_vec == None:
            z_state = torch.atleast_2d(init_vec)
        else:
            z_state = torch.zeros(batches,1)
        w_state = self.w

        self.InitSequence(torch.cat((x_state,y_state,z_state,w_state), dim = -1))

        traj = torch.empty((batches,4,num_steps))

        for i in range(num_steps):
            traj[:,:,i] = self.state.squeeze()
            self.step()

        return traj


    def GenerateBatch(self, Batch_start_vector, Seq_len, parameters):

        #TODO: Change amplitudes for specific waves

        amplitude_scaling  = torch.tensor([[1.2, -5.,30. ,-7.5, 0.75]]).T
        amplitude_scaling = amplitude_scaling/torch.norm(amplitude_scaling)

        std_scaling = torch.tensor( [[0.25, 0.1, 0.1, 0.1, 0.4]]).T
        std_scaling = std_scaling / torch.norm(std_scaling)

        # a = torch.log(torch.exp(torch.tensor(0.1)) + torch.exp(parameters[:,:5].T))

        a = torch.sigmoid(parameters[:,:5].T) * amplitude_scaling*1e-3
        # a = torch.sigmoid(parameters[:,:5].T) * 5e-4 - 1e-3

        # b = torch.log(torch.exp(torch.tensor(1e-2)) + torch.exp(parameters[:,5:10].T))
        b = torch.sigmoid(parameters[:,5:10].T) * std_scaling
        w = torch.sigmoid(parameters[:, 15].T)*2 + 14

        A_sin  = torch.sigmoid(parameters[:, 16].T) *5e-5 + 1e-6
        f2 = torch.sigmoid(parameters[:, 17].T) * 2

        self.A_sin = A_sin
        self.f2 = f2

        # w = torch.log(torch.exp(torch.tensor(3)) + torch.exp(parameters[:,-1].T))
        # theta = torch.log(torch.exp(torch.tensor()) + torch.exp(parameters[:, -1].T))

        self.ChangeParameters(a,b
                              ,parameters[:,10:15].T, w)



        batch_size = len(Batch_start_vector)


        traj = self.GenerateSequence(Seq_len, batch_size, Batch_start_vector)



        return traj

if __name__ == '__main__':

    a = ECG_signal(1)
    traj = a.GenerateSequence(1000,1)

    1