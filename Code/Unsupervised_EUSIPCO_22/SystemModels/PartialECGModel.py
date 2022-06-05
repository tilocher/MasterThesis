# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import matplotlib.pyplot as plt
import scipy.fft
from scipy.special import factorial
import torch
from numpy import pi
import numpy as np
from tqdm import trange

pi = torch.tensor(pi)

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
    options = ['P','Q','R', 'S','T']
    labels = np.array(x.shape[0] * [5*['None']])
    for i, ai in enumerate(a):
        # if i != 4:
        #     continue
        dtheta = (theta - stheta[i])
        std = dtheta + 2*b[i]
        value = ai * dtheta * torch.exp(-dtheta ** 2 / (2 * b[i] ** 2))
        mask = torch.abs(value) > 1e-1
        labels[mask, i]  = options[i]
        dz += value

    dz -= (z - A_sin * torch.sin(2 * pi * f2 * t))



    B[:,2] = dz

    return A,B, labels

class PartialECGModel():

    def __init__(self, batches):

        self.batches = batches

        # Parameters for Matrix exponential
        self.deltaT = 1e-3
        self.order = 3

        # Parameters: P,Q,R,S,T
        self.theta = torch.tensor(batches*[[-pi/3, -pi/12, 0, pi/12, pi/2]]).T
        # self.theta = torch.tensor([-1, -pi/12, 0, pi/12, pi/2])

        self.a = torch.tensor(batches*[[1.2, -5.,30. ,-7.5, 0.75]]).T
        self.b = torch.tensor(batches*[[0.25, 0.1, 0.1, 0.1, 0.4]]).T
        self.w = torch.tensor(batches*[[10]])


        q_2 = 1
        self.Q = 10**(q_2/20)


        self.f2 = 5
        self.A_sin = 0.15

    def InitSequence(self, m1_x0):

        self.m1 = m1_x0
        self.state = self.m1

        self.label = np.array(m1_x0.shape[0] * [5*['None']])

        self.time = 0

    def step(self):
        A,B,labels = RateOfChange(self.state, self.theta,self.a,self.b,self.A_sin, self.f2, self.time)
        self.label = labels
        F = torch.empty(self.state.shape[0],4,4)
        F[:] = torch.eye(4)
        for j in range(1,self.order):
            F += torch.matrix_power(A * self.deltaT,j) / factorial(j)

        self.state  =  torch.einsum('bij,bj->bi',(F,self.state)) + self.deltaT * B
        # state_noise = torch.normal(mean=torch.zeros_like(self.state), )

        state_noise = torch.normal(torch.tensor([0.]), self.Q * self.deltaT)
        self.state[:,2] += state_noise
        # checksum = F[0] @ self.state[0] + self.deltaT * B[0]
        self.time +=1



    def GenerateSequence(self,num_steps = 100, batches = 1, init_vec = None):

        self.batches = batches

        rand_angle = 2* pi* torch.rand(batches)
        x_state = torch.cos(rand_angle).unsqueeze(-1)
        y_state = torch.sin(rand_angle).unsqueeze(-1)
        # x_state = torch.ones(batches,1)
        # y_state = torch.zeros(batches,1)
        if not init_vec == None:
            z_state = torch.atleast_2d(init_vec)
        else:
            z_state = torch.zeros(batches,1)
        w_state = self.w

        self.InitSequence(torch.cat((x_state,y_state,z_state,w_state), dim = -1))

        traj = torch.empty((batches,4,num_steps))
        obs = torch.empty((batches, 4, num_steps))
        labels = np.empty((batches,5,num_steps), dtype=str)
        r_2 = torch.tensor([1.])
        R = 10 ** (r_2/20)
        for i in range(num_steps):
            traj[:,:,i] = self.state.squeeze()
            obs[:,:,i] = (self.state + torch.normal(torch.zeros_like(self.state),R*self.deltaT)).squeeze()
            labels[:,:,i] = self.label.squeeze()
            self.step()

        return traj,obs,labels

    def plot(self):


        bs = self.batches
        leng = 1000


        traj,labels = self.GenerateSequence(leng,bs)
        loss = loss_fn

        # LOSS  = loss(traj[0,2],traj[1,2])


        P = labels[:,0,:] == 'P'
        Q = labels[:,1,:] == 'Q'
        R = labels[:,2,:] == 'R'
        S = labels[:,3,:] == 'S'
        T = labels[:,4,:] == 'T'
        No = labels[:,4,:] == 'N'

        t = np.array(bs*[np.arange(start=0 , stop = leng * self.deltaT, step = self.deltaT)])

        traj_np = traj.detach().numpy()
        plt.plot(t[0, No[0]], traj_np[0, 2, :][No[0]], '.', color='k')
        plt.plot(t[0,P[0]],traj_np[0,2,:][P[0]],'.',color = 'b')
        plt.plot(t[0, R[0]], traj_np[0, 2, :][R[0]], '.', color='g')
        plt.plot(t[0, Q[0]], traj_np[0, 2, :][Q[0]], '.', color='r')

        plt.plot(t[0, S[0]], traj_np[0, 2, :][S[0]], '.', color='y')
        plt.plot(t[0, T[0]], traj_np[0, 2, :][T[0]], '.', color='c')

        plt.show()

        # fft1 = torch.tensor(scipy.fft.fft(traj[0,2].detach().numpy()))
        # fft2 = torch.tensor(scipy.fft.fft(traj[1,2].detach().numpy()))

        # fft_loss = loss(fft1,fft2)
        # print(LOSS)
        # print(fft_loss)

        # plt.plot(traj[0,2])
        # plt.show()

    def SetW(self,w):
        self.w = torch.tensor(self.batches*[[w]])

    def generateTrainingData(self,train,cv,test,traj_len):


        train_input = np.empty((train,4,traj_len))
        train_label = np.empty((train, 4,5, traj_len),dtype=str)

        for i in trange(train,desc='Train'):


            bpm = 60
            f = bpm / 60
            deltaf = 5 / 60
            w = 2*pi*f
            delta_w = 2*pi*deltaf

            w = torch.normal(w, delta_w)

            self.SetW(w)

            traj,obs,label = self.GenerateSequence(traj_len,4)

            train_input[i] = obs[:,2].squeeze()
            train_label[i] = label.squeeze()

        np.save(f'..//Datasets//Syntetic//train_input_{train}.npy',train_input)
        np.save(f'..//Datasets//Syntetic//train_label_{train}.npy', train_label)

        ##################################################################################
        ##################################################################################
        ##################################################################################

        test_input = np.empty((test, 4, traj_len))
        test_label = np.empty((test, 4, 5, traj_len), dtype=str)

        for i in trange(test, desc='Test'):
            bpm = 60
            f = bpm / 60
            deltaf = 5 / 60
            w = 2 * pi * f
            delta_w = 2 * pi * deltaf

            w = torch.normal(w, delta_w)

            self.SetW(w)

            traj, obs, label = self.GenerateSequence(traj_len, 4)

            test_input[i] = obs[:, 2].squeeze()
            test_label[i] = label.squeeze()

        np.save(f'..//Datasets//Syntetic//test_input_{test}.npy', test_input)
        np.save(f'..//Datasets//Syntetic//test_label_{test}.npy', test_label)

        ##################################################################################
        ##################################################################################
        ##################################################################################

        cv_input = np.empty((cv, 4, traj_len))
        cv_label = np.empty((cv, 4, 5, traj_len), dtype=str)

        for i in trange(cv, desc='cv'):
            bpm = 60
            f = bpm / 60
            deltaf = 5 / 60
            w = 2 * pi * f
            delta_w = 2 * pi * deltaf

            w = torch.normal(w, delta_w)

            self.SetW(w)

            traj, obs, label = self.GenerateSequence(traj_len, 4)

            cv_input[i] = obs[:, 2].squeeze()
            cv_label[i] = label.squeeze()

        np.save(f'..//Datasets//Syntetic//cv_input_{cv}.npy', cv_input)
        np.save(f'..//Datasets//Syntetic//cv_label_{cv}.npy', cv_label)




def loss_fn(one,two):
    l = torch.mean(torch.pow(torch.abs(one-two),2))
    return l

if __name__ == '__main__':

    a = PartialECGModel(4)
    # a.plot()
    a.generateTrainingData(1000,100,1000,10000)