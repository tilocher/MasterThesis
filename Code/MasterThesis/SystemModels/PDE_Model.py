import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import torch

from Extended_sysmdl import SystemModel

pi = np.pi

class PDE_Model(SystemModel):

    theta = torch.tensor([-1/3*pi , -1/12 * pi, 0, 1/12 * pi, 1/2 * pi])
    a = torch.tensor([1.2, -5.0, 30.0, -7.5, 0.75])
    b = torch.tensor([0.25, 0.1, 0.1, 0.1, 0.4])
    alpha = torch.tensor([0.2,-0.1,1.2,-0.05,0.1])*1e0

    order = 5




    def __init__(self,fs,channels):

        self.fs = fs




        self.w =  2 * pi

        self.deltaT = 1 /fs

        self.channels = channels

        super(PDE_Model, self).__init__(self.f, 0, self.h,0, T= int(fs), T_test= int(fs), m = channels + 1, n=channels)

        #
        # self.Q = torch.eye(3*len(self.a) + 2)
        #
        # for i in range(len(self.a)):
        #     self.Q[i,i] = (0.1*self.alpha[i])**2
        # for i in range(len(self.a), len(self.a) + len(self.b)):
        #     self.Q[i,i] = (0.05*pi)**2
        #
        # for i in range(len(self.a) + len(self.b), len(self.a) + len(self.b) + len(self.theta)):
        #     self.Q[i, i] = (0.05 * pi) ** 2
        #
        #
        # self.Q[-1,-1] = (0.15)**2
        # self.Q[-2,-2] = (0.1*1.2e-3)**2



    def f(self,state_xy,t):




        # noise = torch.distributions.MultivariateNormal(loc= torch.zeros(3*len(self.a) +2) , covariance_matrix= self.Q).sample()


        b = self.b #+ noise[len(self.a):len(self.a)+len(self.b)]*0
        theta = self.theta #+ noise[len(self.a) + len(self.b):-2]*0
        w = self.w #+ noise[-2]
        # nu = noise[-1]*0.001


        theta_t = state_xy[0]
        z_t = state_xy[1:]

        dtheta = (theta_t- theta) #% (2 * pi)

        theta_t_1 = (theta_t + w * self.deltaT)#%(2*pi)

        z_t_1 = - self.deltaT * torch.sum( self.alpha * self.w/ self.b* dtheta * torch.exp(-dtheta**2/(2*b**2))) + z_t #+ nu

        return torch.cat((theta_t_1.unsqueeze(-1),z_t_1))

        # return state


    def getFJacobian(self,x,t):

        theta = x[0]
        z = x[1:]

        dtheta = (self.theta - theta) % (2 * pi)

        A = torch.zeros((self.channels+1, self.channels+1))
        A[0, 1] = A[1, 0] = 1
        update =  -torch.sum(self.deltaT * self.alpha * self.w/ self.b * (1 - (dtheta ** 2 / self.b ** 2) *
                                                     torch.exp(-(dtheta ** 2 / (2 * self.b ** 2)))))

        for i in range(1,self.channels+1):
            A[i,i] = update
        # NumNoiseVars = 3 * len(self.a) + 2
        # F = torch.zeros((2, NumNoiseVars))
        #
        # F[0, -2] = self.deltaT  # dF1/dw
        #
        # F[1, :len(self.a)] = -self.deltaT * self.w * dtheta / self.b ** 2 * torch.exp(
        #     -dtheta ** 2 / (2 * self.b))  # dF2/da_i
        # F[1, len(self.a):len(self.b) + len(self.a)] = 2 * self.deltaT * self.a / self.b * (
        #             1 - (dtheta ** 2 / self.b ** 2)) * \
        #                                               torch.exp(-dtheta ** 2 / (2 * self.b ** 2))  # dF2/db_i
        # F[1, len(self.b) + len(self.a):-2] = self.deltaT * self.a * (1 - dtheta ** 2 / self.b ** 2) * \
        #                                      torch.exp(-dtheta ** 2 / (2 * self.b ** 2))
        #
        # F[1, -1] = 1
        # F[1, -2] = - torch.sum(self.deltaT * self.a * torch.exp(-dtheta ** 2 / (2 * self.b ** 2)))

        return A

    def getHJacobian(self,x,t):

        H = torch.zeros((self.channels, self.channels +1))
        for i in range(1,self.channels):
            H[i,i] = 1

        return H




    def h(self,x,t):
        return x[1:]

    def UpdateQ(self,Q):
        self.Q = Q
        self.Q[0,0] = 0




# if __name__ == '__main__':
#
#     model = PDE_Model(360)
#
#     ini = torch.zeros(2,1)
#     ini[0] = -pi
#
#
#
#
#     model.InitSequence(ini, torch.eye(2))
#
#     model.GenerateSequence(360)
#
#     # plt.plot(torch.sin(model.x[0]))
#     plt.plot(model.x[1])
#     plt.show()
#     1



