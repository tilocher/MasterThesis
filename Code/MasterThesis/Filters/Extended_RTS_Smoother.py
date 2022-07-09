"""# **Class: Extended RTS Smoother**
Theoretical Non-linear Linear RTS Smoother
"""
import torch
# from filing_paths import path_model
import sys
# sys.path.insert(1, path_model)
# from model import getJacobian

class Extended_rts_smoother:

    def __init__(self, SystemModel, mode='full'):
        self.f = SystemModel.f
        self.m = SystemModel.m

        self.Q = SystemModel.Q

        self.h = SystemModel.h
        self.n = SystemModel.n

        self.R = SystemModel.R

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

        self.sysModel = SystemModel

        # Full knowledge about the model or partial? (Should be made more elegant)
        if(mode == 'full'):
            self.fString = 'ModAcc'
            self.hString = 'ObsAcc'
        elif(mode == 'partial'):
            self.fString = 'ModInacc'
            self.hString = 'ObsInacc'

        self.Q_array = SystemModel.Q_array

    # Compute the Smoothing Gain
    def SGain(self, filter_x, filter_sigma,t):
        # Predict the 1-st moment of x
        self.filter_x_prior = self.f(filter_x,t)
        # Compute the Jacobians
        self.UpdateJacobians(self.sysModel.getFJacobian(filter_x,t), self.sysModel.getHJacobian(self.filter_x_prior,t))
        self.SG = torch.matmul(torch.atleast_2d(filter_sigma), self.F_T)
        self.filter_sigma_prior = torch.matmul(self.F, filter_sigma)
        self.filter_sigma_prior = torch.matmul(self.filter_sigma_prior, self.F_T) + self.Q
        self.SG = torch.matmul(self.SG, torch.inverse(self.filter_sigma_prior))

        self.SG_array[t] = self.SG.squeeze()

    # Innovation for Smoother
    def S_Innovation(self, filter_x, filter_sigma):
        self.dx = self.s_m1x_nexttime - self.filter_x_prior
        self.dsigma = self.filter_sigma_prior - self.s_m2x_nexttime

    # Compute previous time step backwardly
    def S_Correct(self, filter_x, filter_sigma):
        # Compute the 1-st moment
        self.s_m1x_nexttime = filter_x + torch.matmul(self.SG, self.dx)

        # Compute the 2-nd moment
        self.s_m2x_nexttime = torch.matmul(self.dsigma, torch.transpose(self.SG, 0, 1))
        self.s_m2x_nexttime = filter_sigma - torch.matmul(self.SG, self.s_m2x_nexttime)

        # self.smoothed_prior = self.s_m2x_nexttime @ self.SG.T + self.SG @ (self.smoothed_prior - self.F @ self.filter_sigma_prior) @ self.SG.T

    def S_Update(self, filter_x, filter_sigma,t):
        self.SGain(filter_x, filter_sigma,t)
        self.S_Innovation(filter_x, filter_sigma)
        self.S_Correct(filter_x, filter_sigma)

        return self.s_m1x_nexttime,self.s_m2x_nexttime

        #########################

    def UpdateJacobians(self, F, H):
        if type(F) == tuple:
            F = torch.atleast_2d(F[0].squeeze())

        if type(H) == tuple:
            H = torch.atleast_2d(H[0].squeeze())
        self.F = F
        self.F_T = torch.transpose(F, 0, 1)
        self.H = H
        self.H_T = torch.transpose(H, 0, 1)
        #print(self.H,self.F,'\n')

    #########################
    ### Generate Sequence ###
    #########################

    def GenerateSequence(self, KF, T):

        filter_x = KF.x
        filter_sigma = KF.sigma

        # Pre allocate an array for predicted state and variance
        self.s_x = torch.empty(size=[self.m, T])
        self.s_sigma = torch.empty(size=[self.m, self.m, T])
        self.s_sigma_prior = torch.empty(size=[self.m, self.m, T])
        self.s_smooth_prior = torch.empty(size=[self.m, self.m, T])
        # Pre allocate SG array
        self.SG_array = torch.zeros((self.T,self.m,self.m))

        self.s_m1x_nexttime = filter_x[:, T-1].reshape((self.m,-1))
        self.s_m2x_nexttime = filter_sigma[:, :, T-1].reshape((self.m,self.m))
        self.s_x[:, T-1] = self.s_m1x_nexttime.squeeze()
        self.s_sigma[:, :, T-1] = self.s_m2x_nexttime.squeeze()


        for t in range(T-2,-1,-1):
            if self.Q_array != None:
                self.Q = self.Q_array[t]
            filter_xt = filter_x[:, t].reshape((self.m,-1))
            filter_sigmat = filter_sigma[:, :, t].reshape((self.m,-1))
            s_xt,s_sigmat = self.S_Update(filter_xt, filter_sigmat,t)
            self.s_x[:, t] = torch.squeeze(s_xt)
            self.s_sigma[:, :, t] = torch.squeeze(s_sigmat)
            self.s_sigma_prior[:,:,t] = self.filter_sigma_prior

        self.smoothed_prior = (torch.eye(self.m) - KF.KG_array[-1]@ KF.H) @ KF.F @ filter_sigma[:,:,-1]

        for t in range(T-1,-1,-1):
            # self.smoothed_prior = self.s_sigma[:,:,t] @ self.SG_array[t,:,:].T + self.SG_array[t,:,:] @ (self.smoothed_prior - KF.F @ self.s_sigma_prior[:,:,t]) @ self.SG_array[t,:,:].T
            # self.s_smooth_prior[:,:,t] = self.smoothed_prior.squeeze()
            self.s_smooth_prior[:,:,t-1] = self.s_sigma[:,:,t] @ self.SG_array[t-1,:,:].T

        # print('hakuna matata')