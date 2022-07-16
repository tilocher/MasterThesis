import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import torch
from tqdm import trange

from SystemModels.Extended_sysmdl import SystemModel


class KalmanFilter():
    
    def __init__(self, sysModel: SystemModel, em_vars = ['R'], em_averages = ['R']):
        self.ssModel = sysModel

        self.m = sysModel.m
        self.n = sysModel.n

        self.f = sysModel.f
        self.h = sysModel.h

        self.Q = sysModel.Q
        self.R = sysModel.R

        self.Q_arr = None
        self.R_arr = None

        self.em_vars = em_vars
        self.em_averages = list(set(em_vars).intersection(em_averages))

    def f_batch(self,state,t):
        return torch.stack([self.f(x,t) for x in state ])

    def h_batch(self,state,t):
        return torch.stack([self.h(x,t) for x in state ])

    def UpdateQ(self,Q):
        self.Q = Q

    def UpdateR(self,R):
        self.R = R

    def GetQ(self,t):

        if self.Q_arr == None:
            return self.Q
        else:
            if 'Q' in self.em_averages:
                return self.Q_arr
            else:
                return self.Q_arr[:,t]

    def GetR(self, t):

        if self.R_arr == None:
            return self.R
        else:
            if 'R' in self.em_averages:
                return self.R_arr
            else:
                return self.R_arr[:, t]


    def InitSequence(self,InitialMean = None, InitialCovariance = None):

        if InitialMean == None:
            self.Initial_State_Mean = self.ssModel.m1x_0
        else:
            self.Initial_State_Mean =  InitialMean

        if InitialCovariance == None:
            self.Initial_State_Covariance = self.ssModel.m2x_0
        else:
            self.Initial_State_Covariance = InitialCovariance

        if len(self.Initial_State_Mean.shape) == 2:
            self.Initial_State_Mean = self.Initial_State_Mean.unsqueeze(0)

        if len(self.Initial_State_Covariance.shape) == 2:
            self.Initial_State_Covariance = self.Initial_State_Covariance.unsqueeze(0)


    def UpdateJacobians(self,t):

        # Update Gradients
        self.F = torch.stack([self.ssModel.getFJacobian(x.squeeze(), t) for x in self.Filtered_State_Mean])
        self.H = torch.stack([self.ssModel.getHJacobian(x.squeeze(), t) for x in self.Filtered_State_Mean])


    def predict(self,t):

        # Predict the 1-st moment of x
        self.Predicted_State_Mean = self.f_batch(self.Filtered_State_Mean,t)

        # Predict the 2-nd moment of x
        self.Predicted_State_Covariance = torch.bmm(self.F,
                                                    torch.bmm(self.Filtered_State_Covariance,
                                                              self.F.mT)) + self.GetQ(t)


    def KGain(self,t):

        # Predict the 1-st moment of y
        self.Predicted_Observation_Mean = self.h_batch(self.Predicted_State_Mean, t)
        # Predict the 2-nd moment y
        self.Predicted_Observation_Covariance = torch.bmm(self.H,
                                                          torch.bmm(self.Predicted_State_Covariance,
                                                                    self.H.mT)) + self.GetR(t)

        # Compute Kalman Gain
        self.KG = torch.bmm(self.Predicted_State_Covariance, torch.bmm(
            self.H.mT, torch.linalg.pinv(self.Predicted_Observation_Covariance)))

    def Innovation(self,y):

        self.Observation = y

        # Compute Innovation
        self.Predicted_Residual = (self.Observation - self.Predicted_Observation_Mean)

    def Correct(self):

        # Compute the 1-st posterior moment
        self.Filtered_State_Mean = self.Predicted_State_Mean + torch.bmm(self.KG,self.Predicted_Residual)

        # Compute the 2-nd posterior moments
        self.Filtered_State_Covariance = self.Predicted_State_Covariance - \
                                          torch.bmm(self.KG, torch.bmm(
                                              self.H , self.Predicted_State_Covariance
                                          ))
        self.Filtered_Residual = self.Observation - torch.bmm(self.H,self.Filtered_State_Mean)


    def filter(self, observations, T):


        with torch.no_grad():

            # # Get a batch dimension if there is none
            # observations = self.Observations = torch.atleast_3d(observations).unsqueeze(-1)

            if len(observations.shape) == 2:
                observations = observations.unsqueeze(0)
            observations = self.Observations = observations.unsqueeze(-1)


            # Compute Batch size
            self.BatchSize = observations.shape[0]

            # Initialize sequences
            self.Filtered_State_Means = torch.empty((self.BatchSize,T, self.m, 1))
            self.Filtered_State_Covariances = torch.empty((self.BatchSize,T, self.m, self.m))
            self.Kalman_Gains = torch.empty((self.BatchSize,T,self.m,self.n))
            self.Predicted_State_Means = torch.empty((self.BatchSize,T, self.m, 1))
            self.Predicted_State_Covariances = torch.empty((self.BatchSize,T, self.m, self.m))
            self.Predicted_Observation_Means = torch.empty((self.BatchSize, T ,self.n,1))
            self.Predicted_Observation_Covariances = torch.empty((self.BatchSize, T ,self.n , self.n))
            self.Filtered_Residuals = torch.empty((self.BatchSize, T ,self.n ,1))

            self.F_arr = torch.empty((self.BatchSize,T,self.m,self.m))
            self.H_arr = torch.empty((self.BatchSize,T ,self.n, self.m))


            # Initialize Parameters
            self.Filtered_State_Mean = self.Initial_State_Mean
            self.Filtered_State_Covariance = self.Initial_State_Covariance

            # self.Filtered_State_Means[:,0] = self.Filtered_State_Mean
            # self.Filtered_State_Covariances[:,0] = self.Filtered_State_Covariance

            for t in range(T):

                # Compute Jacobians
                self.UpdateJacobians(t)

                if t == 0:
                    self.Predicted_State_Mean = self.Initial_State_Mean
                    self.Predicted_State_Covariance = self.Initial_State_Covariance
                else:
                    self.predict(t)

                self.KGain(t)
                self.Innovation(observations[:,t])
                self.Correct()

                # Update Arrays
                self.Filtered_State_Means[:,t] = self.Filtered_State_Mean
                self.Filtered_State_Covariances[:,t] = self.Filtered_State_Covariance
                self.Kalman_Gains[:,t] = self.KG
                self.Predicted_State_Means[:,t] = self.Predicted_State_Mean
                self.Predicted_State_Covariances[:,t] = self.Predicted_State_Covariance
                self.Predicted_Observation_Means[:,t] = self.Predicted_Observation_Mean
                self.Predicted_Observation_Covariances[:,t] = self.Predicted_Observation_Covariance
                self.Filtered_Residuals[:,t] = self.Filtered_Residual
                self.F_arr[:,t] = self.F
                self.H_arr[:,t] = self.H

            self.F_arr = self.F_arr.mean(1)
            self.H_arr = self.H_arr.mean(1)

    def LogLikelihood(self,Observations, T):

        with torch.no_grad():

            self.filter(Observations,T)

            loglikelihood = 0.

            for t in range(T):

                Residual = self.Filtered_Residuals[:,t]
                Covariance = self.Predicted_Observation_Covariances[:,t]

                loglikelihood -= 0.5 * (torch.bmm(Residual.mT,
                                  torch.bmm(torch.linalg.pinv(Covariance),
                                            Residual)).squeeze() +
                                  torch.log(torch.linalg.det(Covariance)) +
                                  self.n * torch.log(2*torch.tensor(pi)))

        return loglikelihood


class KalmanSmoother(KalmanFilter):

    def __init__(self, ssModel , em_vars = ['R'], em_averages = ['R']):

        super(KalmanSmoother, self).__init__(ssModel,em_vars,em_averages)

    def SGain(self,t):
        self.UpdateJacobians(t)
        self.SG = torch.bmm(self.Filtered_State_Covariance,
                            torch.bmm(self.F.mT,
                            torch.linalg.pinv(self.Predicted_State_Covariance)))

    def SCorrect(self):

        self.Smoothed_State_Mean = self.Filtered_State_Mean + \
                                   torch.bmm(self.SG,
                                             (self.Smoothed_State_Mean - self.Predicted_State_Mean))

        self.Smoothed_State_Covariance = self.Filtered_State_Covariance + \
                                    torch.bmm(self.SG,
                                              torch.bmm(
                                                  (self.Smoothed_State_Covariance - self.Predicted_State_Covariance),
                                              self.SG.mT))


    def smooth(self,observations, T):

        self.filter(observations,T)

        self.Smoothed_State_Means = torch.empty((self.BatchSize, T, self.m ,1))
        self.Smoothed_State_Covariances = torch.empty((self.BatchSize, T ,self.m ,self.m))
        self.SGains = torch.empty((self.BatchSize, T-1 , self.m ,self.m))

        self.Smoothed_State_Mean = self.Filtered_State_Means[:,-1]
        self.Smoothed_State_Covariance = self.Filtered_State_Covariances[:,-1]

        self.Smoothed_State_Means[:,-1] = self.Smoothed_State_Mean
        self.Smoothed_State_Covariances[:, -1] = self.Smoothed_State_Covariance

        for t in reversed(range(T-1)):

            self.Filtered_State_Mean = self.Filtered_State_Means[:,t]
            self.Predicted_State_Mean = self.Predicted_State_Means[:,t+1]
            self.Filtered_State_Covariance = self.Filtered_State_Covariances[:,t]
            self.Predicted_State_Covariance = self.Predicted_State_Covariances[:,t+1]

            self.SGain(t)
            self.SCorrect()

            self.Smoothed_State_Means[:,t] = self.Smoothed_State_Mean
            self.Smoothed_State_Covariances[:,t] = self.Smoothed_State_Covariance
            self.SGains[:,t] = self.SG

    def SmoothPair(self,T):

        self.Pariwise_Covariances = torch.zeros((self.BatchSize, T, self.m, self.m))

        for t in range(1, T):
            self.Smoothed_State_Covariance = self.Smoothed_State_Covariances[:, t]
            self.SG = self.SGains[:, t - 1]

            self.Pariwise_Covariances[:,t] = torch.bmm(self.Smoothed_State_Covariance, self.SG.mT)

    def em(self,num_itts, Observations,T, q_2 ,r_2, states = None):

        self.Q = q_2 * torch.eye(self.m)

        self.R = r_2 * torch.eye(self.n)

        IterationCounter = trange(num_itts, desc = 'EM optimization steps')

        if states != None:
            losses = torch.empty(num_itts)
            loss_fn = torch.nn.MSELoss(reduction='mean')

        for n in IterationCounter:

            self.smooth(Observations, T)

            self.SmoothPair(T)

            self.U_xx = torch.einsum('BTmp,BTpn->BTmn', (self.Smoothed_State_Means, self.Smoothed_State_Means.mT))
            self.U_xx += self.Smoothed_State_Covariances

            self.U_yx = torch.einsum('BTnp,BTpm->BTnm', (self.Observations, self.Smoothed_State_Means.mT))

            self.U_yy = torch.einsum('BTmp,BTpn->BTmn', (self.Observations, self.Observations.mT))

            self.V_xx = self.U_xx[:,:-1]

            self.V_x1x = torch.einsum('BTmp,BTpn->BTmn',(self.Smoothed_State_Means[:,1:], self.Smoothed_State_Means[:,:-1].mT))
            self.V_x1x += self.Pariwise_Covariances[:,1:]

            self.V_x1x1 = self.U_xx[:,1:]

            if 'H' in self.em_vars:
                self.EM_Update_H()

            if 'R' in self.em_vars:
                self.EM_Update_R()

            if 'F' in self.em_vars:
                self.EM_Update_F()

            if 'Q' in self.em_vars:
                self.EM_Update_Q()

            self.Average_EM_vars()

            if states != None:

                loss = loss_fn(self.Smoothed_State_Means.squeeze(), states.squeeze())
                IterationCounter.set_description('Iteration loss: {} [dB]'.format(10*torch.log10(loss).item()))


    def EM_Update_H(self):

        self.H_arr = torch.bmm(self.U_yx.mean(1),torch.linalg.pinv(self.U_xx.mean(1)))


    def EM_Update_R(self):

        HU_xy = torch.einsum('Bmp,BTpn->BTmn',(self.H_arr,self.U_yx.mT))

        R_arr = self.U_yy - HU_xy - HU_xy.mT + torch.einsum('Bmp,BTpk,Bkn->BTmn',(self.H_arr,self.U_xx,self.H_arr.mT))

        self.R_arr = 0.5 * R_arr + 0.5 * R_arr.mT


    def EM_Update_F(self):

        self.F_arr = torch.einsum('Bmp,Bpn->Bmn',(self.V_x1x.mean(1), torch.linalg.pinv(self.V_xx.mean(1))))

    def EM_Update_Q(self):

        FV_xx1 = torch.einsum('Bmp,BTpn->BTmn',(self.F_arr, self.V_x1x.mT))

        Q_arr = self.V_x1x1 - FV_xx1 - FV_xx1.mT +\
                     torch.einsum('Bmp,BTpk,Bkn->BTmn',(self.F_arr,self.V_xx,self.F_arr.mT))
        Q_arr = torch.cat((Q_arr,Q_arr[:,0].unsqueeze(1)),dim = 1)

        self.Q_arr = 0.5 * Q_arr + 0.5 * Q_arr.mT

        # window_size = 360
        # Inter = self.Q_arr.unfold(1, window_size, 1).mean(-1)
        # self.Q_arr = torch.cat((Inter, self.Q_arr[:,-window_size+1:]), dim=1)

    def Average_EM_vars(self):

        for value in self.em_averages:
            self.__setattr__(f'{value}_arr', self.__getattribute__(f'{value}_arr').mean(1))










        
        

if __name__ == '__main__':

    torch.set_default_tensor_type(torch.DoubleTensor)

    torch.random.manual_seed(421)

    loss_fn = torch.nn.MSELoss()

    ssModel = SystemModel(lambda x,t: x,1, lambda x,t:x[:2], 1,360,360,3,2)
    ssModel.InitSequence(torch.zeros((3)), torch.eye(3))
    ssModel.GenerateBatch(10, 360)
    state,obs = ssModel.Target, ssModel.Input
    kf = KalmanSmoother(ssModel, em_vars=['R','H','F','Q'], em_averages=['R','H'])#, em_vars= ['R','Q'],em_averages=['R','Q'])

    kf.InitSequence(torch.zeros((10,3,1)), torch.eye(3).unsqueeze(0).repeat(10,1,1))
    # kf.InitSequence(torch.zeros((3,1)), torch.eye(3))
    kf.smooth(obs.mT, 360)

    print(10*torch.log10(loss_fn(kf.Smoothed_State_Means.squeeze(),state.mT)).item())



    kf.em(10,obs.mT,360,25,1,state.mT)
    print(10*torch.log10(loss_fn(kf.Smoothed_State_Means.squeeze(),state.mT)).item())


