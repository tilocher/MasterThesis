import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import torch
from tqdm import trange

from SystemModels.Extended_sysmdl import SystemModel


def DiagBMM(X: torch.Tensor, Y: torch.Tensor):
    return torch.diag_embed(torch.einsum('bii,bii->bi', (X, Y)))


class KalmanFilter():

    def __init__(self, sysModel: SystemModel, em_vars:list=['R'], em_averages:list =['R'], DiagonalMatrices:list = []):
        self.ssModel = sysModel

        self.m = sysModel.m
        self.n = sysModel.n

        self.f = sysModel.f
        self.h = sysModel.h

        self.Q = sysModel.Q
        self.R = sysModel.R

        self.Q_arr = None
        self.R_arr = None

        self.AllVars = ['R', 'Q', 'F', 'H', 'Sigma']

        self.em_vars = em_vars if not em_vars == 'all' else self.AllVars + ['Mu']
        self.em_averages = list(set(em_vars).intersection(em_averages))

        self.OnlyPrior = False

        for var in self.AllVars:
            self.__setattr__(f'{var}isDiag', var in DiagonalMatrices)

        self.AllDiag = np.all([x in DiagonalMatrices for x in self.AllVars])



    def CheckDiag(self):
        if torch.count_nonzero(self.Q - torch.diag(self.Q)) == 0 and torch.count_nonzero(
                self.R - torch.diag(self.R)) == 0:
            self.Diag = True

    def BMM(self, X, Y, XisDiag,YisDiag):

        #XDiag = torch.diag_embed(torch.einsum('bii->bi',X))
        #YDiag = torch.diag_embed(torch.einsum('bii->bi',Y))

        #if torch.count_nonzero(X-XDiag) != 0 and XDiag:
        #    print('WRONG X')
        #if torch.count_nonzero(Y-YDiag) != 0 and YisDiag:
        #    print('WRONG Y')

        if XisDiag and YisDiag and X.shape[-1] == X.shape[-2] and Y.shape[-1] == Y.shape[-2]:
            return torch.diag_embed(torch.einsum('bii,bij->bi', (X, Y)))
        elif XisDiag and not YisDiag:
            return torch.einsum('bii,bij->bij', (X, Y))
        elif not XisDiag and YisDiag:
            return torch.einsum('bij,bjj->bij', (X, Y))
        else:
            return torch.bmm(X, Y)

    def PINV(self,X,isdiag):

        if isdiag:
            return torch.diag_embed(torch.nan_to_num(1/torch.einsum('bii->bi',X), 0,0,0))
        else:
            return torch.linalg.pinv(X)
    def f_batch(self, state, t):
        return torch.stack([self.f(x, t) for x in state])

    def h_batch(self, state, t):
        return torch.stack([self.h(x, t) for x in state])

    def UpdateQ(self, Q):
        self.Q = Q

    def UpdateR(self, R):
        self.R = R

    def GetQ(self, t):

        if self.Q_arr == None:
            return self.Q
        else:
            if 'Q' in self.em_averages:
                return self.Q_arr
            else:
                return self.Q_arr[:, t]

    def GetR(self, t):

        if self.R_arr == None:
            return self.R
        else:
            if 'R' in self.em_averages:
                return self.R_arr
            else:
                return self.R_arr[:, t]

    def ResetQR(self):
        self.Q_arr = None
        self.R_arr = None

    def InitSequence(self, InitialMean=None, InitialCovariance=None):

        self.InitMean(InitialMean)

        self.InitCovariance(InitialCovariance)

    def InitMean(self, InitialMean=None):

        if InitialMean == None:
            self.Initial_State_Mean = self.ssModel.m1x_0
        else:
            self.Initial_State_Mean = InitialMean

        if len(self.Initial_State_Mean.shape) == 2:
            self.Initial_State_Mean = self.Initial_State_Mean.unsqueeze(0)

    def InitCovariance(self, InitialCovariance=None):

        if InitialCovariance == None:
            self.Initial_State_Covariance = self.ssModel.m2x_0
        else:
            self.Initial_State_Covariance = InitialCovariance

        if len(self.Initial_State_Covariance.shape) == 2:
            self.Initial_State_Covariance = self.Initial_State_Covariance.unsqueeze(0)

    def UpdateJacobians(self, t):

        # Update Gradients
        self.F = torch.stack([self.ssModel.getFJacobian(x.squeeze(), t) for x in self.Filtered_State_Mean])
        self.H = torch.stack([self.ssModel.getHJacobian(x.squeeze(), t) for x in self.Filtered_State_Mean])

    def predict(self, t):

        # Predict the 1-st moment of x
        self.Predicted_State_Mean = self.f_batch(self.Filtered_State_Mean, t)

        # Compute Jacobians
        self.UpdateJacobians(t)

        # Predict the 2-nd moment of x
        self.Predicted_State_Covariance = self.BMM(self.Filtered_State_Covariance,self.F.mT,
                                                   self.AllDiag, self.FisDiag)
        self.Predicted_State_Covariance = self.BMM(self.F, self.Predicted_State_Covariance,
                                                   self.FisDiag, self.AllDiag)
        self.Predicted_State_Covariance += self.GetQ(t)

        # Predict the 1-st moment of y
        self.Predicted_Observation_Mean = self.h_batch(self.Predicted_State_Mean, t)
        # Predict the 2-nd moment y
        self.Predicted_Observation_Covariance = self.BMM(self.Predicted_State_Covariance, self.H.mT,
                                                         self.AllDiag,self.HisDiag)
        self.Predicted_Observation_Covariance = self.BMM(self.H,self.Predicted_Observation_Covariance,
                                                         self.HisDiag, self.AllDiag)
        self.Predicted_Observation_Covariance += self.GetR(t)

    def KGain(self, t):

        # Compute Kalman Gain
        self.KG = self.PINV(self.Predicted_Observation_Covariance, self.AllDiag)
        self.KG = torch.bmm(self.H.mT,self.KG)
        self.KG = torch.bmm(self.Predicted_State_Covariance,self.KG)

        if self.OnlyPrior:
            self.KG = torch.zeros_like(self.KG)

    def Innovation(self, y):

        self.Observation = y

        # Compute Innovation
        self.Predicted_Residual = (self.Observation - self.Predicted_Observation_Mean)

    def Correct(self):

        # Compute the 1-st posterior moment
        self.Filtered_State_Mean = self.Predicted_State_Mean + torch.bmm(self.KG, self.Predicted_Residual)

        # Compute the 2-nd posterior moments
        self.Filtered_State_Covariance = self.BMM(self.H,self.Predicted_State_Covariance,self.HisDiag,self.AllDiag)
        self.Filtered_State_Covariance = torch.bmm(self.KG,self.Filtered_State_Covariance)
        self.Filtered_State_Covariance = self.Predicted_State_Covariance - self.Filtered_State_Covariance

        self.Filtered_Residual = self.Observation - torch.bmm(self.H, self.Filtered_State_Mean)

    def filter(self, observations, T):

        with torch.no_grad():

            # # Get a batch dimension if there is none
            # observations = self.Observations = torch.atleast_3d(observations).unsqueeze(-1)

            if len(observations.shape) == 2:
                observations = observations.unsqueeze(0)
            observations = self.Observations = observations.unsqueeze(-1)

            # Compute Batch size
            self.BatchSize = observations.shape[0]

            if self.Initial_State_Mean.shape[0] == 1 and self.BatchSize != 1:
                self.Initial_State_Mean = self.Initial_State_Mean.repeat(self.BatchSize, 1, 1)

            if self.Initial_State_Covariance.shape[0] == 1 and self.BatchSize != 1:
                self.Initial_State_Covariance = self.Initial_State_Covariance.repeat(self.BatchSize, 1, 1)

            # Initialize sequences
            self.Filtered_State_Means = torch.empty((self.BatchSize, T, self.m, 1))
            self.Filtered_State_Covariances = torch.empty((self.BatchSize, T, self.m, self.m))
            self.Kalman_Gains = torch.empty((self.BatchSize, T, self.m, self.n))
            self.Predicted_State_Means = torch.empty((self.BatchSize, T, self.m, 1))
            self.Predicted_State_Covariances = torch.empty((self.BatchSize, T, self.m, self.m))
            self.Predicted_Observation_Means = torch.empty((self.BatchSize, T, self.n, 1))
            self.Predicted_Observation_Covariances = torch.empty((self.BatchSize, T, self.n, self.n))
            self.Filtered_Residuals = torch.empty((self.BatchSize, T, self.n, 1))

            self.F_arr = torch.empty((self.BatchSize, T, self.m, self.m))
            self.H_arr = torch.empty((self.BatchSize, T, self.n, self.m))

            # Initialize Parameters
            self.Filtered_State_Mean = self.Initial_State_Mean
            self.Filtered_State_Covariance = self.Initial_State_Covariance

            for t in range(T):
                self.predict(t)

                self.KGain(t)
                self.Innovation(observations[:, t])
                self.Correct()

                # Update Arrays
                self.Filtered_State_Means[:, t] = self.Filtered_State_Mean
                self.Filtered_State_Covariances[:, t] = self.Filtered_State_Covariance
                self.Kalman_Gains[:, t] = self.KG
                self.Predicted_State_Means[:, t] = self.Predicted_State_Mean
                self.Predicted_State_Covariances[:, t] = self.Predicted_State_Covariance
                self.Predicted_Observation_Means[:, t] = self.Predicted_Observation_Mean
                self.Predicted_Observation_Covariances[:, t] = self.Predicted_Observation_Covariance
                self.Filtered_Residuals[:, t] = self.Filtered_Residual
                self.F_arr[:, t] = self.F
                self.H_arr[:, t] = self.H

            self.F_arr = self.F_arr.mean(1)
            self.H_arr = self.H_arr.mean(1)

    def LogLikelihood(self, Observations, T):

        with torch.no_grad():
            self.filter(Observations, T)

            loglikelihood = 0.

            for t in range(T):
                Residual = self.Filtered_Residuals[:, t]
                Covariance = self.Predicted_Observation_Covariances[:, t]

                loglikelihood -= 0.5 * (torch.bmm(Residual.mT,
                                                  torch.bmm(torch.linalg.pinv(Covariance),
                                                            Residual)).squeeze() +
                                        torch.log(torch.linalg.det(Covariance)) +
                                        self.n * torch.log(2 * torch.tensor(pi)))

        return loglikelihood


class KalmanSmoother(KalmanFilter):

    def __init__(self, ssModel, em_vars=['R'], em_averages=['R'], DiagonalMatrices = []):

        super(KalmanSmoother, self).__init__(ssModel, em_vars, em_averages, DiagonalMatrices)

    def SGain(self, t):
        self.UpdateJacobians(t)
        self.SG = torch.bmm(self.Filtered_State_Covariance,
                            torch.bmm(self.F.mT,
                                      torch.linalg.pinv(self.Predicted_State_Covariance)))

        if self.OnlyPrior:
            self.SG = torch.zeros_like(self.SG)

    def SCorrect(self):

        self.Smoothed_State_Mean = self.Filtered_State_Mean + \
                                   torch.bmm(self.SG,
                                             (self.Smoothed_State_Mean - self.Predicted_State_Mean))

        self.Smoothed_State_Covariance = self.Filtered_State_Covariance + \
                                         torch.bmm(self.SG,
                                                   torch.bmm(
                                                       (
                                                                   self.Smoothed_State_Covariance - self.Predicted_State_Covariance),
                                                       self.SG.mT))

    def smooth(self, observations, T):

        self.filter(observations, T)

        self.Smoothed_State_Means = torch.empty((self.BatchSize, T, self.m, 1))
        self.Smoothed_State_Covariances = torch.empty((self.BatchSize, T, self.m, self.m))
        self.SGains = torch.empty((self.BatchSize, T - 1, self.m, self.m))

        self.Smoothed_State_Mean = self.Filtered_State_Means[:, -1]
        self.Smoothed_State_Covariance = self.Filtered_State_Covariances[:, -1]

        self.Smoothed_State_Means[:, -1] = self.Smoothed_State_Mean
        self.Smoothed_State_Covariances[:, -1] = self.Smoothed_State_Covariance

        for t in reversed(range(T - 1)):
            self.Filtered_State_Mean = self.Filtered_State_Means[:, t]
            self.Predicted_State_Mean = self.Predicted_State_Means[:, t + 1]
            self.Filtered_State_Covariance = self.Filtered_State_Covariances[:, t]
            self.Predicted_State_Covariance = self.Predicted_State_Covariances[:, t + 1]

            self.SGain(t)
            self.SCorrect()

            self.Smoothed_State_Means[:, t] = self.Smoothed_State_Mean
            self.Smoothed_State_Covariances[:, t] = self.Smoothed_State_Covariance
            self.SGains[:, t] = self.SG

    def SmoothPair(self, T):

        self.Pariwise_Covariances = torch.zeros((self.BatchSize, T, self.m, self.m))

        for t in range(1, T):
            self.Smoothed_State_Covariance = self.Smoothed_State_Covariances[:, t]
            self.SG = self.SGains[:, t - 1]

            self.Pariwise_Covariances[:, t] = torch.bmm(self.Smoothed_State_Covariance, self.SG.mT)

    def em(self, Observations: torch.Tensor, T: int, q_2: float, r_2: float, num_itts: int = 20,
           states: torch.Tensor = None,
           ConvergenceThreshold=1e-5):

        with torch.no_grad():

            self.Q = q_2 * torch.eye(self.m)

            self.R = r_2 * torch.eye(self.n)

            IterationCounter = trange(num_itts, desc='EM optimization steps')

            if states != None:
                losses = []
                loss_fn = torch.nn.MSELoss(reduction='mean')

            for n in IterationCounter:

                self.smooth(Observations, T)

                self.SmoothPair(T)

                self.U_xx = torch.einsum('BTmp,BTpn->BTmn', (self.Smoothed_State_Means, self.Smoothed_State_Means.mT))
                self.U_xx += self.Smoothed_State_Covariances

                self.U_yx = torch.einsum('BTnp,BTpm->BTnm', (self.Observations, self.Smoothed_State_Means.mT))

                self.U_yy = torch.einsum('BTmp,BTpn->BTmn', (self.Observations, self.Observations.mT))

                self.V_xx = self.U_xx[:, :-1]

                self.V_x1x = torch.einsum('BTmp,BTpn->BTmn',
                                          (self.Smoothed_State_Means[:, 1:], self.Smoothed_State_Means[:, :-1].mT))
                self.V_x1x += self.Pariwise_Covariances[:, 1:]

                self.V_x1x1 = self.U_xx[:, 1:]

                for EmVar in self.em_vars:
                    self.__getattribute__(f'EM_Update_{EmVar}')()

                self.Average_EM_vars()

                if states != None:
                    loss = loss_fn(self.Smoothed_State_Means.squeeze(), states.squeeze())
                    losses.append(10 * torch.log10(loss))
                    IterationCounter.set_description('Iteration loss: {} [dB]'.format(10 * torch.log10(loss).item()))

                if all([self.__getattribute__(f'{i}_diff') < ConvergenceThreshold for i in self.em_vars]):
                    print('Converged')
                    break

            if states != None:
                return torch.tensor(losses)

    def EM_Update_H(self):

        H_arr = torch.einsum('Bmp,Bpn->Bmn', (self.U_yx.mean(1), torch.linalg.pinv(self.U_xx.mean(1))))

        try:
            self.H_diff = torch.abs(torch.mean(H_arr - self.H_arr))
        except TypeError:
            self.H_diff = torch.inf

        self.H_arr = torch.einsum('Bmp,Bpn->Bmn', (self.U_yx.mean(1), torch.linalg.pinv(self.U_xx.mean(1))))

    def EM_Update_Mu(self):

        self.Mu_diff = torch.abs(torch.mean(self.Initial_State_Mean - self.Smoothed_State_Means[:, 0]))

        self.InitMean(self.Smoothed_State_Means[:, 0])

    def EM_Update_Sigma(self):

        self.Sigma_diff = torch.abs(torch.mean(self.Initial_State_Covariance - self.Smoothed_State_Covariances[:, 0]))
        self.InitCovariance(self.Smoothed_State_Covariances[:, 0])

    def EM_Update_R(self):

        HU_xy = torch.einsum('Bmp,BTpn->BTmn', (self.H_arr, self.U_yx.mT))

        R_arr = self.U_yy - HU_xy - HU_xy.mT + torch.einsum('Bmp,BTpk,Bkn->BTmn',
                                                            (self.H_arr, self.U_xx, self.H_arr.mT))

        try:
            self.R_diff = torch.abs(torch.mean(R_arr.mean(1) - self.R_arr))
        except TypeError:
            self.R_diff = torch.inf

        self.R_arr = 0.5 * R_arr + 0.5 * R_arr.mT

    def EM_Update_F(self):

        F_arr = torch.einsum('Bmp,Bpn->Bmn', (self.V_x1x.mean(1), torch.linalg.pinv(self.V_xx.mean(1))))

        try:
            self.F_diff = torch.abs(torch.mean(F_arr - self.F_arr))
        except TypeError:
            self.F_diff = torch.inf

        self.F_arr = F_arr

    def EM_Update_Q(self):

        FV_xx1 = torch.einsum('Bmp,BTpn->BTmn', (self.F_arr, self.V_x1x.mT))

        Q_arr = self.V_x1x1 - FV_xx1 - FV_xx1.mT + \
                torch.einsum('Bmp,BTpk,Bkn->BTmn', (self.F_arr, self.V_xx, self.F_arr.mT))
        Q_arr = torch.cat((Q_arr, Q_arr[:, 0].unsqueeze(1)), dim=1)

        try:
            self.Q_diff = torch.abs(torch.mean(Q_arr - self.Q_arr))
        except TypeError:
            self.Q_diff = torch.inf

        self.Q_arr = 0.5 * Q_arr + 0.5 * Q_arr.mT

        # window_size = 7
        # windowed_Q = torch.empty_like(Q_arr)
        #
        # for t in range(Q_arr.shape[1]):
        #
        #     if t == 360:
        #         windowed_Q[:,-1,:,:] = Q_arr[:,-1]
        #     elif t == 0:
        #         windowed_Q[:,0] = Q_arr[:,0]
        #     else:
        #         upper = min(360,t+int(window_size/2))
        #         lower = max(0,int(window_size/2))
        #
        #         mean = Q_arr[:,lower:upper].mean(1)
        #         windowed_Q[:,t] = mean
        #
        # self.Q_arr = windowed_Q

        # Inter = Q_arr.unfold(1, window_size, 1).mean(-1)
        # self.Q_arr = torch.cat((Inter, Q_arr[:,-window_size+1:]), dim=1)

    def Average_EM_vars(self):

        for value in self.em_averages:
            self.__setattr__(f'{value}_arr', self.__getattribute__(f'{value}_arr').mean(1))

    def SetOnlyPrior(self):
        self.OnlyPrior = True

    def UpdateSSModel(self, ssModel):
        self.ssModel = ssModel

    def em_diag(self, Observations: torch.Tensor, T: int, q_2: float, r_2: float, num_itts: int = 20,
                states: torch.Tensor = None,
                ConvergenceThreshold=1e-5):
        """
        Perform EM assuming that both process noise and observation noise matrices are diagonal
        """

        with torch.no_grad():
            self.Q = q_2 * torch.eye(self.m)
            self.R = r_2 * torch.eye(self.n)

            self.smooth(Observations, T)

            IterationCounter = trange(num_itts, desc='EM optimization steps')

            if states != None:
                losses = []
                loss_fn = torch.nn.MSELoss(reduction='mean')

                self.smooth()


if __name__ == '__main__':
    dim = 100
    dimH = 98
    T = 20

    sys = SystemModel(lambda x,t:x+1, 1, lambda x,t:x[:dimH], 10,T,T,dim,dimH)
    sys.InitSequence(torch.ones(dim,1),torch.eye(dim))

    sys.GenerateBatch(10,T)
    obs = sys.Input

    Smoother = KalmanSmoother(sys,'all',DiagonalMatrices = ['F','H','Q','R','Sigma'] )
    Smoother.InitSequence()
    Smoother.filter(obs.mT,T)

    plt.plot(Smoother.Filtered_State_Means[0,:,0].squeeze())
    plt.plot(sys.Target[0,0].squeeze())
    plt.plot(obs[0,0].squeeze())
    plt.show()





