"""# **Class: Extended Kalman Filter**
Theoretical Non Linear Kalman
"""
import torch
from numpy import pi
pi = torch.tensor(pi)
# from filing_paths import path_model

import sys
# sys.path.insert(1, path_model)
# from model import getJacobian

# if torch.cuda.is_available():
#     dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
#    dev = torch.device("cpu")
#    print("Running on the CPU")

class ExtendedKalmanFilter:

    def __init__(self, SystemModel, mode='full'):
        self.f = SystemModel.f
        self.m = SystemModel.m

        # Has to be transformed because of EKF non-linearity
        self.Q = SystemModel.Q

        self.h = SystemModel.h
        self.n = SystemModel.n

        # Has to be transofrmed because of EKF non-linearity
        self.R = SystemModel.R

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

        self.sysModel = SystemModel

        # Pre allocate KG array
        self.KG_array = torch.zeros((self.T_test,self.m,self.n))

        # Full knowledge about the model or partial? (Should be made more elegant)
        if(mode == 'full'):
            self.fString = 'ModAcc'
            self.hString = 'ObsAcc'
        elif(mode == 'partial'):
            self.fString = 'ModInacc'
            self.hString = 'ObsInacc'

        self.Q_array = SystemModel.Q_array
   
    # Predict
    def Predict(self,t):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.squeeze(self.f(self.m1x_posterior,t)).reshape(self.m,-1)
        # Compute the Jacobians
        self.UpdateJacobians(self.sysModel.getFJacobian(self.m1x_posterior,t), self.sysModel.getHJacobian(self.m1x_prior, t))
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior)
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = torch.squeeze(self.h(self.m1x_prior,t)).reshape(-1,1)
        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H, self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H_T)
        self.KG = torch.matmul(self.KG, torch.linalg.pinv(self.m2y))

        #Save KalmanGain
        self.KG_array[self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = (y.reshape(-1,1) - self.m1y)

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

    def Update(self, y,t):
        self.Predict(t)
        self.KGain()
        self.Innovation(y)
        # self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

        #########################

    def UpdateJacobians(self, F, H):

        if type(F) == tuple:
            F = torch.atleast_2d(F[0].squeeze())

        if type(H) == tuple:
            H = torch.atleast_2d(H[0].squeeze())
        self.F = F
        self.F_T = torch.transpose(F,0,1)
        self.H = H
        self.H_T = torch.transpose(H,0,1)
        #print(self.H,self.F,'\n')
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, y, T, ll = False):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T])
        self.x_prior = torch.empty(size=[self.m, T])
        self.y = torch.empty(size = [self.n, T])
        self.sigma = torch.empty(size=[self.m, self.m, T])
        self.sigma_y = torch.empty(size = [self.n,self.n ,T])
        self.sigma_prior = torch.empty(size=[self.m,self.m,T])
        # Pre allocate KG array
        self.KG_array = torch.zeros((T,self.m,self.n))
        self.i = 0 # Index for KG_array alocation


        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0

        self.log_likelyhood = torch.empty(T)

        for t in range(0, T):
            if self.Q_array != None:
                self.Q = self.Q_array[t]
            yt = torch.squeeze(y[:, t])
            xt,sigmat = self.Update(yt,t)
            self.x[:, t] = torch.squeeze(xt)
            self.x_prior[:,t] = self.m1x_prior.squeeze()
            self.sigma[:, :, t] = torch.squeeze(sigmat)
            self.sigma_prior[:,:,t] = torch.squeeze(self.m2x_prior)
            self.sigma_y[:,:,t] = torch.squeeze(self.m2y)
            self.y[:,t] = torch.squeeze(self.m1y)

            if ll:
                self.log_likelyhood[t] = log_multivariate_normal_density(yt.reshape(1,self.n),self.m1y.reshape(1,1,self.n),
                                                                     self.m2y.reshape(1,self.n,self.n))


    def LogLikelyhood(self,y,T):
        self.GenerateSequence(y,T,ll=True)
        return torch.sum(self.log_likelyhood) + self.m1x_0.reshape(1,-1) @ self.m2x_0 @ self.m1x_0.reshape(-1,1)


def log_multivariate_normal_density(X, means, covars, min_covar=1.e-6):
    """Log probability for full covariance matrices. """
        # only in scipy since 0.9
    solve_triangular = torch.linalg.solve_triangular
    n_samples = 1
    n_dim = X.shape[-1]
    nmix = len(means)
    log_prob = torch.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = torch.linalg.cholesky(cv, upper=False)
        except torch.linalg.LinAlgError:
            # The model is most probabily stuck in a component with too
            # few observations, we need to reinitialize this components
            cv_chol = torch.linalg.cholesky(cv + min_covar * torch.eye(n_dim),
                                      upper=False)
        cv_log_det = 2 * torch.sum(torch.log(torch.diagonal(cv_chol)))
        cv_sol = solve_triangular(cv_chol, (X - mu).T, upper=False).T
        log_prob[:, c] = - .5 * (torch.sum(cv_sol ** 2, dim=1) + \
                                 n_dim * torch.log(2 * pi) + cv_log_det)

    return log_prob