# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.special import factorial
from Code.ECG_Modeling.SystemModels.Extended_sysmdl import SystemModel

class ConstantModel():

    def __init__(self,state_order,observed_states,q,r,T = 472, deltaT = 1.):

        # Get the order of the systems
        self.state_order = state_order
        self.observed_states = observed_states
        self.observation_order = len(self.observed_states) if isinstance(observed_states, tuple) else 1

        # Set noise statistics
        self.q = q
        self.r = r

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

        self.Q = q**2 * self.base_Q
        self.R = r**2 * torch.eye(self.observation_order)

        def f(x,t):
            return self.F @ x
        def h(x,t):
            return self.H @ x

        self.ssModel = SystemModel(f,1,h, 1, T,T,state_order,self.observation_order)
        self.ssModel.setFJac(lambda x,t: self.F)
        self.ssModel.setHJac(lambda x,y: self.H)
        self.ssModel.UpdateCovariance_Matrix(self.Q,self.R)

    def GetSSModel(self):
        return self.ssModel



if __name__ == '__main__':



    from Code.ECG_Modeling.Filters.EM import EM_algorithm

    from Code.ECG_Modeling.DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH


    m = 3
    n = 1
    q = 1e5

    snr = 0

    loader = PhyioNetLoader_MIT_NIH(1, 1, SNR_dB=snr, random_sample=False)

    obs, state = loader.GetData(4)

    Cmodel = ConstantModel(m, (0), q, 1, obs.shape[-1], deltaT = 1/1000)

    ssModel = Cmodel.GetSSModel()
    ssModel.InitSequence(torch.zeros((m)), torch.eye(m))
    #
    # ssModel.GenerateSequence(ssModel.Q,ssModel.R, 372)
    # plt.plot(ssModel.x[0])
    # plt.plot(ssModel.x[1])
    # plt.plot(ssModel.x[2])
    #
    #
    # plt.show()

    import pykalman

    kf = pykalman.KalmanFilter(Cmodel.F.numpy(),Cmodel.H.numpy(),Cmodel.Q.numpy(),Cmodel.R.numpy(),
                               initial_state_mean= torch.zeros((m)).numpy(),initial_state_covariance=torch.eye(m).numpy(),
                               n_dim_state=Cmodel.state_order, n_dim_obs= 1)

    # ob = np.array(obs.mean((0,1)).unsqueeze(0))
    # ll = kf.loglikelihood(ob)
    kf.em()

    EM = EM_algorithm(ssModel, Plot_title='SNR: {} [dB]'.format(snr), units='mV',parameters=('R','Q'))

    EM.EM(obs, state, num_itts=20, q_2 = 0.1, Q = Cmodel.Q, Plot='_SNR_{}'.format(snr))
    print(EM.ssModel.R)