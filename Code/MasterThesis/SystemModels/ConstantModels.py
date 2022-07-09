# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.special import factorial
from Code.ECG_Modeling.SystemModels.Extended_sysmdl import SystemModel
from Code.ECG_Modeling.SystemModels.ECG_statespace import ECG_StateSpace


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

        def f(x,t):
            return self.F @ x
        def h(x,t):
            return self.H @ x

        self.ssModel = SystemModel(f,np.sqrt(q_2),h, np.sqrt(r_2), T,T,state_order,self.observation_order)
        self.ssModel.setFJac(lambda x,t: self.F)
        self.ssModel.setHJac(lambda x,y: self.H)
        self.ssModel.UpdateCovariance_Matrix(self.Q,self.R)

    def GetSSModel(self):
        return self.ssModel

    def UpdateGain(self,q_2,r_2):

        self.q_2 = q_2
        self.r_2 = r_2
        self.Q = q_2 * self.base_Q
        self.R = r_2 * torch.eye(self.observation_order)
        self.ssModel.UpdateCovariance_Matrix(self.Q , self.R)



if __name__ == '__main__':



    from Code.ECG_Modeling.Filters.EM import EM_algorithm

    from Code.ECG_Modeling.DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH

    # Data Parameters

    m = 3
    n = 1
    q = 0.002

    snr = 10
    num_itts = 15

    save_plot = False
    save_loss = False

    # init loss
    losses = torch.empty((4,num_itts))

    loader = PhyioNetLoader_MIT_NIH(1, 2, SNR_dB=snr, random_sample=False)
    obs, state = loader.GetData(num_batches= 2)

    print('-----Constant Acceleration Model-----')

    # constant Models
    Cmodel = ConstantModel(m, 0, q, 1, obs.shape[-1], deltaT = 1)#1/loader.fs)# 7e-3)#1/1000)
    Cmodel.UpdateGain(q / Cmodel.base_Q[0,0], Cmodel.r_2)
    ssModel = Cmodel.GetSSModel()
    ssModel.InitSequence(torch.zeros((m)), torch.eye(m))

    EM = EM_algorithm(ssModel, Plot_title='SNR: {} [dB]'.format(snr), units='mV', parameters=['R'])

    if save_plot:
        loss = EM.EM(obs, state, num_itts=num_itts, q_2=1000, Q=Cmodel.Q, Plot='_SNR_{}_ConstantA'.format(snr))
    else:
        loss = EM.EM(obs, state, num_itts=num_itts, q_2=1000, Q=Cmodel.Q)
    losses[0] = torch.tensor(loss)

    print('-----Constant Velocity Model-----')

    # constant Models
    Cmodel = ConstantModel(m-1, 0, q, 1, obs.shape[-1], deltaT = 1/loader.fs)  # 7e-3)#1/1000)
    Cmodel.UpdateGain(q/ Cmodel.base_Q[0, 0], Cmodel.r_2)
    ssModel = Cmodel.GetSSModel()
    ssModel.InitSequence(torch.zeros((m-1)), torch.eye(m-1))

    EM = EM_algorithm(ssModel, Plot_title='SNR: {} [dB]'.format(snr), units='mV', parameters=['R'])
    if save_plot:
        loss = EM.EM(obs, state, num_itts=num_itts, q_2=1000, Q=Cmodel.Q, Plot='_SNR_{}_ConstantV'.format(snr))
    else:
        loss = EM.EM(obs, state, num_itts=num_itts, q_2=1000, Q=Cmodel.Q)
    losses[1] = torch.tensor(loss)

    print('-----Taylor Model-----')
    # Taylor Model
    from Code.ECG_Modeling.Filters.Taylor_model import Taylor_model
    taylor_model = Taylor_model(1)
    taylor_model.fit(obs)
    ssModel = taylor_model.GetSysModel()
    ssModel.InitSequence(torch.zeros((1)) , torch.eye(1))

    EM = EM_algorithm(ssModel, Plot_title='SNR: {} [dB]'.format(snr), units='mV', parameters=['R'])

    if save_plot:
        loss = EM.EM(obs, state, num_itts=num_itts, q_2= q, Plot='_SNR_{}_Taylor'.format(snr))
    else:
        loss = EM.EM(obs, state, num_itts=num_itts, q_2=q)

    losses[2] = torch.tensor(loss)

    print('-----ECG ODE Model-----')
    # ECG Model from paper
    ssModel = ECG_StateSpace(0.001,1,obs.shape[-1])
    init_vec = torch.zeros(4)
    init_vec[0] = 1
    init_vec[-1] = 2 * np.pi
    init_cov = torch.eye(4)
    ssModel.InitSequence(init_vec,init_cov)
    ssModel.UpdateCovariance_Gain(0.07,1)

    EM = EM_algorithm(ssModel, Plot_title='SNR: {} [dB]'.format(snr), units='mV', parameters=['R'])

    if save_plot:
        loss = EM.EM(obs, state, num_itts=num_itts, q_2= q, Plot='_SNR_{}_ODE'.format(snr))
    else:
        loss = EM.EM(obs, state, num_itts=num_itts, q_2=q)
    losses[3] = torch.tensor(loss)

    if save_loss:
        np.save('..\\Arrays\\Taylor_ConstantVel_DiffEq_snr_{}.npy'.format(snr),losses.detach().numpy())

