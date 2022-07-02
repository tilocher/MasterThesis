# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import matplotlib.pyplot as plt
import torch
import os
from Code.ECG_Modeling.Filters.ConstantModels import ConstantModel
from Code.ECG_Modeling.Filters.EKF import ExtendedKalmanFilter
from Code.ECG_Modeling.Filters.Extended_RTS_Smoother import Extended_rts_smoother

if __name__ == '__main__':

    from Code.ECG_Modeling.Filters.EM import EM_algorithm

    from Code.ECG_Modeling.DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH

    # Data Parameters

    m = 2
    n = 1
    # q = 0.002
    q = 1
    r = 10

    snr = 10
    num_itts = 15

    save_plot = False
    save_loss = False

    # init loss
    losses = torch.empty((4,num_itts))

    loader = PhyioNetLoader_MIT_NIH(1, 4 ,5*256, SNR_dB=snr, random_sample=False)
    obs, state = loader.GetData(num_batches= 2)

    obs = obs[0,0]
    state = state[0,0]


    print('-----Constant Acceleration Model-----')
    mean = torch.zeros(m)
    cov = torch.eye(m)

    # constant Models
    Cmodel = ConstantModel(m, 0, q, r, obs.shape[-1], deltaT = 0.1)#1/loader.fs)# 7e-3)#1/1000)
    Cmodel.UpdateGain(q / Cmodel.base_Q[0,0], Cmodel.r_2)
    ssModel = Cmodel.GetSSModel()
    ssModel.InitSequence(mean, cov)


    EM = EM_algorithm(ssModel, Plot_title='SNR: {} [dB]'.format(snr), units='mV', parameters=['R','Q'])

    loss = EM.EM(obs, state, num_itts=num_itts, q_2=1000, Q=Cmodel.Q)

    KF = ExtendedKalmanFilter(ssModel)
    KF.InitSequence(ssModel.m1x_0, ssModel.m2x_0)
    RTS = Extended_rts_smoother(ssModel)



    # Run Loop

    with torch.no_grad():
        KF.GenerateSequence(obs.reshape((ssModel.n, -1)), KF.T_test)
        RTS.GenerateSequence(KF, RTS.T_test)


