# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
if __name__ == '__main__':



    from Code.ECG_Modeling.Filters.EM import EM_algorithm
    from Code.ECG_Modeling.Filters.ConstantModels import ConstantModel
    import torch

    from Code.ECG_Modeling.DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH
    from Code.ECG_Modeling.Filters.EKF import ExtendedKalmanFilter
    from Code.ECG_Modeling.Filters.Extended_RTS_Smoother import Extended_rts_smoother

    m = 3
    n = 1
    q = 1e5

    snr = 0

    loader = PhyioNetLoader_MIT_NIH(1, 1, SNR_dB=snr, random_sample=False)

    obs, state = loader.GetData(1)

    obs = obs[:,0]
    state = state[:,0]

    Cmodel = ConstantModel(m, (0), q, 1, obs.shape[-1], deltaT = 7e-3)
    dt = 0.0005036314647878088
    ssModelCV = Cmodel.GetSSModel()
    ssModelCV.InitSequence(torch.zeros((m)), torch.eye(m))




    # fstate, _ = kf.smooth(obs)
    MyKF = ExtendedKalmanFilter(ssModelCV)
    MyKF.InitSequence(ssModelCV.m1x_0,ssModelCV.m2x_0)
    MyRTS = Extended_rts_smoother(ssModelCV)

    MyKF.GenerateSequence(torch.tensor(obs),ssModelCV.T)
    MyRTS.GenerateSequence(MyKF.x,MyKF.sigma,MyKF.T)





    from Code.ECG_Modeling.Filters.Taylor_model import Taylor_model

    taylor_model = Taylor_model(1)
    taylor_model.fit(obs)
    ssModel = taylor_model.GetSysModel()
    ssModel.InitSequence(torch.randn((1)), torch.randn((1, 1)) ** 2)
    ssModel.F = ssModel.getFJacobian(0,0)
    ssModel.HJacSet = True
    ssModel.H = torch.eye(1)
    ssModel.Q =  0.0001 * torch.eye(1)
    ssModel.R = 0.001 * torch.eye(1)

    ssModel = Cmodel

    print(MyKF.LogLikelyhood(obs, ssModelCV.T).item())



    def funcToMin(t):
        print('t:',t)
        ConstMopdel = ConstantModel(m, (0), q, 1, obs.shape[-1], deltaT=t)
        Kf = ExtendedKalmanFilter(ConstMopdel.GetSSModel())
        Kf.InitSequence(torch.zeros((m)), torch.eye(m))
        ll = Kf.LogLikelyhood(obs,ssModelCV.T)

        return -ll.item()

    # minimize_scalar(funcToMin,bounds=(1e-9,1000),method='bound')

    obs = obs.numpy()
    state = state.numpy()

    import pykalman

    kf = pykalman.KalmanFilter(ssModel.F.numpy(), ssModel.H.numpy(), ssModel.Q.numpy(), ssModel.R.numpy(),
                               initial_state_mean=torch.zeros(ssModel.m).numpy(),
                               initial_state_covariance=torch.eye(ssModel.m).numpy(),
                               n_dim_state=ssModel.m, n_dim_obs=1,
                               )#em_vars=['transition_covariance', 'observation_covariance'])

    # kf = kf.em(obs,n_iter=10)
    fstate,_ = kf.smooth(obs)
    plt.plot(obs.squeeze(),label = 'noisy signal',alpha = 0.3, color= 'r')
    plt.plot(state.squeeze(), label = 'gt', color = 'g')
    plt.plot(fstate[:,0].squeeze(),label = 'smoothed with lib', color = 'b')
    plt.plot(MyRTS.s_x[0].squeeze(), label='smoothed with My', color='c')
    # plt.plot(torch.exp(MyKF.log_likelyhood), label='smoothed with My', color='c')
    plt.legend()
    plt.show()
    # plt.plot(MyRTS.s_x[0] - fstate[:,0] , label = 'diffrence')
    # plt.legend()
    # plt.show()

    kf.loglikelihood(obs)

    # ob = np.array(obs.mean((0,1)).unsqueeze(0))
    # ll = kf.loglikelihood(ob)
    # kf.em()