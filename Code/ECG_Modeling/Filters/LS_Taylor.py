# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import matplotlib.pyplot as plt
import torch
from Code.ECG_Modeling.SystemModels.ECG_model import ECG_signal, GetNextState, pi
import numpy as np
from Code.ECG_Modeling.Filters.Extended_RTS_Smoother import Extended_rts_smoother
from Code.ECG_Modeling.SystemModels.Extended_sysmdl import SystemModel
from Code.ECG_Modeling.Filters.EKF import ExtendedKalmanFilter

torch.set_default_dtype(torch.float32)

class LS_Taylor():

    def __init__(self, taylor_order = 1):

        assert taylor_order >= 1, 'Taylor order must be at least 1'
        self.taylor_order = taylor_order



    def fit(self, data:torch.tensor, batch_first = True):

        if not batch_first: data = torch.transpose(data,0,-1)

        data = data.squeeze()

        time_steps = data.shape[1]

        coefficients = torch.empty((time_steps-1,self.taylor_order))

        for t in range(time_steps-1):

            input_tensor = torch.tensor(np.array([data[:,t].numpy()**i for i in range(1,self.taylor_order + 1)])).T
            target_tensor = data[:,t].unsqueeze(1)
            coefficients[t] = torch.linalg.lstsq(target_tensor,input_tensor).solution

        self.coefficients = coefficients.float()
        return coefficients

    def Result(self,data):

        time_series = torch.empty((data.shape[1],1))
        time_series[0] = torch.tensor([[0.01]])#data[:,0].mean()


        for t in range(1,data.shape[1]):
            input_tensor = torch.tensor(np.array([time_series[t-1].numpy() ** i for i in range(1, self.taylor_order + 1)]),dtype=torch.float).T
            test = input_tensor @ self.coefficients[t-1]
            time_series[t] = test




        plt.plot(time_series)
        plt.show()

    def RTS_Filter(self,data,gt):


        def f(x,t):
            x_input = torch.tensor(np.array([x.detach().numpy() ** i for i in range(1, self.taylor_order + 1)])).T.float()
            return (x_input @ self.coefficients[int(t)]).float()

        def h(x,t):
            return x

        def h_true(x,t):
            return x[2,:]

        deltaT = 5e-3
        T = int(1 / deltaT)

        sysModel = SystemModel(f,5e-3,h,0.01,T, T-2 , 1,1)

        trueSysmodel = SystemModel(GetNextState,5e-3,h_true,0.01,T,T-2,4,1)


        m1x0 = torch.atleast_2d(torch.tensor(0.))
        m2x0 = torch.atleast_2d(torch.tensor(1.))
        sysModel.InitSequence(m1x0,m2x0)

        # KF_filter = ExtendedKalmanFilter(sysModel)
        # KF_filter.InitSequence(sysModel.m1x_0,sysModel.m2x_0)
        # KF_filter.GenerateSequence(data,sysModel.T_test)
        #
        #
        # RTS_filter = Extended_rts_smoother(sysModel)
        #
        # RTS_filter.GenerateSequence(KF_filter.x,KF_filter.sigma, RTS_filter.T_test)
        # loss = torch.nn.MSELoss()

        # def MAPELoss(output, target):
        #     return torch.mean(torch.abs((target - output) / torch.clip(target,min=1e-7)))
        # loss = MAPELoss
        # print('KF loss: ', 10*torch.log10(loss(KF_filter.x,gt[:,:-2])).item(),'[dB]')
        # print('Unfiltered loss: ', 10 * torch.log10(loss(data[:,:-2], gt[:,:-2])).item(), '[dB]')

        # sysModel = SystemModel(f, 9e-3, h, 0.01, T, T - 2, 1, 1)
        #
        m1x0 = torch.atleast_2d(torch.tensor([1., 0, 0, 2 * pi])).T
        m2x0 = torch.eye(4)
        trueSysmodel.InitSequence(m1x0, m2x0)

        KF_filter_new = ExtendedKalmanFilter(trueSysmodel)
        KF_filter_new.InitSequence(trueSysmodel.m1x_0, trueSysmodel.m2x_0)
        KF_filter_new.GenerateSequence(data, trueSysmodel.T_test)

        # RTS_filter = Extended_rts_smoother(sysModel)
        #
        # RTS_filter.GenerateSequence(KF_filter.x, KF_filter.sigma, RTS_filter.T_test)
        # loss = torch.nn.MSELoss()

        # def MAPELoss(output, target):
        #     return torch.mean(torch.abs((target - output) / torch.clip(target,min=1e-7)))
        # loss = MAPELoss
        # print('KF loss: ', 10 * torch.log10(loss(KF_filter.x, gt[:, :-2])).item(), '[dB]')
        # print('Unfiltered loss: ', 10 * torch.log10(loss(data[:, :-2], gt[:, :-2])).item(), '[dB]')
        # print('KF loss "wrong": ', 10 * torch.log10(loss(KF_filter_new.x, gt[:, :-2])).item(), '[dB]')



        plt.plot(data.squeeze(),label= 'noisy data')
        # plt.plot(KF_filter.x.squeeze(),label= 'KF data')
        plt.plot(KF_filter_new.x[2,:].squeeze(), label='KF wrong data')
        plt.plot(gt.squeeze(),label = 'gt data')
        # plt.plot((RTS_filter.s_x-KF_filter.x).squeeze())
        plt.legend()
        plt.show()

        # filterd = RTS_filter.s_x

        1


if __name__ == '__main__':

    ecg_signal = torch.load('..\\Datasets\\Synthetic\\15.06.22--23.15.pt')

    data_noisy = ecg_signal.traj_noisy[:,2,:]
    data_noiseless = ecg_signal.traj[:,2,:]

    taylor = LS_Taylor(1)

    coeffs = taylor.fit(data_noisy)

    randint = np.random.randint(0,data_noisy.shape[0])

    taylor.RTS_Filter(data_noisy[randint].unsqueeze(0), data_noiseless[randint].unsqueeze(0))

    1

