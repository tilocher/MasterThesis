# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import torch
import numpy as np
from Code.ECG_Modeling.SystemModels.Extended_sysmdl import SystemModel


class Taylor_model():

    def __init__(self, taylor_order = 1):

        assert taylor_order >= 1, 'Taylor order must be at least 1'
        self.taylor_order = taylor_order

    def fit(self, data: torch.tensor, batch_first=True):

        if not batch_first: data = torch.transpose(data, 0, -1)

        data = data.squeeze()

        time_steps = data.shape[-1]

        data = data.reshape((-1,time_steps))


        coefficients = torch.ones((time_steps, self.taylor_order))

        for t in range(1, time_steps):
            input_tensor = torch.tensor(np.array([data[:, t].numpy() ** i for i in range(1, self.taylor_order + 1)])).T
            target_tensor = data[:, t].unsqueeze(1)
            coefficients[t] = torch.linalg.lstsq(target_tensor, input_tensor).solution

        self.coefficients = coefficients.float()

        return coefficients

    def predict(self, x, t):

        x_input = torch.tensor(
                np.array([x.detach().numpy() ** i for i in range(1, self.taylor_order + 1)])).T.float()
        if np.abs((x_input @ self.coefficients[int(t)]).float()) > 10:
            print('ola')

        return (x_input @ self.coefficients[int(t)]).float()

    def Jacobian(self,x,t):
        return torch.atleast_2d(self.coefficients[t,0])

    def GetSysModel(self):


        self.Taylor_model = SystemModel(self.predict, 0, lambda x,t: x, 0,0,0,1,1)
        self.Taylor_model.setFJac(self.Jacobian)

        return self.Taylor_model


if __name__ == '__main__':


    from Code.ECG_Modeling.Filters.EM import EM_algorithm


    from Code.ECG_Modeling.DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH

    snr = 10

    loader = PhyioNetLoader_MIT_NIH(1,1,SNR_dB=snr,random_sample=False)



    obs,state = loader.GetData(2)


    taylor_model = Taylor_model(1)
    taylor_model.fit(obs)
    ssModel = taylor_model.GetSysModel()
    ssModel.InitSequence(torch.randn((1)) , torch.randn((1,1))**2)


    EM = EM_algorithm(ssModel,Plot_title= 'SNR: {} [dB]'.format(snr), units= 'mV')

    EM.EM(obs,state,num_itts=20, q_2= 0.0001,Plot= 'SNR_{}'.format(snr))