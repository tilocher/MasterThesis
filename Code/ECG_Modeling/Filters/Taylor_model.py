# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import matplotlib.pyplot as plt
import scipy.special
import torch
import numpy as np
from tqdm import trange

from Code.ECG_Modeling.SystemModels.Extended_sysmdl import SystemModel


class Taylor_model():

    def __init__(self, taylor_order = 1, deltaT = 1, **kwargs):

        assert taylor_order >= 1, 'Taylor order must be at least 1'
        self.taylor_order = taylor_order
        self.deltaT = deltaT

        self.basis_functions = torch.from_numpy(np.array([[deltaT**k/scipy.special.factorial(k)] for k in range(1,taylor_order + 1)])).float()

        # Get window parameters
        if 'window' in kwargs.keys():
            self.window = kwargs['window']

            if 'window_parameters' in kwargs.keys():
                self.window_parameters = kwargs['window_parameters']

            else:
                self.window_parameters = None

            if 'window_size' in kwargs.keys():
                self.window_size = kwargs['window_size']

            else:
                self.window_size = 5
        elif 'window_size' in kwargs.keys():

            self.window,self.window_parameters = 'rectangular', 1.

            self.window_size = kwargs['window_size']

        else:
            self.window = ''

        self.CreateWindow()

    def CreateWindow(self):

        if self.window == '':
            return
        else:

            if self.window == 'rectangular':

                if self.window_parameters == None:
                    self.window_parameters = 1.

                self.window_weights = torch.from_numpy(np.array([self.window_parameters for w in range(self.window_size)]))



            elif self.window == 'exponential':

                if self.window_parameters == None:
                    self.window_parameters = 0.9

                self.window_weights = torch.from_numpy(
                    np.array([self.window_parameters**(np.abs(w - int(self.window_size/2))) for w in range(self.window_size)]))

            elif self.window == 'gaussian':

                if self.window_parameters == None:
                    self.window_parameters = [1,1]

                self.window_weights = torch.from_numpy(
                    np.array([ self.window_parameters[0] * np.exp(- (w - int(self.window_size/2))**2 /(2*self.window_parameters[1]))
                              for w in range(self.window_size)]))

            elif self.window == 'linear':

                if self.window_parameters == None:
                    self.window_parameters = 1.

                slope = 1 / self.window_size

                self.window_weights = torch.from_numpy(
                    np.array([ -slope * np.abs(w - int(self.window_size/2)) + self.window_parameters
                              for w in range(self.window_size)]))

            else:
                raise ValueError('Window not supported')

            self.window_weights = self.window_weights.float()
            self.window_sum = torch.sum(self.window_weights)
            self.n_prediction_weight = torch.arange(-int(self.window_size/2)+1,int(self.window_size/2)+2)





    def fit(self,data):

        if self.window == '':
            return self._FitWithoutWindow(data)

        else:
            return self._FitWithWindow(data)

    def _FitWithWindow(self,data):


        time_steps = self.time_steps =  data.shape[-1]

        data = data.reshape((-1,time_steps))

        batch_size = data.shape[0]

        basis_functions = self.basis_functions.repeat((1,batch_size))
        basis_functions = basis_functions.reshape((batch_size,1,-1))

        projections = torch.linalg.pinv(torch.bmm(basis_functions.mT, basis_functions))

        projections = torch.bmm(projections, basis_functions.mT)


        coefficients = torch.zeros((self.taylor_order, time_steps))
        half_window = int(self.window_size/2)

        lower_bound = - half_window
        upper_bound = half_window + 1

        padded_data = torch.nn.functional.pad(data, (-lower_bound, upper_bound), 'replicate')


        squared_weight_sum = (self.window_weights * self.n_prediction_weight**2).sum()
        squared_weight_sum = (self.n_prediction_weight**2).sum()

        lin_weight_sum = (self.window_weights * self.n_prediction_weight).sum()

        for t in range(0, time_steps):

            current_state = padded_data[:,-lower_bound + t ].reshape((-1,1))
            observations = padded_data[:,t -lower_bound-half_window+1 :t+half_window - lower_bound + 2]

            target_tensor = observations - current_state

            target_tensor = (self.window_weights * self.n_prediction_weight * target_tensor).sum(-1).reshape((-1,1,1))

            c =  (1 / squared_weight_sum) * torch.bmm(projections,target_tensor)
            # c = torch.bmm(projections,target_tensor)
            # c =  torch.bmm(projections,target_tensor)

            coefficients[:,t] = c.mean(0).squeeze()



        self.coefficients = coefficients

        return coefficients



    def _FitWithoutWindow(self, data: torch.tensor):


        time_steps = self.time_steps =  data.shape[-1]

        data = data.reshape((-1,time_steps))

        batch_size = data.shape[0]

        basis_functions = self.basis_functions.repeat((1,batch_size)).T





        coefficients = torch.zeros((self.taylor_order, time_steps))


        for t in range(1,time_steps):
            target_tensor = (data[:,t] - data[:,t-1]).reshape((batch_size,-1))


            c = torch.linalg.lstsq(basis_functions,target_tensor).solution
            coefficients[:,t-1] = c.T

        self.coefficients = coefficients

        return coefficients

    def predict(self,x,t):

        if self.window == '':
            return self._predictNoWindow(x,t)
        else:
            return self._predictWindow(x,t)

    def _predictWindow(self,x,t):


        return (x + self.basis_functions.T @ self.coefficients[:, t])#/ (self.window_weights * self.n_prediction_weight**2).sum() )

    def _predictNoWindow(self, x, t):

        return x + self.basis_functions.T @ self.coefficients[:,t]

    def Jacobian(self,x,t):
        # return torch.atleast_2d(self.coefficients[0,t])
        return torch.ones((1,1))

    def GetSysModel(self):


        self.ssModel = SystemModel(self.predict, 0, lambda x,t: x, 0,self.time_steps, self.time_steps,1,1)
        self.ssModel.setFJac(self.Jacobian)
        self.ssModel.setHJac(lambda x,t: torch.ones((1,1)))

        return self.ssModel


if __name__ == '__main__':


    from Code.ECG_Modeling.Filters.EM import EM_algorithm


    from Code.ECG_Modeling.DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH

    snr = -6

    ts = 360

    loader = PhyioNetLoader_MIT_NIH(1,1, 1*ts,SNR_dB=snr,random_sample=False)

    num_batches = 1000

    obs,state = loader.GetData(num_batches)



    taylor_model = Taylor_model(taylor_order= 5,  window= 'exponential', window_size= 11)#,window_parameters=[1,10])



    taylor_model.fit(obs[:num_batches,0,:])
    #
    ssModel = taylor_model.GetSysModel()
    ssModel.InitSequence(torch.ones((1,1)), torch.eye(1))
    ssModel.GenerateSequence(ssModel.Q,ssModel.R, ssModel.T)
    #
    plt.plot(ssModel.x.squeeze(), label= 'Predicted signal', color = 'b')
    plt.ylabel('Amplitude [mV]')
    plt.xlabel('Time-steps')
    plt.title('Generated signal from learned weights, given {} centered heartbeats'.format(num_batches))
    plt.savefig('..\\Plots\\Taylor_models\\sample_prior_{}.pdf'.format(num_batches))
    plt.show()
    #
    #
    # ssModel = taylor_model.GetSysModel()
    # ssModel.InitSequence(torch.ones((1)) , torch.eye(1))

    taylor_model1 = Taylor_model(taylor_order=5)#, window='gaussian', window_size=19)  # ,window_parameters=[1,10])

    taylor_model1.fit(obs[:num_batches, 0, :])

    # print(torch.max(state[:,0,:],1).values.mean())
    print(ssModel.x.max() / torch.max(state[:,0,:],1).values.mean())
    # #
    # #
    # EM = EM_algorithm(ssModel,Plot_title= 'SNR: {} [dB]'.format(snr), units= 'mV', parameters=['R'])
    # # #
    # EM.EM(obs[-1,0,:],state[-1,0,:],num_itts=20, q_2= 0.001,Plot='_SNR_{}_Taylor'.format(snr))

