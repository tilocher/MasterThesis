# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
#
import scipy.special
import torch
import numpy as np
from tqdm import trange
from SystemModels.Extended_sysmdl import SystemModel
from PriorModels.BasePrior import BasePrior






class TaylorPrior(BasePrior):

    def __init__(self,**kwargs):

        taylorOrder = kwargs['TaylorOrder'] if 'TaylorOrder' in kwargs.keys() else 1
        deltaT = kwargs['deltaT'] if 'deltaT' in kwargs.keys() else 1
        channels = kwargs['channels'] if 'channels' in kwargs.keys() else 2



        super(TaylorPrior, self).__init__()

        assert taylorOrder >= 1, 'Taylor order must be at least 1'
        self.taylor_order = taylorOrder
        self.deltaT = deltaT

        self.channels = channels

        self.basis_functions = torch.from_numpy(np.array([[deltaT**k/scipy.special.factorial(k)] for k in range(1,taylorOrder + 1)])).float()

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
            self.window_size = 1
            self.window_parameters = 1

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
                    self.window_parameters = 1

                self.window_weights = torch.from_numpy(
                    np.array([ self.window_parameters * np.exp(- (w - int(self.window_size/2))**2 /(2*self.window_parameters))
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

            self.window_weights = torch.diag(self.window_weights).float()

            self.window_sum = torch.sum(self.window_weights)
            self.n_prediction_weight = torch.arange(-int(self.window_size/2)+1,int(self.window_size/2)+2).reshape(-1,1).float()




    def fit(self,data):

        self.T = data.shape[-1]

        data = data.detach().cpu()

        with torch.no_grad():

            if self.window == '':
                return self._FitWithoutWindow(data)

            else:
                return self._FitWithWindow(data)

    def _FitWithWindow(self,data):


        # time_steps = self.time_steps =  data.shape[-1]

        batch_size, channels, time_steps = data.shape

        self.time_steps = time_steps
        self.channels = channels


        basis_functions = self.basis_functions.reshape((1, 1, -1)).repeat((self.window_size,batch_size, 1))


        weights = torch.diag(self.window_weights)


        coefficients = torch.zeros((self.taylor_order,channels, time_steps))
        covariances = torch.zeros((self.taylor_order, self.taylor_order, time_steps))

        half_window = int(self.window_size/2)

        lower_bound = - half_window
        upper_bound = half_window + 1

        padded_data = torch.nn.functional.pad(data, (-lower_bound, upper_bound), 'replicate')


        for t in trange(0, time_steps, desc = 'Calculating prior'):

            current_state = padded_data[:,:,t -lower_bound-half_window :t+half_window - lower_bound + 1]
            observations = padded_data[:,:,t -lower_bound-half_window+1 :t+half_window - lower_bound + 2]

            target_tensor = (observations - current_state)

            target_tensor = torch.transpose(target_tensor,0,-1).mT


            covariance = (torch.mm(self.basis_functions,self.basis_functions.T)*batch_size).repeat(self.window_size,1,1)

            Y = torch.bmm(basis_functions.mT,target_tensor)


            weighted_cov = (weights*torch.transpose(covariance,0 ,-1)).sum(-1)
            weighted_Y = (weights * torch.transpose(Y, 0, -1)).sum(-1)


            theta = torch.mm(torch.linalg.pinv(weighted_cov),weighted_Y.T)

            coefficients[...,t] = theta
            covariances[...,t] = weighted_cov


        self.coefficients = coefficients
        self.covariances = covariances

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

    def f(self,x,t):
        t = t + self.offset
        return (x.squeeze() + (self.basis_functions.T @ self.coefficients[...,t] ).squeeze()).unsqueeze(-1)

    @property
    def gradients(self):
        basis_functions = self.basis_functions.reshape((1,-1,1))
        basis_functions = basis_functions.repeat((self.T,1,1))


        return torch.bmm(torch.transpose(self.coefficients.unsqueeze(2),0,1).mT, basis_functions)


    def FJacobian(self,x,t):
        return torch.eye(self.channels)

    def h(self,x, t):
        return x

    def HJacobian(self,x, t):
        return torch.eye(self.channels)

    def getSysModel(self,timesteps = None, offset = 0, gpu = False):


        self.offset = offset

        if timesteps == None:
            timesteps = self.time_steps

        if gpu:
            dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.ssModel = SystemModel(self.f, 0, self.h, 0, timesteps, timesteps, self.channels,
                                       self.channels)

            self.basis_functions = self.basis_functions.to(dev)
            self.ssModel.setFJac(self.FJacobian)
            self.ssModel.setHJac(self.HJacobian)


        self.ssModel = SystemModel(self.f, 0, self.h, 0,timesteps, timesteps,self.channels,self.channels)
        self.ssModel.setFJac(self.FJacobian)
        self.ssModel.setHJac(self.HJacobian)



        return self.ssModel

    def Identity(self, x, t):
        return x

    def UpdateWeights(self,y):
        """
        Function to update weights recursively online
        :param:
        x: Current state
        y: New Observation
        t: timestep of the weights
        """
        gamma = 0.97
        Current_weights = torch.transpose(self.coefficients,0,-1)
        Current_covariance = torch.transpose(self.covariances,0,-1)

        batched_basis = self.basis_functions.unsqueeze(0).repeat(self.time_steps,1,self.window_size)
        y = torch.nn.functional.pad(y.squeeze(), (0,0,self.window_size,0), 'constant')
        y = y.unfold(0,self.window_size,1)



        prediction = y[:-1] + torch.bmm(torch.transpose(self.coefficients,0,-1),
                                             batched_basis).squeeze()


        Innovation = (y[1:] - prediction) *  torch.diag(self.window_weights)

        Gain = torch.bmm(Current_covariance,batched_basis)

        Var = torch.bmm(batched_basis.mT,torch.bmm(Current_covariance,batched_basis))*  torch.diag(self.window_weights)

        Gain = torch.bmm(Gain, torch.linalg.pinv(gamma + Var))

        Updated_weights = Current_weights + torch.bmm(Gain,Innovation.mT).mT

        Updated_covariance = 1/gamma*(Current_covariance -  torch.bmm(Gain,
                                                            torch.bmm(batched_basis.mT, Current_covariance)))

        self.coefficients = torch.transpose(Updated_weights,0,-1)
        self.covariances = torch.transpose(Updated_covariance,0,-1)





