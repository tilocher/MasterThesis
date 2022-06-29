# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import scipy
import torch
from Code.ECG_Modeling.SystemModels.Extended_sysmdl import SystemModel
import numpy as np


pi = np.pi

class ECG_StateSpace(SystemModel):

    def __init__(self, q, r, T, prior_Q=None, prior_Sigma=None, prior_S=None,
                 deltaT = 1e-3, order = 3, f2 = 0.25, A_sin = 0.15):
        m = 4
        n = 1



        # Parameters for Matrix exponential
        self.deltaT = deltaT
        self.order = order

        # Parameters: P,Q,R,S,T

        # Phase of the waves, relative to R-Wave
        self.theta = torch.tensor([[-pi / 3, -pi / 12, 0, pi / 12, pi / 2]]).T

        # Peak amplitudes of the waves in [mV]
        self.a = torch.tensor( [[1.2, -5., 30., -7.5, 0.75]]).T

        # Standard deviation of waves
        self.b = torch.tensor([[0.25, 0.1, 0.1, 0.1, 0.4]]).T

        # Angular velocity of the system
        self.w = torch.tensor([[2 * pi]])

        # Frequency of the baseline wanderer
        self.f2 = f2

        # Amplitude of the baseline wanderer
        self.A_sin = A_sin

        super(ECG_StateSpace, self).__init__( self.f, q, self.h, r, T, T, m, n, prior_Q=prior_Q,
                                              prior_Sigma=prior_Sigma, prior_S=prior_S)

        self.setFJac(self.f_mat)

    def h(self,state,t):
        return state[2]


    def f(self,state,t):

        # Get the different state components
        x = state[ 0]
        y = state[ 1]
        z = state[ 2]
        w = state[ 3]

        # Initialize the evolution matrix
        A = torch.zeros(( 4, 4))

        # Calculate evolution parameters:

        # Angle of the trajectory
        alpha = 1 - torch.sqrt(x ** 2 + y ** 2)

        # Phase of the trajectory
        theta = torch.atan2(x, y)

        # Fill evolution matrix

        # Rotational part
        A[ 0, 0] = alpha
        A[ 0, 1] = -w
        A[ 1, 0] = w
        A[ 1, 1] = alpha

        # Baseline wanderer
        A[ 2, 2] = -1

        # Initialize input matrix
        B = torch.zeros((4,1))

        # Set up change in amplitude
        dz = 0

        # Update change in amplitude
        for a_i, b_i, theta_i in zip(self.a, self.b, self.theta):
            d_theta = (theta - theta_i)
            dz += a_i * d_theta * torch.exp(-d_theta ** 2 / (2 * b_i ** 2))

        # Add baseline wanderer
        dz -= (self.A_sin * torch.sin(2 * torch.tensor(pi * self.f2 * t)))

        # Fill input matrix
        B[ 2] = dz

        # Calculate matrix exponential of the SS-model
        F = torch.eye(4)

        for j in range(1, self.order):
            F += torch.matrix_power(A * self.deltaT, j) / scipy.special.factorial(j)

        return (F @ state).reshape((4,1)) + (self.deltaT * B)

    def f_mat(self,state,t):

        """
        Calculate the current rate of change based in the state and given parameters for the amplitude, standard
         deviation and phase. Also add a baseline wanderer with a given frequency and amplitude.
        :param state: State of the system (x,y,z,w)
        :param thetas: Phase of the waves, relative to the R-peak
        :param a: Amplitude of the waves
        :param b: Standard deviation of the waves
        :param A_sin: Amplitude of the baseline wanderer
        :param f2: Frequency of the baseline wanderer
        :param t: Current time step
        :return: A,B a tuple of matrices corresponding to the evolution and the input matrix
        """

        # Get the different state components
        x = state[ 0]
        y = state[ 1]
        z = state[ 2]
        w = state[ 3]

        # Initialize the evolution matrix
        A = torch.zeros((4, 4))

        # Calculate evolution parameters:

        # Angle of the trajectory
        alpha = 1 - torch.sqrt(x ** 2 + y ** 2)

        # Phase of the trajectory
        theta = torch.atan2(x, y)

        # Fill evolution matrix

        # Rotational part
        A[ 0, 0] = alpha
        A[ 0, 1] = -w
        A[ 1, 0] = w
        A[ 1, 1] = alpha

        # Baseline wanderer
        A[ 2, 2] = -1

        # Calculate matrix exponential of the SS-model
        F = torch.zeros( 4, 4)
        F[:] = torch.eye(4)
        for j in range(1, self.order):
            F += torch.matrix_power(A * self.deltaT, j) / scipy.special.factorial(j)


        return F
