# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import warnings

import numpy as np
import torch
import scipy.special
from matplotlib import pyplot as plt
from tqdm import trange
from datetime import datetime
from Code.ECG_Modeling.SystemModels.Extended_sysmdl import SystemModel


# System model based on https://ieeexplore.ieee.org/abstract/document/1186732

# Get a value for pi
pi = np.pi

# Enable Latex in matplotlib
plt.rcParams['text.usetex'] = True



def RateOfChange(state: torch.tensor, thetas: torch.tensor, a: torch.tensor, b: torch.tensor,
                 A_sin: float, f2: float, t: float) -> (torch.tensor, torch.tensor):
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
    x = state[:, 0]
    y = state[:, 1]
    z = state[:, 2]
    w = state[:, 3]

    # Initialize the evolution matrix
    A = torch.zeros((x.shape[0], 4, 4))

    # Calculate evolution parameters:

    # Angle of the trajectory
    alpha = 1 - torch.sqrt(x ** 2 + y ** 2)

    # Phase of the trajectory
    theta = torch.atan2(x, y)

    # Fill evolution matrix

    # Rotational part
    A[:, 0, 0] = alpha
    A[:, 0, 1] = -w
    A[:, 1, 0] = w
    A[:, 1, 1] = alpha

    # Baseline wanderer
    A[:, 2, 2] = -1

    # Initialize input matrix
    B = torch.zeros((x.shape[0], 4))

    # Set up change in amplitude
    dz = 0

    # Update change in amplitude
    for a_i, b_i, theta_i in zip(a, b, thetas):
        d_theta = (theta - theta_i)
        dz += a_i * d_theta * torch.exp(-d_theta ** 2 / (2 * b_i ** 2))

    # Add baseline wanderer
    dz -= (A_sin * torch.sin(2 * torch.tensor(pi * f2 * t)))

    # Fill input matrix
    B[:, 2] = dz

    return A, B


class ECG_signal():

    def __init__(self, f2: float = 0.25, A_sin: float = 0.15, batch_size: int = 1) -> None:
        """
        Initialize the Class to produce synthetic ECG signals
        :param f2: Frequency of the baseline wanderer
        :param A_sin: Amplitude of the baseline wanderer
        :param batch_size: Number of batches to be generated
        """

        # Set batch size
        self.batch_size = batch_size

        # Parameters for Matrix exponential
        self.deltaT = 1e-3
        self.order = 3

        # Parameters: P,Q,R,S,T

        # Phase of the waves, relative to R-Wave
        self.theta = torch.tensor(batch_size * [[-pi / 3, -pi / 12, 0, pi / 12, pi / 2]]).T

        # Peak amplitudes of the waves in [mV]
        self.a = torch.tensor(batch_size * [[1.2, -5., 30., -7.5, 0.75]]).T

        # Standard deviation of waves
        self.b = torch.tensor(batch_size * [[0.25, 0.1, 0.1, 0.1, 0.4]]).T

        # Angular velocity of the system
        self.w = torch.tensor(batch_size * [[2 * pi]])

        # Frequency of the baseline wanderer
        self.f2 = f2

        # Amplitude of the baseline wanderer
        self.A_sin = A_sin

        # Dont ue noise
        self.UseNoise = False

    def SetNoise(self,r_2:float= 0., q_2:float= 0.) -> None:

        # Change useNoise parameter
        self.UseNoise = True

        # Set noise
        self.r_2 = torch.tensor(r_2, dtype=torch.float)
        self.q_2 = torch.tensor(q_2, dtype=torch.float)

        # Set noise in dB
        if not self.r_2 == 0: self.r_2_dB = 10 * torch.log10(self.r_2)
        if not self.q_2 == 0: self.q_2_dB = 10 * torch.log10(self.q_2)

    def SetParams(self,**kwargs):

        for k,v in kwargs.items():
            if k in self.__dict__:
                self.__setattr__(k,v)
            else:
                warnings.warn('Parameter not found')

    def ChangeParameters(self, a: torch.tensor = None, b: torch.tensor = None,
                         theta: torch.tensor = None, w: torch.tensor = None) -> None:
        """
        Chnage the given parameters
        :param a: Peak amplitudes of the waves in [mV]
        :param b: Standard deviation of waves
        :param theta: Phase of the waves, relative to R-Wave
        :param w: Angular velocity of the system
        :return:
        """
        if a != None: self.a = a
        if b != None: self.b = b
        if w != None: self.w = w.unsqueeze(-1)
        if theta != None: self.theta = theta

    def InitSequence(self, m1_x0: torch.tensor) -> None:
        """
        Initialize the Sequence with a given starting vector
        :param m1_x0: The initial state of the system, (x,y,z,w) with shape (batch_size,4)
        :return: None
        """
        # Set Initial state
        self.m1 = m1_x0

        # Initialize propagating state
        self.state = self.m1

        # Initialize time
        self.time = 0

    def step(self) -> None:
        """
        Update the state of the system
        :return: None
        """

        # Get the Rate of change at the current time step according to the system of equations
        # A is the evolution matrix and the B the input matrix of a linear ss Model
        A, B = RateOfChange(self.state, self.theta, self.a, self.b, self.A_sin, self.f2, self.time)

        # Calculate matrix exponential of the SS-model
        F = torch.zeros(self.state.shape[0], 4, 4)
        F[:] = torch.eye(4)
        for j in range(1, self.order):
            F += torch.matrix_power(A * self.deltaT, j) / scipy.special.factorial(j)



        # Update state with batch size in mind
        self.state = torch.einsum('bij,bj->bi', (F, self.state)) + self.deltaT * B

        if self.UseNoise:
            self.state += (B * self.deltaT) * torch.normal(mean=torch.zeros_like(self.state), std= self.r_2 /self.deltaT)

        # Update Time
        self.time += 1

    def GenerateBatch(self, num_steps: int = 100,
                      init_vec: torch.tensor = None, Random_phase: bool = False) -> torch.tensor:
        """
        Generate a batch of synthetic ECG time-series
        :param num_steps: Number of iteration steps
        :param batches: Number of desired batches
        :param init_vec: Initial amplitude for the heartbeat
        :param Random_phase: Flag if a random starting phase is desired
        :return:
        """

        # get and set params
        batches = self.batch_size
        self.num_steps = num_steps

        # Handle inputs

        # If  a random, uniform phase is desired
        if Random_phase:
            phase = 2 * pi * torch.rand(size=(batches, 1))
            x_state = torch.cos(phase)
            y_state = torch.sqrt(1 - x_state ** 2)
        # Else use 1 for the x state and 0 for y
        else:
            x_state = torch.ones(batches, 1)
            y_state = torch.zeros(batches, 1)

        # If a specific initial amplitude is desired, else 0
        if not init_vec == None:
            z_state = torch.atleast_2d(init_vec)
            assert len(z_state.shape) == 2, 'Initial amplitude must be 2-D vector'
            assert z_state.shape[0] == batches, 'Initial amplitudes must match batch size'
        else:
            z_state = torch.zeros(batches, 1)

        # Set the entire initial state
        InitialState = torch.cat((x_state, y_state, z_state, self.w), dim=-1)
        self.InitSequence(InitialState)

        # Initialize the trajectories vector (batches, 4 (x,y,z,w), num_itts)
        traj = torch.empty((batches, 4, num_steps))

        # Iterate state through time
        for t in trange(num_steps):
            traj[:, :, t] = self.state.squeeze()
            if self.UseNoise and t != 0:
                traj[:, 2, t] += torch.normal(mean=torch.zeros_like(self.state[:,2]), std=self.q_2)
            self.step()
        self.traj = traj
        return traj

    def SamplePlot(self) -> None:
        """
        Plot a random sample from the batch of generated data
        :return: None
        """
        assert 'traj' in self.__dict__, 'No data to plot yet'

        randint = np.random.randint(0,self.batch_size)

        t = np.linspace(0, self.num_steps * self.deltaT / self.w[randint] * 2 * pi, self.num_steps)

        plt.plot(t, traj[randint, 2, :].T, label='Synthetic ECG signal')
        plt.title('Process noise: $r^2 = {}$, Observation noise: $q^2 = {}$'.format(round(self.r_2.item(),2) ,
                                                                                round(self.q_2.item(),2)))
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [mV]')
        plt.legend()
        plt.savefig('..\\Plots\\Synthetic_ECG.pdf')
        plt.show()

    def SaveBatch(self) -> None:
        """
        Save the whole model
        :return: None
        """
        torch.save(self,'..\\Datasets\\Synthetic\\{}.pt'.format(datetime.today().strftime('%d.%m.%y--%H.%M')))



if __name__ == '__main__':
    signal = ECG_signal(batch_size=10000)

    T = 1000
    signal.SetNoise(r_2= 0, q_2= 0.007)
    signal.SetParams(deltaT = 1e-3)
    traj = signal.GenerateBatch(T,Random_phase=False)
    signal.SamplePlot()
    signal.SaveBatch()


