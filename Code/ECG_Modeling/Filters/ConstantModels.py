# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import torch
import numpy as np
from scipy.special import factorial

class ConstantModel():

    def __init__(self,state_order,observed_states,q,r,T = 472, deltaT = 1):

        # Get the order of the systems
        self.state_order = state_order
        self.observed_states = observed_states

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
        self.Q = torch.zeros_like(self.F)

        for i in range(0,state_order):
            for j in range(0,state_order):
                i_prim = (state_order - i)
                j_prim = (state_order - j)
                exponent = (i_prim -1 + j_prim)
                self.Q[i,j] = deltaT**exponent / (exponent * (factorial(j_prim-1) * factorial(i_prim-1)))








if __name__ == '__main__':

    model = ConstantModel(5,(1,2),1,1,100,deltaT=1)
    print('ola')