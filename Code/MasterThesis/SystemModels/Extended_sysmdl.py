
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

# if torch.cuda.is_available():
#     dev = torch.device("cuda:0")
#     torch.set_default_tensor_type("torch.cuda.FloatTensor")
# else:
#     dev = torch.device("cpu")

class SystemModel:

    def __init__(self, f, q, h, r, T, T_test, m, n, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        if f == 'Identity':
            self.f = self.Identity
            self.setFJac(self.dIdentityF)
        else:
            self.f = f
            self.FJacSet = False

        self.q = q
        self.m = m
        self.Q = q * q * torch.eye(self.m)
        #########################
        ### Observation Model ###
        #########################
        if h == 'Identity':
            self.h = self.Identity
            self.setHJac(self.dIdentityH)
        else:
            self.h = h
            self.HJacSet = False

        self.r = r
        self.n = n
        self.R = r * r * torch.eye(self.n)
        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T
        self.T_test = T_test

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m))
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S

        self.Q_array = None


    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0= None, m2x_0= None):

        if m1x_0 == None:
            self.m1x_0 = torch.zeros((self.m,1))
        else:
            self.m1x_0 = m1x_0

        if m2x_0 == None:
            self.m2x_0 = torch.eye(self.m)
        else:
            self.m2x_0 = m2x_0



    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Gain(self, q, r):

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.r = r
        self.R = r * r * torch.eye(self.n)

    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R

    def getFJacobian(self,x,t):

        # return self.f(x,t)
        if not self.FJacSet:
            return torch.autograd.functional.jacobian(self.f,(x,torch.tensor(t,dtype=torch.float32)))[0]
        else:
            return self.FJac(x,t)

    def setFJac(self,df):
        self.FJacSet = True
        self.FJac = df
        return

    def setHJac(self,dh):
        self.HJacSet = True
        self.HJac = dh
        return

    def getHJacobian(self,x,t):

        if not self.HJacSet:
            return torch.autograd.functional.jacobian(self.h,(x,torch.tensor(t,dtype=torch.float32)))[0]
        else:
            return self.HJac(x,t)

    def UpdateR(self,R):
        self.R = R

    def UpdateQ(self,Q):
        self.Q = Q

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, T):
        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev

        # Generate Sequence Iteratively
        for t in range(0, T):

            ########################
            #### State Evolution ###
            ########################
            # Process Noise
            if self.q == 0:
                xt = self.f(self.x_prev,t)
            else:
                xt = self.f(self.x_prev,t)
                mean = torch.zeros([self.m])
                
                eq = torch.distributions.MultivariateNormal(loc= mean.reshape(1,-1), covariance_matrix=self.Q).sample().reshape(self.m,1)
                         
                # Additive Process Noise
                xt = torch.add(xt,eq)

            ################
            ### Emission ###
            ################
            yt = self.h(xt,t)

            # Observation Noise
            mean = torch.zeros([self.n,1])
            er = torch.normal(mean, self.r)
            # er = np.random.multivariate_normal(mean, R_gen, 1)
            # er = torch.transpose(torch.tensor(er), 0, 1)

            # Additive Observation Noise
            yt = torch.add(yt,er)
            
            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt


    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size, T, randomInit=False, seqInit=False, T_test=0):

        # Allocate Empty Array for Input
        self.Input = torch.empty(size, self.n, T)

        # Allocate Empty Array for Target
        self.Target = torch.empty(size, self.m, T)

        ### Generate Examples
        initConditions = self.m1x_0

        for i in range(0, size):
            # Generate Sequence

            # Randomize initial conditions to get a rich dataset
            if(randomInit):
                variance = 100
                initConditions = torch.rand_like(self.m1x_0) * variance
            if(seqInit):
                initConditions = self.x_prev
                if((i*T % T_test)==0):
                    initConditions = torch.zeros_like(self.m1x_0)

            self.InitSequence(initConditions, self.m2x_0)
            self.GenerateSequence(T)

            # Training sequence input
            self.Input[i, :, :] = self.y

            # Training sequence output
            self.Target[i, :, :] = self.x


    def sampling(self, q, r, gain):

        if (gain != 0):
            gain_q = 0.1
            #aq = gain * q * np.random.randn(self.m, self.m)
            aq = gain_q * q * torch.eye(self.m)
            #aq = gain_q * q * torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        else:
            aq = 0

        Aq = q * torch.eye(self.m) + aq
        Q_gen = np.transpose(Aq) * Aq

        if (gain != 0):
            gain_r = 0.5
            #ar = gain * r * np.random.randn(self.n, self.n)
            ar = gain_r * r * torch.eye(self.n)
            #ar = gain_r * r * torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        else:
            ar = 0

        Ar = r * torch.eye(self.n) + ar
        R_gen = np.transpose(Ar) * Ar

        return [Q_gen, R_gen]


    def SetQ_array(self,Q_array):
        self.Q_array = Q_array

    def Identity(self,x,t):
        return x

    def dIdentityF(self,x,t):
        return torch.eye(self.m)

    def dIdentityH(self,x,t):
        return torch.eye(self.n)