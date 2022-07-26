"""# **Class: KalmanNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func

class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self,gpu):
        super().__init__()

        if torch.cuda.is_available() and gpu:
            dev = torch.device("cuda:0")
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            dev = torch.device("cpu")

        self.device = dev
        self.to(self.device)
    #############
    ### Build ###
    #############
    def Build(self, ssModel):



        self.InitSystemDynamics(ssModel.f, ssModel.h, ssModel.m, ssModel.n, infoString = "partialInfo")

        self.ssModel = ssModel

        self.T = ssModel.T

        # Number of neurons in the 1st hidden layer
        H1_KNet = (ssModel.m + ssModel.n) * (10) * 8

        # Number of neurons in the 2nd hidden layer
        H2_KNet = (ssModel.m * ssModel.n) * 1 * (10)

        self.InitKGainNet(H1_KNet, H2_KNet)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, H1, H2):
        # Input Dimensions
        D_in = self.m + self.n  # x(t-1), y(t)

        # Output Dimensions
        D_out = self.m * self.n  # Kalman Gain

        ###################
        ### Input Layer ###
        ###################
        # Linear Layer
        self.KG_l1 = torch.nn.Linear(D_in, H1, bias=True).to(self.device,non_blocking = True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu1 = torch.nn.ReLU()

        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        self.hidden_dim = (self.m * self.m + self.n * self.n) * 10
        # Number of Layers
        self.n_layers = 1
        # Batch Size
        self.batch_size = 1
        # Input Sequence Length
        self.seq_len_input = 1
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # batch_first = False
        # dropout = 0.1 ;

        # Initialize a Tensor for GRU Input
        # self.GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)

        # Initialize a Tensor for Hidden State
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim).to(self.device,non_blocking = True)

        # Iniatialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers).to(self.device,non_blocking = True)

        ####################
        ### Hidden Layer ###
        ####################
        self.KG_l2 = torch.nn.Linear(self.hidden_dim, H2, bias=True).to(self.device,non_blocking = True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu2 = torch.nn.ReLU()

        ####################
        ### Output Layer ###
        ####################
        self.KG_l3 = torch.nn.Linear(H2, D_out, bias=True).to(self.device,non_blocking = True)
        # torch.nn.init.constant_(self.KG_l3.weight, 0)
        # torch.nn.init.constant_(self.KG_l3.bias, 0)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, f, h, m, n, infoString='fullInfo'):

        if (infoString == 'partialInfo'):
            self.fString = 'ModInacc'
            self.hString = 'ObsInacc'
        else:
            self.fString = 'ModAcc'
            self.hString = 'ObsAcc'

        # Set State Evolution Function
        self.f = f
        self.m = m

        # Set Observation Function
        self.h = h
        self.n = n




    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0):

        self.m1x_prior = M1_0.to(self.device,non_blocking = True)

        self.m1x_posterior = M1_0.to(self.device,non_blocking = True)

        self.state_process_posterior_0 = M1_0.to(self.device,non_blocking = True)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self,t):


        # Predict the 1-st moment of x
        self.m1x_prev_prior = self.m1x_prior

        self.m1x_prior = torch.squeeze(self.f(self.m1x_posterior, t))

        # Predict the 1-st moment of y
        self.m1y = torch.squeeze(self.h(self.m1x_prior, t))

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):

        # Reshape and Normalize the difference in X prior
        #dm1x = self.m1x_prior - self.state_process_prior_0
        dm1x = self.m1x_posterior - self.m1x_prev_prior
        dm1x_reshape = torch.squeeze(dm1x)
        dm1x_norm = func.normalize(dm1x_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Normalize y
        dm1y = y - torch.squeeze(self.m1y)
        dm1y_norm = func.normalize(dm1y, p=2, dim=0, eps=1e-12, out=None)

        # KGain Net Input
        KGainNet_in = torch.cat([dm1y_norm, dm1x_norm], dim=-1)
        # KGainNet_in = torch.cat([KGainNet_in,y],dim = 0)

        # Kalman Gain Network Step
        KG = self.KGain_step(KGainNet_in)


        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (-1,self.m, self.n))
        # del KG,KGainNet_in,dm1y,dm1x,dm1y_norm,dm1x_norm,dm1x_reshape


    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y,t):
        # Compute Priors
        self.step_prior(t)

        # y = (y.reshape((self.n,1))-self.m1y).squeeze()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        y_obs = torch.unsqueeze(y, 1)
        dy = y_obs.squeeze() - self.m1y

        # Compute the 1-st posterior moment
        INOV = torch.bmm(self.KGain, dy.unsqueeze(-1))
        self.m1x_posterior = self.m1x_prior + INOV.squeeze()

        del INOV,dy,y_obs

        return torch.squeeze(self.m1x_posterior )


    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, KGainNet_in):

        ###################
        ### Input Layer ###
        ###################
        L1_out = self.KG_l1(KGainNet_in);
        La1_out = self.KG_relu1(L1_out);

        ###########
        ### GRU ###
        ###########
        GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim).to(self.device,non_blocking = True)
        GRU_in[0, :, :] = La1_out
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)
        GRU_out_reshape = torch.reshape(GRU_out, (1,-1, self.hidden_dim))

        ####################
        ### Hidden Layer ###
        ####################
        L2_out = self.KG_l2(GRU_out_reshape)
        La2_out = self.KG_relu2(L2_out)

        ####################
        ### Output Layer ###
        ####################
        self.L3_out = self.KG_l3(La2_out)
        del L2_out,La2_out,GRU_out,GRU_in,GRU_out_reshape,L1_out,La1_out
        return self.L3_out

    ###############
    ### Forward ###
    ###############
    def forward(self, yt, t):
        yt = yt.to(self.device,non_blocking = True)
        return self.KNet_step(yt,t)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self,batch_size):
        self.batch_size = batch_size
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data