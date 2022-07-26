"""# **Class: RTSNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func
from KalmanNet_nn_new_arch import KalmanNetNN


in_mult = 5
out_mult = 40
nGRU = 4

class RTSNetNN(KalmanNetNN):

    ###################
    ### Constructor ###
    ###################
    def __init__(self,gpu= True):
        super().__init__(gpu)

        from KalmanNet_nn import KalmanNetNN

        if torch.cuda.is_available() and gpu:
            self.dev = torch.device("cuda:0")
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            self.dev = torch.device("cpu")

    #############
    ### Build ###
    #############
    def Build(self, ssModel, infoString = 'fullInfo'):

        super().Build(ssModel)

        self.InitRTSGainNet(ssModel.prior_Q, ssModel.prior_Sigma)

    #################################################
    ### Initialize Backward Smoother Gain Network ###
    #################################################
    def InitRTSGainNet(self, prior_Q, prior_Sigma):
        # self.seq_len_input = 1
        # self.batch_size = 1

        self.prior_Q = prior_Q
        self.prior_Sigma = prior_Sigma       


        # BW GRU to track Q
        self.d_input_Q_bw = self.m * in_mult
        self.d_hidden_Q_bw = self.m ** 2
        self.GRU_Q_bw = nn.GRU(self.d_input_Q_bw, self.d_hidden_Q_bw)
        self.h_Q_bw = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Q_bw).to(self.dev, non_blocking=True)

        # BW GRU to track Sigma
        self.d_input_Sigma_bw = self.d_hidden_Q_bw + 2 * self.m * in_mult
        self.d_hidden_Sigma_bw = self.m ** 2
        self.GRU_Sigma_bw = nn.GRU(self.d_input_Sigma_bw, self.d_hidden_Sigma_bw)
        self.h_Sigma_bw = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Sigma_bw).to(self.dev, non_blocking=True)

        # BW Fully connected 1
        self.d_input_FC1_bw = self.d_hidden_Sigma_bw # + self.d_hidden_Q
        self.d_output_FC1_bw = self.m * self.m
        self.d_hidden_FC1_bw = self.d_input_FC1_bw * out_mult
        self.FC1_bw = nn.Sequential(
                nn.Linear(self.d_input_FC1_bw, self.d_hidden_FC1_bw),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC1_bw, self.d_output_FC1_bw))

        torch.nn.init.constant_(self.FC1_bw[-1].weight, 0)
        torch.nn.init.constant_(self.FC1_bw[-1].bias, 0)

        # BW Fully connected 2
        self.d_input_FC2_bw = self.d_hidden_Sigma_bw + self.d_output_FC1_bw
        self.d_output_FC2_bw = self.d_hidden_Sigma_bw
        self.FC2_bw = nn.Sequential(
                nn.Linear(self.d_input_FC2_bw, self.d_output_FC2_bw),
                nn.ReLU())
        
        # BW Fully connected 3
        self.d_input_FC3_bw = self.m
        self.d_output_FC3_bw = self.m * in_mult
        self.FC3_bw = nn.Sequential(
                nn.Linear(self.d_input_FC3_bw, self.d_output_FC3_bw),
                nn.ReLU())

        # BW Fully connected 4
        self.d_input_FC4_bw = 2 * self.m
        self.d_output_FC4_bw = 2 * self.m * in_mult
        self.FC4_bw = nn.Sequential(
                nn.Linear(self.d_input_FC4_bw, self.d_output_FC4_bw),
                nn.ReLU())

    ####################################
    ### Initialize Backward Sequence ###
    ####################################
    def InitBackward(self, filter_x):
        self.s_m1x_nexttime = torch.squeeze(filter_x)

    ##############################
    ### Innovation Computation ###
    ##############################
    def S_Innovation(self, filter_x,t):
        self.filter_x_prior = self.f(filter_x,t).squeeze()
        self.dx = self.s_m1x_nexttime - self.filter_x_prior

    ################################
    ### Smoother Gain Estimation ###
    ################################
    def step_RTSGain_est(self, filter_x_nexttime, smoother_x_tplus2):

        # Reshape and Normalize Delta tilde x_t+1 = x_t+1|T - x_t+1|t+1
        dm1x_tilde = self.s_m1x_nexttime - filter_x_nexttime
        dm1x_tilde_reshape = torch.squeeze(dm1x_tilde)
        bw_innov_diff = func.normalize(dm1x_tilde_reshape, p=2, dim=0, eps=1e-12, out=None)
        
        if smoother_x_tplus2 is None:
            # Reshape and Normalize Delta x_t+1 = x_t+1|t+1 - x_t+1|t (for t = T-1)
            dm1x_input2 = filter_x_nexttime - self.filter_x_prior
            dm1x_input2_reshape = torch.squeeze(dm1x_input2)
            bw_evol_diff = func.normalize(dm1x_input2_reshape, p=2, dim=0, eps=1e-12, out=None)
        else:
            # Reshape and Normalize Delta x_t+1|T = x_t+2|T - x_t+1|T (for t = 1:T-2)
            dm1x_input2 = smoother_x_tplus2 - self.s_m1x_nexttime
            dm1x_input2_reshape = torch.squeeze(dm1x_input2)
            bw_evol_diff = func.normalize(dm1x_input2_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Feature 7:  x_t+1|T - x_t+1|t
        dm1x_f7 = self.s_m1x_nexttime - filter_x_nexttime
        dm1x_f7_reshape = torch.squeeze(dm1x_f7)
        bw_update_diff = func.normalize(dm1x_f7_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Smoother Gain Network Step
        SG = self.RTSGain_step(bw_innov_diff, bw_evol_diff, bw_update_diff)

        # Reshape Smoother Gain to a Matrix
        self.SGain = torch.reshape(SG, (-1,self.m, self.m))

    ####################
    ### RTS Net Step ###
    ####################
    def RTSNet_step(self,t, filter_x, filter_x_nexttime, smoother_x_tplus2):
        # filter_x = torch.squeeze(filter_x)
        # filter_x_nexttime = torch.squeeze(filter_x_nexttime)
        # smoother_x_tplus2 = torch.squeeze(smoother_x_tplus2)
        # Compute Innovation
        self.S_Innovation(filter_x,t)

        # Compute Smoother Gain
        self.step_RTSGain_est(filter_x_nexttime, smoother_x_tplus2)

        # Compute the 1-st posterior moment
        INOV = torch.bmm(self.SGain, self.dx.unsqueeze(-1)).squeeze()
        self.s_m1x_nexttime = filter_x + INOV

        # return
        return torch.squeeze(self.s_m1x_nexttime)

    ##########################
    ### Smoother Gain Step ###
    ##########################
    def RTSGain_step(self, bw_innov_diff, bw_evol_diff, bw_update_diff):

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1])
            expanded[0, :, :] = x
            return expanded

        bw_innov_diff = expand_dim(bw_innov_diff)
        bw_evol_diff = expand_dim(bw_evol_diff)
        bw_update_diff = expand_dim(bw_update_diff)
        
        ####################
        ### Forward Flow ###
        ####################
        
        # FC 3
        in_FC3 = bw_update_diff
        out_FC3 = self.FC3_bw(in_FC3)

        # Q-GRU
        in_Q = out_FC3
        out_Q, self.h_Q_bw = self.GRU_Q_bw(in_Q, self.h_Q_bw)

        # FC 4
        in_FC4 = torch.cat((bw_innov_diff, bw_evol_diff), 2)
        out_FC4 = self.FC4_bw(in_FC4)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC4), 2)
        out_Sigma, self.h_Sigma_bw = self.GRU_Sigma_bw(in_Sigma, self.h_Sigma_bw)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1_bw(in_FC1)

        #####################
        ### Backward Flow ###
        #####################

        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_FC1), 2)
        out_FC2 = self.FC2_bw(in_FC2)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma_bw = out_FC2

        return out_FC1

    ###############
    ### Forward ###
    ###############
    def forward(self, yt,t, filter_x, filter_x_nexttime, smoother_x_tplus2):
        if yt is None:
            return self.RTSNet_step(t,filter_x, filter_x_nexttime, smoother_x_tplus2)
        else:
            yt = yt.to(self.dev, non_blocking=True)
            return self.KNet_step(yt,t)


    def init_hidden(self,Batch_size):
        super(RTSNetNN, self).init_hidden(Batch_size)
        weight = next(self.parameters()).data
        hidden = weight.new(1, self.batch_size, self.d_hidden_Sigma_bw).zero_()

        self.h_Sigma_bw = hidden.data
        self.h_Sigma_bw[0,0,:] = self.prior_Sigma.flatten()

        hidden = weight.new(1, self.batch_size, self.d_hidden_Q_bw).zero_()
        self.h_Q_bw = hidden.data
        self.h_Q_bw[0,0,:] = self.prior_Q.flatten()

        
