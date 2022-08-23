# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
from datetime import datetime as dt
import os

import numpy as np
import torch
import tqdm
import wandb
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data.dataloader import DataLoader, Dataset
from log.BaseLogger import WandbLogger,LocalLogger
from SystemModels.Taylor_model import Taylor_model
from Filters.KalmanSmoother import KalmanSmoother
from DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH



class EM_Pipeline(nn.Module):

    def __init__(self,PriorModel, Logger: LocalLogger, em_parameters = ['R','Q', 'Mu','Sigma'], Fit = 'Taylor',
                 Mode = 'All'):
        super(EM_Pipeline, self).__init__()
        self.Logs = {'EM_Iter_Loss':'.npy',
                'EM_Sample_Plot': '.pdf',
                'EM_Convergence':'.pdf',
                'KGain':'.npy',
                'Pipelines':'.pt',
                'Prior_Plot':'.pdf'}

        self.Logger = Logger

        self.Logger.AddLocalLogs(self.Logs)

        self.PriorModel  = PriorModel

        self.wandb = isinstance(Logger, WandbLogger)

        self.em_parameters = em_parameters

        self.Fit = Fit

        self.Mode = Mode

        self.Zoom = True

    def save(self):
        torch.save(self, self.Logger.GetLocalSaveName('Pipelines'))



    def PlotPrior(self,prefix = ''):

        self.ssModel.GenerateSequence(self.ssModel.T)

        sample = self.ssModel.x.T[:,0]

        t = np.arange(start=0, stop=1, step=1 /self.ssModel.T)
        fig, ax = plt.subplots(dpi=200)

        ax.plot(t,sample, label = 'Learned Prior', color = 'g')
        ax.grid()
        ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude [mV]')
        ax.set_title('Prior learned by windowed Taylor Approximation \n'
                     'Window size: {}, Window type: {}'.format(self.PriorModel.window_size,self.PriorModel.window))

        axins = ax.inset_axes([0.05, 0.5, 0.4, 0.4])
        axins.plot(t, sample, color='g')
        axins.get_xaxis().set_visible(False)
        axins.get_yaxis().set_visible(False)

        x1, x2, y1, y2 = 0.4, 0.6, torch.min(sample).item(), torch.max(sample).item()
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axins.grid()

        ax.indicate_inset_zoom(axins, edgecolor="black")

        plt.savefig(self.Logger.GetLocalSaveName('Prior_Plot',prefix=prefix))
        plt.show()


    def TestEM(self, TestSet: PhyioNetLoader_MIT_NIH, em_its = 10, Num_Plot_Samples = 10, ConvergenceThreshold = 1e-5):

        if self.Fit == 'Prior':
            em_its = 1

        if self.Mode == 'All':
            self.EM(TestSet = TestSet, em_its= em_its, Num_Plot_Samples= Num_Plot_Samples,ConvergenceThreshold= ConvergenceThreshold)
        elif self.Mode == 'Segmented':
            self.SegmentedEM(TestSet = TestSet, em_its= em_its, Num_Plot_Samples= Num_Plot_Samples,ConvergenceThreshold= ConvergenceThreshold)
        elif self.Mode == 'Consecutive':
            self.ConsecutiveEM(TestSet = TestSet, em_its= em_its, Num_Plot_Samples= Num_Plot_Samples,ConvergenceThreshold= ConvergenceThreshold)

        else:
            raise ValueError(f'Mode {self.Mode} not supported')






    def EM(self, TestSet, em_its = 10, Num_Plot_Samples = 10,ConvergenceThreshold= 1e-5):

        self.TestSet = TestSet

        DataSet_length = len(TestSet)

        self.Logger.SaveConfig({'TestSamples': DataSet_length,
                                'EM_Parameters': self.em_parameters})


        TestDataset = DataLoader(TestSet, shuffle=False, batch_size=DataSet_length)

        Test_Inputs, Test_Targets = next(iter(TestDataset))

        np.random.seed(42)

        Initial_r_2 = np.random.random()

        Initial_q_2 = np.random.random()

        self.Logger.SaveConfig({'TestSamples': DataSet_length,
                                'EM_Parameters': self.em_parameters,
                                'Initial_r_2': Initial_r_2,
                                'Initial_q_2': Initial_q_2,
                                'EM_Its': em_its})

        self.KalmanSmoother.InitMean(Test_Inputs[:,0,0].unsqueeze(-1))

        if self.Fit == 'Prior':
            self.KalmanSmoother.InitMean(Test_Targets[:, 0, 0].unsqueeze(-1))

        self.EM_losses = self.KalmanSmoother.em(num_itts= em_its, Observations= Test_Inputs.squeeze(), T = self.ssModel.T,
                               q_2= Initial_q_2, r_2= Initial_r_2, states= Test_Targets.squeeze()
                                                ,ConvergenceThreshold= ConvergenceThreshold)

        np.save(self.Logger.GetLocalSaveName('EM_Iter_Loss'),self.EM_losses.numpy())
        np.save(self.Logger.GetLocalSaveName('KGain'), self.KalmanSmoother.Kalman_Gains.numpy())

        if self.wandb:
            wandb.log({'EM Iteration Losses': self.EM_losses})
            wandb.log({'Final Loss [dB]': self.EM_losses[-1]})

        self.PlotEMResults(Test_Inputs,Test_Targets, Num_Plot_Samples= Num_Plot_Samples)

        self.save()


    def PlotEMResults(self,Observations, States, Num_Plot_Samples = 10, prefix = '',Smoothed_states = None):


        plt.plot(self.EM_losses, '*', color = 'g', label = 'EM Iteration Loss')
        plt.grid()
        plt.legend()
        plt.title('EM MSE Convergence')
        plt.xlabel('Iteration Step')
        plt.ylabel('MSE Loss [dB]')
        plt.savefig(self.Logger.GetLocalSaveName('EM_Convergence',prefix=prefix))
        if self.wandb:
            wandb.log({'chart':plt})
        else:
            plt.show()

        t = np.arange(start=0, stop=1, step=1 / self.ssModel.T)

        for i in range(Num_Plot_Samples):


            index = i
            channel = 0

            fig, ax = plt.subplots(dpi=200)

            observation = Observations.squeeze()[index,:,channel]
            state = States.squeeze()[index,:,channel]

            if Smoothed_states == None:
                smoothed_states = self.KalmanSmoother.Smoothed_State_Means[index,:,channel,0]
            else:
                smoothed_states = Smoothed_states[index,:,channel,0]

            ax.plot(t,observation, label = 'Observations', alpha = 0.4, color  = 'r')

            ax.plot(t,state, label = 'Ground Truth', color = 'g')

            if self.Fit == 'Prior':
                ax.plot(t,smoothed_states, label = 'Fitted prior', color = 'b')
            else:
                ax.plot(t,smoothed_states, label = 'EM Smoothed States', color = 'b')

            ax.legend()

            ax.set_xlabel('Time Steps')

            ax.set_ylabel('Amplitude [mV]')

            fit = 'Identity' if  self.Fit == 'Identity' else 'Taylor Model'

            ax.set_title('Sample of EM filtered Observations \n'
                      f'SNR: {self.TestSet.dataset.SNR_dB} [dB], Em Iterations: {self.Logger.GetConfig()["EM_Its"]},'
                      f'Channel: {channel}, Evolution Model: {self.Fit}')

            if self.Zoom:

                axins = ax.inset_axes([0.05, 0.5, 0.4, 0.4])


                axins.plot(t,state,color = 'g')
                axins.plot(t, smoothed_states, color='b')
                axins.get_xaxis().set_visible(False)
                axins.get_yaxis().set_visible(False)

                x1, x2, y1, y2 = 0.4, 0.6, torch.min(torch.min(state),torch.min(smoothed_states)).item(), \
                                 torch.max(torch.max(state),torch.max(smoothed_states)).item()
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.set_xticklabels([])
                axins.set_yticklabels([])
                axins.grid()


                ax.indicate_inset_zoom(axins, edgecolor="black")

            plt.savefig(self.Logger.GetLocalSaveName('EM_Sample_Plot',prefix= prefix+f'{i}_'))

            if self.wandb:
                wandb.log({'chart':plt})
            else:
                plt.show()


    def SegmentedEM(self,TestSet: PhyioNetLoader_MIT_NIH, em_its: int  = 10, Num_Plot_Samples: int = 10,ConvergenceThreshold= 1e-5):
        """
        Function to Test EM on Segmented heartbeats
        :params:
        TestLoader: A Dataset loader type object to sample the observations and states from
        em_its: Number of EM iterations
        Num_Plot_Samples: Number of plots to create from filtered samples
        """

        # Make the Test set a class variable for later access when saved
        self.TestSet = TestSet

        # Get the Unsegmented data, for comparison
        UnsegmentedData = DataLoader(TestSet, shuffle=False, batch_size=len(TestSet))
        UnsegmentedInput, UnsegmentedTarget = next(iter(UnsegmentedData))

        # Size of the data set
        DataSet_length = len(TestSet)

        # Call function to segment data
        TestSet.dataset.SplitToSegments()

        # Get the segmented data
        TestDataset = DataLoader(TestSet, shuffle=False, batch_size=DataSet_length)
        Test_Inputs, Test_Targets = next(iter(TestDataset))

        # Initialize EM covariances
        Initial_r_2 = (np.random.random() + 1e-9)**2 # Guarantee >0
        Initial_q_2 = (np.random.random() + 1e-9)**2 # Guarantee >0

        # Log the testing parameters
        self.Logger.SaveConfig({'TestSamples': DataSet_length,
                                'EM_Parameters': self.em_parameters,
                                'Initial_r_2': Initial_r_2,
                                'Initial_q_2': Initial_q_2,
                                'EM_Its': em_its})

        # Initialized Smoothed segments
        smoothed_segments = []

        self.Zoom = False

        for j,(SegmentInput, SegmentTarget, segment) in enumerate(zip(Test_Inputs,Test_Targets,TestSet.dataset.segments)):

            # Update ssModel for the specific segment
            self.ssModel = self.PriorModel.GetSysModel(timesteps=SegmentTarget.shape[-2]
                                                       ,offset=segment, channels= SegmentTarget.shape[-1])

            # Initialize priors
            self.ssModel.InitSequence(torch.zeros((self.ssModel.m, 1)), torch.eye(self.ssModel.m))

            # Initialize Kalman Smoother
            self.KalmanSmoother.UpdateSSModel(self.ssModel)
            self.KalmanSmoother.ResetQR()

            if len(smoothed_segments) > 0 :
                self.KalmanSmoother.InitSequence(smoothed_segments[-1][:,-1])
            else:
                self.KalmanSmoother.InitSequence(SegmentInput[:,0,0].unsqueeze(-1))

            if self.Fit == 'Prior':
                self.KalmanSmoother.InitSequence(SegmentTarget[:,0,0].unsqueeze(-1))


            # Plot and save the obtained prior
            self.PlotPrior(prefix = f'Segment_{j}_')

            # Run EM
            self.EM_losses = self.KalmanSmoother.em(num_itts=em_its, Observations=SegmentInput.squeeze(),
                                                    T=self.ssModel.T,
                                                    q_2=Initial_q_2, r_2=Initial_r_2, states=SegmentTarget.squeeze(),
                                                    ConvergenceThreshold=ConvergenceThreshold)

            # Save results
            np.save(self.Logger.GetLocalSaveName('EM_Iter_Loss',prefix=f'Segment_{j}_'), self.EM_losses.numpy())
            np.save(self.Logger.GetLocalSaveName('KGain',prefix=f'Segment_{j}_'), self.KalmanSmoother.Kalman_Gains.numpy())

            smoothed_segments.append(self.KalmanSmoother.Smoothed_State_Means)


            if self.wandb:
                wandb.log({'EM Iteration Losses Segment {}'.format(j): self.EM_losses})
                wandb.log({'Final Loss [dB] Segment {}'.format(j): self.EM_losses[-1]})

            self.PlotEMResults(SegmentInput, SegmentTarget, Num_Plot_Samples=Num_Plot_Samples, prefix=f'Segment_{j}_')



        self.StitchedSegments = torch.cat(smoothed_segments,dim = 1)

        loss_fn = torch.nn.MSELoss(reduction='mean')

        self.StitchedLoss = 10*torch.log10(loss_fn(self.StitchedSegments.squeeze(),
                                                   UnsegmentedTarget.squeeze())).item()
        self.ssModel.T = UnsegmentedInput.shape[2]

        self.Zoom = True
        self.PlotEMResults(UnsegmentedInput, UnsegmentedTarget, Num_Plot_Samples=Num_Plot_Samples, prefix=f'Stitched_',
                           Smoothed_states=self.StitchedSegments)

        print(f'Stitched Loss [dB]:  {self.StitchedLoss}')

        self.save()

    def ConsecutiveEM(self,TestSet: PhyioNetLoader_MIT_NIH, em_its: int  = 10, Num_Plot_Samples: int = 10, ConvergenceThreshold= 1e-5):
        """
        Function to Test EM on consecutive heartbeats
        :params:
        TestLoader: A Dataset loader type object to sample the observations and states from
        em_its: Number of EM iterations
        Num_Plot_Samples: Number of plots to create from filtered samples
        """

        # Make the Test set a class variable for later access when saved
        self.TestSet = TestSet

        # Size of the data set
        DataSet_length = len(TestSet)

        # Get the first hearbeat
        TestDataset = DataLoader(TestSet, shuffle=False, batch_size=1)

        # Initialize EM covariances
        Initial_r_2 = (np.random.random() + 1e-9)**2 # Guarantee >0
        Initial_q_2 = (np.random.random() + 1e-9)**2 # Guarantee >0

        # initialize RTSSmoother
        self.KalmanSmoother = KalmanSmoother(self.ssModel, em_vars= self.em_parameters)
        self.KalmanSmoother.InitSequence()

        # Run EM on the first heartbeat
        FirstHB_obs, FirstHB_state = next(iter(TestDataset))
        self.KalmanSmoother.InitSequence(FirstHB_obs[:,0,0].unsqueeze(-1))
        self.EM_losses = self.KalmanSmoother.em(num_itts=1, Observations=FirstHB_obs.squeeze(), states= FirstHB_state.squeeze(),
                               q_2=Initial_q_2, r_2= Initial_r_2, T = self.ssModel.T, ConvergenceThreshold=ConvergenceThreshold)

        for Observation, State in TestDataset:
            self.ssModel.InitSequence(torch.zeros((self.ssModel.m, 1)), torch.eye(self.ssModel.m))


            self.PriorModel.UpdateWeights(Observation)

            self.ssModel = self.PriorModel.GetSysModel(self.ssModel.m, timesteps=self.ssModel.T)

        self.ssModel.InitSequence(torch.zeros((self.ssModel.m,1)), torch.eye(self.ssModel.m))
        self.PlotPrior()

# Additional specific pipelines
class EM_Taylor_Pipeline(EM_Pipeline):

    def __init__(self,PriorModel, Logger: LocalLogger, em_parameters = ['R','Q', 'Mu','Sigma'], Fit = 'Taylor',
                 Mode = 'All'):

        super(EM_Taylor_Pipeline, self).__init__(PriorModel,Logger,em_parameters,Fit,Mode)

        self.HyperParams = {'Window': PriorModel.window,
                            'WindowSize': PriorModel.window_size,
                            'WindowParameter': PriorModel.window_parameters,
                            'TaylorOrder': PriorModel.taylor_order

                            }

        self.Logger.SaveConfig(self.HyperParams)

    def TrainPrior(self, TrainLoader):

        try:
            self._TrainPrior(TrainLoader)
            self.PlotPrior()

        except:
            self.Logger.ForceClose()
            raise

    def _TrainPrior(self, TrainLoader):

        DataSet_length = len(TrainLoader)

        self.Logger.SaveConfig({'TrainSamples': DataSet_length})

        TrainDataset = DataLoader(TrainLoader, shuffle=False, batch_size=DataSet_length)

        train_inputs, _ = next(iter(TrainDataset))

        if not self.Fit == 'Identity':

            self.PriorModel.fit(train_inputs.squeeze().mT)

        else:

            self.PriorModel.f = self.PriorModel.Identity

        self.ssModel = self.PriorModel.GetSysModel(train_inputs.shape[-1], timesteps=train_inputs.shape[-2])

        self.ssModel.InitSequence(torch.zeros((self.ssModel.m, 1)), torch.eye(self.ssModel.m))

        self.KalmanSmoother = KalmanSmoother(ssModel=self.ssModel, em_vars=self.em_parameters)
        self.KalmanSmoother.InitSequence()

        if self.Fit == 'Prior':
            self.KalmanSmoother.SetOnlyPrior()

        self.PlotPrior()

















