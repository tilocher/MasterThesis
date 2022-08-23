from matplotlib import pyplot as plt

from log.BaseLogger import LocalLogger,WandbLogger
from torch import nn
from SystemModels.Extended_sysmdl import SystemModel
from torch.utils.data.dataloader import DataLoader, Dataset
import wandb
import numpy as np
import torch
from Filters.KalmanSmoother import KalmanSmoother


class EM_Pipeline(nn.Module):

    def __init__(self,PriorModel, Logger: LocalLogger, em_parameters:list = ('R','Q', 'Mu','Sigma'),
                 Mode:str = 'All', AdditionalLogs: dict = {}):
        super(EM_Pipeline, self).__init__()

        # logs for all pipeline instances
        self.Logs = {'EM_Iter_Loss':'.npy',
                'EM_Sample_Plot': '.pdf',
                'EM_Convergence':'.pdf',
                'KGain':'.npy',
                'Pipelines':'.pt',
                'Prior_Plot':'.pdf'}

        # Add pipeline specific logs
        self.Logs.update(AdditionalLogs)

        # Save Logger instance
        self.Logger = Logger

        # Add all local logs
        self.Logger.AddLocalLogs(self.Logs)

        self.PriorModel  = PriorModel

        self.wandb = isinstance(Logger, WandbLogger)

        # Parameter to perform EM on
        self.em_parameters = em_parameters

        # What type of Em should be performed
        self.Mode = Mode

        self.Zoom = True

    def save(self):
        torch.save(self, self.Logger.GetLocalSaveName('Pipelines'))


    def Run(self, Data: Dataset,PriorSamples:int = 100 , em_its:int = 10,
            Num_Plot_Samples:int = 10,ConvergenceThreshold:float = 1e-5):

        try:


            def GetSubset(Data,SplitIndex):

                LowerIndices = torch.arange(0,SplitIndex,step=1)
                UpperIndices = torch.arange(SplitIndex,len(Data),step = 1)

                LowerSet = torch.utils.data.Subset(Data,LowerIndices)
                UpperSet = torch.utils.data.Subset(Data,UpperIndices)
                return LowerSet,UpperSet


            PriorDataset, TestDataset = GetSubset(Data,PriorSamples)

            self.FitPrior(PriorDataset)

            self.PlotPrior()

            self.EM(TestDataset,em_its,Num_Plot_Samples, ConvergenceThreshold)

        except:
            self.Logger.ForceClose()
            raise

    def FitPrior(self, PriorSet: Dataset):
        print('Fitting Prior')

        try:
            observations,_ = next(iter(DataLoader(PriorSet,batch_size=len(PriorSet))))
            self.PriorModel.fit(observations.squeeze().mT)
            self.ssModel = self.PriorModel.GetSysModel()
        except NotImplementedError:
            def f(x,t):
                return x
            def h(x,t):
                return x
            self.ssModel = SystemModel(f = f, q = 1, h = h,
                                       T = PriorSet.T,T_test= PriorSet.T,
                                       m= PriorSet.m, n= PriorSet.n)

        self.ssModel.InitSequence(torch.zeros((self.ssModel.m, 1)), torch.eye(self.ssModel.m))

        self.KalmanSmoother = KalmanSmoother(ssModel=self.ssModel, em_vars=self.em_parameters)
        self.KalmanSmoother.InitSequence()

    def PlotPrior(self):

        self.ssModel.GenerateSequence(self.PriorModel.T)

    def EM(self, Data: Dataset, em_its: int, Num_Plot_Samples: int, ConvergenceThreshold: float):

        if self.Mode == 'All':
            self.RegularEM(Data=Data, em_its=em_its,
                    ConvergenceThreshold=ConvergenceThreshold)
        elif self.Mode == 'Segmented':
            self.SegmentedEM(TestSet=Data, em_its=em_its, Num_Plot_Samples=Num_Plot_Samples,
                             ConvergenceThreshold=ConvergenceThreshold)
        elif self.Mode == 'Consecutive':
            self.ConsecutiveEM(TestSet=Data, em_its=em_its, Num_Plot_Samples=Num_Plot_Samples,
                               ConvergenceThreshold=ConvergenceThreshold)

        else:
            raise ValueError(f'Mode {self.Mode} not supported')

        self.PlotEMResults(Data, Num_Plot_Samples=Num_Plot_Samples)


    def RegularEM(self, Data: Dataset, em_its:int , ConvergenceThreshold: float):

        self.EM_Data = Data

        DataSet_length = len(Data)

        self.Logger.SaveConfig({'EMSamples': DataSet_length,
                                'EM_Parameters': self.em_parameters})

        TestDataset = DataLoader(Data, shuffle=False, batch_size=DataSet_length)

        Test_Inputs, Test_Targets = next(iter(TestDataset))


        Initial_r_2 = np.random.random()

        Initial_q_2 = np.random.random()

        self.Logger.SaveConfig({'EMSamples': DataSet_length,
                                'EM_Parameters': self.em_parameters,
                                'Initial_r_2': Initial_r_2,
                                'Initial_q_2': Initial_q_2,
                                'EM_Its': em_its,
                                'ConvergenceThreshold':ConvergenceThreshold})

        self.InitFilter(Test_Inputs,Test_Targets)

        self.PerformEM(em_its=em_its, Observations=Test_Inputs.squeeze(), T=self.ssModel.T,
                                                q_2=Initial_q_2, r_2=Initial_r_2, states=Test_Targets.squeeze()
                                                , ConvergenceThreshold=ConvergenceThreshold)


        if self.EM_losses != None:
            np.save(self.Logger.GetLocalSaveName('EM_Iter_Loss'), self.EM_losses.numpy())
        np.save(self.Logger.GetLocalSaveName('KGain'), self.KalmanSmoother.Kalman_Gains.numpy())

        if self.wandb:
            wandb.log({'EM Iteration Losses': self.EM_losses})
            wandb.log({'Final Loss [dB]': self.EM_losses[-1]})

        self.save()

    def InitFilter(self,Input,Target):
        self.KalmanSmoother.InitMean(Input[:, 0, 0].unsqueeze(-1))

    def PerformEM(self,em_its, Observations, T,q_2, r_2,states,ConvergenceThreshold):

        self.EM_losses = self.KalmanSmoother.em(num_itts=em_its, Observations=Observations, T=T,
                                            q_2=q_2, r_2=r_2, states=states
                                            , ConvergenceThreshold=ConvergenceThreshold)

    def PlotEMResults(self,Data, Num_Plot_Samples = 10, prefix = '',Smoothed_states = None, label = 'Smoothed States'):
        self._PlotEMResults(Data,Num_Plot_Samples,prefix,Smoothed_states,label)

    def _PlotEMResults(self,Data, Num_Plot_Samples = 10, prefix = '',Smoothed_states = None, label = 'Smoothed States'):

        if not self.EM_losses == None:
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

            Observations,States = Data[i]


            index = i
            channel = 0

            fig, ax = plt.subplots(dpi=200)

            observation = Observations[...,channel].squeeze()
            state = States[...,channel].squeeze()

            if Smoothed_states == None:
                smoothed_states = self.KalmanSmoother.Smoothed_State_Means[index,:,channel,0]
            else:
                smoothed_states = Smoothed_states[index].squeeze()

            ax.plot(t,observation, label = 'Observations', alpha = 0.4, color  = 'r')

            ax.plot(t,state, label = 'Ground Truth', color = 'g')


            ax.plot(t,smoothed_states, label = label, color = 'b')


            ax.legend()

            ax.set_xlabel('Time Steps')

            ax.set_ylabel('Amplitude [mV]')

            ax.set_title('Sample of EM filtered Observations \n'
                      f'SNR: {self.EM_Data.dataset.dataset.SNR_dB} [dB], Em Iterations: {self.Logger.GetConfig()["EM_Its"]},'
                      f'Channel: {channel}')

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


class ConstantModel_Pipeline(EM_Pipeline):

    def __init__(self,PriorModel, Logger: LocalLogger, em_parameters:list = ('R','Q', 'Mu','Sigma'),
                 Mode:str = 'All', AdditionalLogs: dict = {}):

        super(ConstantModel_Pipeline, self).__init__(PriorModel,Logger,em_parameters,
                                                     Mode,AdditionalLogs)
    def PlotPrior(self):
        pass

    def InitFilter(self,Input,Target):
        self.KalmanSmoother.InitMean(torch.zeros(self.ssModel.m, 1))

    def PerformEM(self,em_its, Observations, T,q_2, r_2,states,ConvergenceThreshold):
        self.EM_losses = self.KalmanSmoother.em(num_itts=em_its, Observations=Observations, T=T,
                                                q_2=q_2, r_2=r_2, states=None
                                                , ConvergenceThreshold=ConvergenceThreshold)

    def PlotEMResults(self,Data, Num_Plot_Samples = 10, prefix = '',Smoothed_states = None, label = 'Smoothed States'):
        self._PlotEMResults(Data,Num_Plot_Samples,prefix,Smoothed_states = self.KalmanSmoother.Smoothed_State_Means[:,:,0])


class TaylorModel_Pipeline(EM_Pipeline):

    def __init__(self,PriorModel, Logger: LocalLogger, em_parameters:list = ('R','Q', 'Mu','Sigma'),
                 Mode:str = 'All', AdditionalLogs: dict = {}):

        super(TaylorModel_Pipeline, self).__init__(PriorModel,Logger,em_parameters,
                                                     Mode,AdditionalLogs)