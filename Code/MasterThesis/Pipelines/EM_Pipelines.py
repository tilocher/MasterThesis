from matplotlib import pyplot as plt
from tqdm import tqdm

from log.BaseLogger import LocalLogger,WandbLogger
from torch import nn
from SystemModels.Extended_sysmdl import SystemModel
from torch.utils.data.dataloader import DataLoader, Dataset
import wandb
import numpy as np
import torch
from Filters.KalmanSmoother import KalmanSmoother, KalmanFilter


class EM_Pipeline(nn.Module):

    # Default Logs
    Logs = {'EM_Iter_Loss': '.npy',
            'Sample_Plots': '.pdf',
            'EM_Convergence': '.pdf',
            'KGain': '.npy',
            'Pipelines': '.pt',
            'Prior_Plot': '.pdf'}

    def __init__(self,PriorModel, Logger: LocalLogger, em_parameters:list = ('R','Q', 'Mu','Sigma'),
                 Mode:str = 'All', AdditionalLogs: dict = {}):
        super(EM_Pipeline, self).__init__()


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

        # Flag if we want to zoom into a specific area
        self.Zoom = True

    def save(self):
        # Save the whole Pipeline as .pt file
        torch.save(self, self.Logger.GetLocalSaveName('Pipelines'))


    def Run(self, Data: Dataset,PriorSamples:int = 100 , em_its:int = 10,
            Num_Plot_Samples:int = 10,ConvergenceThreshold:float = 1e-5, EM_rep = None):

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

            self.EM(TestDataset,em_its,Num_Plot_Samples, ConvergenceThreshold, EM_rep= EM_rep)

            self.save()

        except:
            self.Logger.ForceClose()
            raise

    def FitPrior(self, PriorSet: Dataset):
        print('Fitting Prior')

        try:
            observations,_ = next(iter(DataLoader(PriorSet,batch_size=len(PriorSet))))
            self.PriorModel.fit(observations.squeeze().mT)
            self.ssModel = self.PriorModel.GetSysModel()
        except ValueError:
            observations = next(iter(DataLoader(PriorSet, batch_size=len(PriorSet))))
            self.PriorModel.fit(observations.squeeze().mT)
            self.ssModel = self.PriorModel.GetSysModel()
        except NotImplementedError:

            self.ssModel = SystemModel(f = 'Identity', q = 1, h = 'Identity',
                                       T = PriorSet.T,T_test= PriorSet.T,
                                       m= PriorSet.m, n= PriorSet.n)

        self.ssModel.InitSequence(torch.zeros((self.ssModel.m, 1)), torch.eye(self.ssModel.m))

        self.KalmanSmoother = KalmanSmoother(ssModel=self.ssModel, em_vars=self.em_parameters)
        self.KalmanSmoother.InitSequence()

        self.ssModel.GenerateSequence(self.ssModel.T)
        self.prior = self.ssModel.x

    def PlotPrior(self):

        self.ssModel.GenerateSequence(self.PriorModel.T)

    def EM(self, Data: Dataset, em_its: int, Num_Plot_Samples: int, ConvergenceThreshold: float, EM_rep = None):

        if self.Mode == 'All':
            self.RegularEM(Data=Data, em_its=em_its,
                    ConvergenceThreshold=ConvergenceThreshold, Num_Plot_Samples= Num_Plot_Samples)
        elif self.Mode == 'Segmented':
            self.SegmentedEM(TestSet=Data, em_its=em_its, Num_Plot_Samples=Num_Plot_Samples,
                             ConvergenceThreshold=ConvergenceThreshold)
        elif self.Mode == 'Consecutive':
            self.ConsecutiveEM(TestSet=Data, em_its=em_its, Num_Plot_Samples=Num_Plot_Samples,
                               ConvergenceThreshold=ConvergenceThreshold, EM_rep= EM_rep)
        else:
            raise ValueError(f'Mode {self.Mode} not supported')



    def RegularEM(self, Data: Dataset, em_its:int , ConvergenceThreshold: float, Num_Plot_Samples: int):

        self.EM_Data = Data

        DataSet_length = len(Data)


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
                                                q_2=Initial_q_2, r_2=Initial_r_2, states=Test_Targets
                                                , ConvergenceThreshold=ConvergenceThreshold)


        if self.EM_losses != None:
            np.save(self.Logger.GetLocalSaveName('EM_Iter_Loss'), self.EM_losses.numpy())
        np.save(self.Logger.GetLocalSaveName('KGain'), self.KalmanSmoother.Kalman_Gains.numpy())

        if self.wandb:
            wandb.log({'EM Iteration Losses': self.EM_losses})
            wandb.log({'Final Loss [dB]': self.EM_losses[-1]})



        self.PlotEMResults(Data, Num_Plot_Samples=Num_Plot_Samples)



    def ConsecutiveEM(self, TestSet, em_its,  Num_Plot_Samples, ConvergenceThreshold, EM_rep ):


        self.EM_Data = TestSet

        channel = 0

        try:
            loader = TestSet.dataset
        except AttributeError:

            try:
                loader = TestSet
            except:
                raise

        DataSet_length = len(TestSet)

        self.SmoothedResults = torch.empty((DataSet_length,self.ssModel.T,self.ssModel.m))


        TestDataset = DataLoader(TestSet, shuffle=False, batch_size=1)

        self.ConsecutiveSSModel = SystemModel(f = 'Identity', q = 0.01, h = 'Identity', r = 1,
                                              T = DataSet_length,T_test= DataSet_length,
                                              m= loader.num_samples, n=loader.num_samples )



        self.ConsecutiveFilter = KalmanFilter(self.ConsecutiveSSModel, DiagonalMatrices=['F','H','Q','R','Sigma'])

        Initial_r_2 = np.random.random()

        Initial_q_2 = np.random.random()

        SmootherLoss = 0.
        FilterLoss = 0.
        loss_fn = torch.nn.MSELoss(reduction='mean')

        self.Logger.SaveConfig({'EMSamples': DataSet_length,
                                'EM_Parameters': self.em_parameters,
                                'Initial_r_2': Initial_r_2,
                                'Initial_q_2': Initial_q_2,
                                'EM_Its': em_its,
                                'ConvergenceThreshold': ConvergenceThreshold,
                                'EM_reps': EM_rep})

        self.ConsecutiveFilter.InitSequence(self.prior[channel,:].reshape(-1,1), torch.eye(self.ConsecutiveFilter.m))
        self.ConsecutiveFilter.InitOnline(DataSet_length)

        try:

            for n,(Heartbeat_Obs, Heartbeat_State) in enumerate(tqdm(TestDataset)):

                if n % EM_rep == 0:

                    self.InitFilter(Heartbeat_Obs, Heartbeat_State)

                    self.PerformEM(em_its=em_its, Observations=Heartbeat_Obs.squeeze(), T = self.ssModel.T,
                                   q_2=Initial_q_2, r_2=Initial_r_2, states=Heartbeat_State.squeeze()
                                   , ConvergenceThreshold=ConvergenceThreshold)

                    obs = self.KalmanSmoother.Smoothed_State_Means[0,:,channel]
                else:

                    self.KalmanSmoother.smooth(Heartbeat_Obs.squeeze(), T = self.ssModel.T)
                    obs = self.KalmanSmoother.Smoothed_State_Means[:,:,channel]

                self.SmoothedResults[n] = obs

                SmootherLoss += loss_fn(obs.squeeze(),Heartbeat_State[...,channel].squeeze())

                self.ConsecutiveFilter.UpdateR(torch.diag_embed(self.KalmanSmoother.Smoothed_State_Covariances[0,:,channel,channel]))
                self.ConsecutiveFilter.UpdateOnline(obs)
                self.ConsecutiveFilter.UpdateRik(obs)

                FilterLoss += loss_fn(self.ConsecutiveFilter.Filtered_State_Mean.squeeze(), Heartbeat_State[...,channel].squeeze())

                FilterLoss_dB = 10 * torch.log10(FilterLoss / DataSet_length)
                SmootherLoss_dB = 10 * torch.log10(SmootherLoss / DataSet_length)

                observations, states = TestSet[-Num_Plot_Samples:]
                observations = observations[..., channel].reshape(Num_Plot_Samples, -1, 1)
                states = states[..., channel].reshape(Num_Plot_Samples, -1, 1)
                results = [self.SmoothedResults[-Num_Plot_Samples:, :, channel],
                           self.ConsecutiveFilter.Filtered_State_Means[0, -Num_Plot_Samples:]]
                labels = ['Smoothed states within HB', 'Filtered State consecutive HB']

                self.PlotResults(observations, states, results, labels)
                print('Filtered Loss: {}[dB]'.format(FilterLoss_dB))
                print('Smoothed Loss: {}[dB]'.format(SmootherLoss_dB))

        except ValueError:

            for n, Heartbeat_Obs in enumerate(tqdm(TestDataset)):

                if n % EM_rep == 0:

                    self.InitFilter(Heartbeat_Obs, None)

                    self.PerformEM(em_its=em_its, Observations=Heartbeat_Obs.squeeze(), T=self.ssModel.T,
                                   q_2=Initial_q_2, r_2=Initial_r_2, states=None
                                   , ConvergenceThreshold=ConvergenceThreshold)

                    obs = self.KalmanSmoother.Smoothed_State_Means[0, :, channel]
                else:

                    self.KalmanSmoother.smooth(Heartbeat_Obs.squeeze(), T=self.ssModel.T)
                    obs = self.KalmanSmoother.Smoothed_State_Means[:, :, channel]

                self.SmoothedResults[n] = obs


                self.ConsecutiveFilter.UpdateR(
                    torch.diag_embed(self.KalmanSmoother.Smoothed_State_Covariances[0, :, channel, channel]))
                self.ConsecutiveFilter.UpdateOnline(obs)
                self.ConsecutiveFilter.UpdateRik(obs)

            results = [self.SmoothedResults[-Num_Plot_Samples:, :, channel],
                       self.ConsecutiveFilter.Filtered_State_Means[0, -Num_Plot_Samples:]]
            labels = ['Smoothed states within HB', 'Filtered State consecutive HB']

            observations = TestSet[-Num_Plot_Samples:]
            observations = observations[..., channel].reshape(Num_Plot_Samples, -1, 1)
            self.PlotResults(observations, None, results, labels)




    def InitFilter(self,Input,Target):
        self.KalmanSmoother.InitMean(Input[:, 0, 0].unsqueeze(-1))

    def PerformEM(self,em_its, Observations, T,q_2, r_2,states,ConvergenceThreshold):

        self.EM_losses = self.KalmanSmoother.em(num_itts=em_its, Observations=Observations, T=T,
                                            q_2=q_2, r_2=r_2, states=states
                                            , ConvergenceThreshold=ConvergenceThreshold)

    def PlotResults(self,observations: torch.Tensor, states:torch.Tensor = None, results: list = None, labels: str or list = 'results',
                       prefix:str = ''):
        return self._PlotResults(observations,states,results,labels,prefix)

    def PlotEMconvergence(self,prefix):
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

    def _PlotResults(self,observations: torch.Tensor, states:torch.Tensor = None, results: torch.Tensor = None, labels: str or list = 'results',
                       prefix:str = ''):
        """
        Plot filtered samples as well as the observation and the state
        observations: The observed signal with shape (samples, Time, channels)
        states: The ground truth signal with shape (samples, Time, channels)
        """

        samples, T, channels = observations.shape

        t = np.arange(start=0, stop=1, step=1 / T)

        nrows = 2
        ncols = 2

        multi_figures = [plt.subplots(nrows=nrows,ncols=ncols, figsize = (16,9),dpi=120) for i in range(int(np.ceil(samples/(ncols*nrows))))]
        for fig,_ in multi_figures:
            fig.set_tight_layout(True)
            fig.suptitle('Filtered Signal Samples')

        channel = 0



        if states == None:
            stateFlag = False
            states = [None for _ in range(samples)]
        else:
            stateFlag = True

        for j,(observation, state) in enumerate(zip(observations, states)):

            fig_single, ax_single = plt.subplots(figsize = (16,9),dpi=120)

            fig_multi ,ax_multi = multi_figures[int(j/(nrows * ncols))]

            current_axes = ax_multi[int(j%(nrows*ncols)/nrows), j % ncols]


            if state != None:
                ax_single.plot(t,state[...,channel].squeeze(), label = 'Ground Truth', color = 'g')
                current_axes.plot(t,state[...,channel].squeeze(), label = 'Ground Truth', color = 'g')

            ax_single.plot(t,observation[...,channel].squeeze(), label = 'Observation', color = 'r', alpha = 0.4)
            current_axes.plot(t,observation[...,channel].squeeze(), label = 'Observation', color = 'r', alpha = 0.4)

            for i,(result, label) in enumerate(zip(results, labels)):
                color = (max(0, i - 1) * 0.5 ** (i - 2), max(0, i) * 0.5 ** (i-1), max(0, i+1) * 0.5 ** i)

                ax_single.plot(t,result[j].squeeze(), label= label, color = color)
                current_axes.plot(t,result[j].squeeze(), label = label, color = color)

            ax_single.legend()
            current_axes.legend()

            ax_single.set_xlabel('Time Steps')
            current_axes.set_xlabel('Time Steps')

            ax_single.set_ylabel('Amplitude [mV]')
            current_axes.set_ylabel('Amplitude [mV]')

            ax_single.set_title('Filtered Signal Sample')



            if self.Zoom:

                axins = ax_single.inset_axes([0.05, 0.5, 0.4, 0.4])

                if state != None:
                    axins.plot(t,state,color = 'g')

                for i, (result, label) in enumerate(zip(results, labels)):
                    color = (max(0, i - 1) * 0.5 ** (i - 2), max(0, i) * 0.5 ** (i - 1), max(0, i + 1) * 0.5 ** i)

                    axins.plot(t, result[j].squeeze(), label=label, color=color)

                axins.get_xaxis().set_visible(False)
                axins.get_yaxis().set_visible(False)

                x1, x2, y1, y2 = 0.4, 0.6, ax_single.dataLim.intervaly[0], ax_single.dataLim.intervaly[1]
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.set_xticklabels([])
                axins.set_yticklabels([])
                axins.grid()


                ax_single.indicate_inset_zoom(axins, edgecolor="black")

            fig_single.savefig(self.Logger.GetLocalSaveName('Sample_Plots',prefix= f'{prefix}Single_{j}_'))
            if self.wandb:
                wandb.log({'chart': fig_single})
            else:
                fig_single.show()

        for n,(multi_fig,_) in enumerate(multi_figures):
            multi_fig.savefig(self.Logger.GetLocalSaveName('Sample_Plots',prefix= f'{prefix}Multi_{n}_'))

        if self.wandb:
            wandb.log({'chart':fig_multi})
        else:
            fig_multi.show()

        # Plot multiple HBs

        consecutive_beats = min(7, samples)

        StackedObservations = torch.cat([observations[-beat] for beat in range(consecutive_beats)])
        if stateFlag:
            StackedStates = torch.cat([states[-beat] for beat in range(consecutive_beats)])

        t_cons = np.arange(start=0, stop=consecutive_beats, step= 1 / T)

        Stackedresults = []

        for result in results:
            Stackedresults.append(torch.cat([result[-beat] for beat in range(consecutive_beats)]))

        num_signal = 2 if stateFlag else 1

        fig_con, ax_cons = plt.subplots(nrows= num_signal + len(Stackedresults), ncols= 1, figsize = (16,9), dpi = 120)
        fig_con.set_tight_layout(True)

        ax_cons[0].plot(t_cons,StackedObservations.squeeze(), label = 'Observations', color = 'r', alpha = 0.4)

        ax_cons[0].set_xlabel('Time [s]')
        ax_cons[0].set_ylabel('Amplitude [mV]')
        title_cons = 'Observations'
        ax_cons[0].set_title(title_cons)

        if stateFlag:
            ax_cons[1].plot(t_cons,StackedStates.squeeze(), label = 'Ground Truth', color = 'g')

            ax_cons[1].set_xlabel('Time [s]')
            ax_cons[1].set_ylabel('Amplitude [mV]')
            title_cons = 'Ground Truth'
            ax_cons[1].set_title(title_cons)

        for j,(result, label) in enumerate(zip(Stackedresults, labels)):
            color = (max(0, j - 1) * 0.5 ** (j - 2), max(0, j) * 0.5 ** (j - 1), max(0, j + 1) * 0.5 ** j)
            ax_cons[j+num_signal].plot(t_cons,result.squeeze(), color = color)
            ax_cons[j+num_signal].set_title(label)
            ax_cons[j+num_signal].set_xlabel('Time [s]')
            ax_cons[j+num_signal].set_ylabel('Amplitude [mV]')

        fig_con.savefig(self.Logger.GetLocalSaveName('Sample_Plots',prefix= f'{prefix}Cons_'))
        fig_con.show()











########################################################################################################################
########################################################################################################################
################################################ Constant Model Pipeline ################################################
########################################################################################################################
########################################################################################################################


class Constant_Pipeline(EM_Pipeline):

    def __init__(self,PriorModel, Logger: LocalLogger, em_parameters:list = ('R','Q', 'Mu','Sigma'),
                 Mode:str = 'All', AdditionalLogs: dict = {}):

        super(Constant_Pipeline, self).__init__(PriorModel,Logger,em_parameters,
                                                     Mode,AdditionalLogs)
    def PlotPrior(self):
        pass

    def InitFilter(self,Input,Target):
        self.KalmanSmoother.InitMean(torch.zeros(self.ssModel.m, 1))

    def PerformEM(self,em_its, Observations, T,q_2, r_2,states,ConvergenceThreshold):
        self.EM_losses = self.KalmanSmoother.em(num_itts=em_its, Observations=Observations, T=T,
                                                q_2=q_2, r_2=r_2, states=None
                                                , ConvergenceThreshold=ConvergenceThreshold)

    def PlotEMResults(self,Data, Num_Plot_Samples = 10, prefix = '',Results = None, labels = 'Smoothed States'):
        self._PlotEMResults(Data,Num_Plot_Samples,prefix,
                            Results = [self.KalmanSmoother.Smoothed_State_Means[:,:,0]], labels = [labels])

########################################################################################################################
########################################################################################################################
################################################ Taylor Model Pipeline ################################################
########################################################################################################################
########################################################################################################################

class Taylor_Pipeline(EM_Pipeline):

    def __init__(self,PriorModel, Logger: LocalLogger, em_parameters:list = ('R','Q', 'Mu','Sigma'),
                 Mode:str = 'All', AdditionalLogs: dict = {}):

        super(Taylor_Pipeline, self).__init__(PriorModel,Logger,em_parameters,
                                                     Mode,AdditionalLogs)


########################################################################################################################
########################################################################################################################
################################################ Rik Model Pipeline ################################################
########################################################################################################################
########################################################################################################################


class Rik_Pipeline(EM_Pipeline):

    def __init__(self, PriorModel, Logger: LocalLogger, em_parameters: list = ('R', 'Q', 'Mu', 'Sigma'),
                 Mode: str = 'All', AdditionalLogs: dict = {}):

        self.Logs = {'Pipeline':'.pt',
                     'Sample_Plots':'.pdf',
                     }

        super(Rik_Pipeline, self).__init__(PriorModel, Logger, em_parameters,
                                                   Mode, AdditionalLogs)


    def FitPrior(self, PriorSet: Dataset):

        Data = DataLoader(PriorSet, batch_size= len(PriorSet))
        obs,states = next(iter(Data))

        self.ObservationCovariave = obs.std()


    def Run(self, Data: Dataset,PriorSamples:int = 100 , em_its:int = 10,
            Num_Plot_Samples:int = 10,ConvergenceThreshold:float = 1e-5, EM_rep = None):

        try:


            def GetSubset(Data,SplitIndex):

                LowerIndices = torch.arange(0,SplitIndex,step=1)
                UpperIndices = torch.arange(SplitIndex,len(Data),step = 1)

                LowerSet = torch.utils.data.Subset(Data,LowerIndices)
                UpperSet = torch.utils.data.Subset(Data,UpperIndices)
                return LowerSet,UpperSet


            PriorDataset, TestDataset = GetSubset(Data,PriorSamples)

            self.FitPrior(PriorDataset)

            self.FilterConsecutive(TestDataset, Num_Plot_Samples)

        except:
            self.Logger.ForceClose()
            raise


    def FilterConsecutive(self,TestDataset,Num_Plot_Samples = 10):

        T = TestDataset.dataset.T
        q_init = np.random.random()
        self.ConsecutiveSSModel = SystemModel('Identity',q_init, 'Identity', self.ObservationCovariave,
                                              len(TestDataset), len(TestDataset), T,T)

        self.ConsecutiveSSModel.InitSequence()

        self.ConsecutiveFilter = KalmanFilter(self.ConsecutiveSSModel, DiagonalMatrices= ['F','H','R','Q','Sigma'])
        self.ConsecutiveFilter.InitSequence()

        TestLoader = DataLoader(TestDataset, batch_size=1)

        self.ConsecutiveFilter.InitOnline(len(TestLoader))


        for j, (observation, state) in enumerate(TestLoader):

            self.ConsecutiveFilter.UpdateOnline(observation[...,0].reshape(-1,1))
            self.ConsecutiveFilter.UpdateRik(observation[...,0].reshape(-1,1))

            # plt.plot(observation[..., 0].squeeze(), label='Observation', color='r', alpha=0.4)
            # plt.plot(state[..., 0].squeeze(), label='State', color='g')
            # plt.plot(self.ConsecutiveFilter.Filtered_State_Mean.squeeze(), label='Filter', color='b')
            # plt.legend()
            # plt.show()



        loss_fn = torch.nn.MSELoss(reduction = 'mean')

        loss = loss_fn(TestDataset[:][1][...,0].mT, self.ConsecutiveFilter.Filtered_State_Means)

        print(f'Loss {10*torch.log10(loss)}[dB]')




        observations, states = TestDataset[-Num_Plot_Samples:]
        observations = observations[...,0].reshape(Num_Plot_Samples,-1,1)
        states = states[...,0].reshape(Num_Plot_Samples,-1,1)
        results = [self.ConsecutiveFilter.Filtered_State_Means[0,-Num_Plot_Samples:]]
        labels = ['Filtered State consecutive HB']

        self.PlotResults(observations,states,results,labels)








