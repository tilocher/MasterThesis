import time

import librosa.core
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
from utils import Stich

class EM_Pipeline(nn.Module):

    # Default Logs
    Logs = {'EM_Iter_Loss': '.npy',
            'Sample_Plots': '.pdf',
            'EM_Convergence': '.pdf',
            'KGain': '.npy',
            'Pipelines': '.pt',
            'Prior_Plot': '.pdf'}

    def __init__(self,PriorModel, Logger: LocalLogger, em_parameters:list = ('R','Q', 'Mu','Sigma'),
                 Mode:str = 'All', AdditionalLogs: dict = {}, smoothing_window_Q = -1, smoothing_window_R = -1):
        """
        PriorModel: A model
        """
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

        # The time window for which to smooth em vars
        self.smoothing_window_Q = smoothing_window_Q
        self.smoothing_window_R = smoothing_window_R

    def save(self):
        # Save the whole Pipeline as .pt file
        # pass
        torch.save(self, self.Logger.GetLocalSaveName('Pipelines'))


    def Run(self, Data: Dataset,PriorSamples:int = 100 , em_its:int = 10,
            Num_Plot_Samples:int = 10,ConvergenceThreshold:float = 1e-5, EM_rep = None, **kwargs):

        try:
            with torch.no_grad():

                def GetSubset(Data,SplitIndex):

                    LowerIndices = torch.arange(0,SplitIndex,step=1)
                    UpperIndices = torch.arange(SplitIndex,len(Data),step = 1)

                    LowerSet = torch.utils.data.Subset(Data,LowerIndices)
                    UpperSet = torch.utils.data.Subset(Data,UpperIndices)
                    return LowerSet,UpperSet


                PriorDataset, TestDataset = GetSubset(Data,PriorSamples)

                self.FitPrior(PriorDataset)

                self.PlotPrior()

                self.EM(TestDataset,em_its,Num_Plot_Samples, ConvergenceThreshold, EM_rep= EM_rep,**kwargs)

                self.save()

                # return self.comulativeLosses

        except:
            self.Logger.ForceClose()
            raise

    def FitPrior(self, PriorSet: Dataset):
        print('Fitting Prior')
        self.PriorLearnSetsize = len(PriorSet)
        try:
            observations,_ = next(iter(DataLoader(PriorSet,batch_size=len(PriorSet))))
            self.PriorModel.fit(observations.squeeze().mT)
            self.ssModel = self.PriorModel.getSysModel()
        except ValueError:
            observations = next(iter(DataLoader(PriorSet, batch_size=len(PriorSet))))
            self.PriorModel.fit(observations.squeeze().mT)
            self.ssModel = self.PriorModel.getSysModel()
        except NotImplementedError:

            self.ssModel = SystemModel(f = 'Identity', q = 1, h = 'Identity', r = 1,
                                       T = PriorSet.dataset.T,T_test= PriorSet.dataset.T,
                                       m= PriorSet.dataset.m, n= PriorSet.dataset.n)

        self.ssModel.InitSequence(torch.zeros((self.ssModel.m, 1)), torch.eye(self.ssModel.m))

        self.KalmanSmoother = KalmanSmoother(ssModel=self.ssModel, em_vars=self.em_parameters)
        self.KalmanSmoother.InitSequence()

        self.ssModel.GenerateSequence(self.ssModel.T)
        self.prior = self.ssModel.x

    def PlotPrior(self):

        self.ssModel.GenerateSequence(self.ssModel.T)


        # plt.plot(self.ssModel.x[0])
        # plt.show()



    def EM(self, Data: Dataset, em_its: int, Num_Plot_Samples: int, ConvergenceThreshold: float, EM_rep = None,**kwargs):


        if self.Mode == 'All':
            self.RegularEM(Data=Data, em_its=em_its,
                    ConvergenceThreshold=ConvergenceThreshold, Num_Plot_Samples= Num_Plot_Samples)
        elif self.Mode == 'Segmented':
            self.SegmentedEM(TestSet=Data, em_its=em_its, Num_Plot_Samples=Num_Plot_Samples,
                             ConvergenceThreshold=ConvergenceThreshold)
        elif self.Mode == 'Consecutive':
            self.ConsecutiveEM(TestSet=Data, em_its=em_its, Num_Plot_Samples=Num_Plot_Samples,
                               ConvergenceThreshold=ConvergenceThreshold, EM_rep= EM_rep,**kwargs)
        else:
            raise ValueError(f'Mode {self.Mode} not supported')



    def RegularEM(self, Data: Dataset, em_its:int , ConvergenceThreshold: float, Num_Plot_Samples: int):

        # self.EM_Data = Data

        self.Overlaps = Data.dataset.Overlap

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


        channel = 0
        observations, states = Data[-Num_Plot_Samples:]
        observations = observations[..., channel].reshape(Num_Plot_Samples, -1, 1)
        states = states[..., channel].reshape(Num_Plot_Samples, -1, 1)
        results = [self.KalmanSmoother.Smoothed_State_Means[-Num_Plot_Samples:, :, channel]]
        labels = ['Smoothed states within HB', 'Filtered State consecutive HB']

        self.PlotResults(observations, states, results, labels)




    def ConsecutiveEM(self, TestSet, em_its,  Num_Plot_Samples, ConvergenceThreshold, EM_rep ,**kwargs):

        self.Overlaps = TestSet.dataset.Overlap

        # self.EM_Data = TestSet

        channel = 0

        try:
            loader = TestSet.dataset
        except AttributeError:

            try:
                loader = TestSet
            except:
                raise

        DataSet_length = len(TestSet)
        nResiduals = kwargs['nResiduals'] if 'nResiduals' in kwargs.keys() else 1



        self.SmoothedResults = torch.empty((DataSet_length,self.ssModel.T,self.ssModel.m))
        self.SmoothedCovariances = torch.empty((DataSet_length,self.ssModel.T,self.ssModel.m,self.ssModel.m))
        self.FilteredResults = torch.empty((DataSet_length, self.ssModel.T, self.ssModel.m))
        self.FilteredCovariances = torch.empty((DataSet_length,self.ssModel.T,self.ssModel.m,self.ssModel.m))
        self.comulativeLosses = []

        TestDataset = DataLoader(TestSet, shuffle=False, batch_size=1)
        Num_Plot_Samples = min(Num_Plot_Samples, len(TestDataset))

        q_ini = np.random.random()

        self.ConsecutiveSSModel = SystemModel(f = 'Identity', q = q_ini, h = 'Identity', r = 1,
                                              T = DataSet_length,T_test= DataSet_length,
                                              m= loader.num_samples, n=loader.num_samples )


        self.ConsecutiveFilters = [KalmanFilter(self.ConsecutiveSSModel, DiagonalMatrices=['F','H','Q','R','Sigma'], nResiduals=nResiduals)
                                   for _ in range(self.ssModel.m)]

        self.SecondKalmanSmoother = KalmanSmoother(ssModel=self.ssModel, em_vars=self.em_parameters)
        self.SecondKalmanSmoother.InitSequence()
        # self.ConsecutiveFilter = KalmanFilter(self.ConsecutiveSSModel, DiagonalMatrices=['F','H','Q','R','Sigma'])

        Initial_r_2 = np.random.random()

        Initial_q_2 = np.random.random()

        loss_fn = torch.nn.MSELoss(reduction='mean')

        timerSmoother = []
        timerWhole = []

        self.Logger.SaveConfig({'EMSamples': DataSet_length,
                                'EM_Parameters': self.em_parameters,
                                'Initial_r_2': Initial_r_2,
                                'Initial_q_2': Initial_q_2,
                                'EM_Its': em_its,
                                'ConvergenceThreshold': ConvergenceThreshold,
                                'EM_reps': EM_rep})

        for n,ConsecutiveFilter in enumerate(self.ConsecutiveFilters):

            ConsecutiveFilter.InitSequence(self.prior[n,:].reshape(-1,1), torch.eye(ConsecutiveFilter.m))
            ConsecutiveFilter.InitOnline(DataSet_length)

        try:

            for n,(Heartbeat_Obs, Heartbeat_State) in enumerate(tqdm(TestDataset)):

                # if n >= len(TestSet) - 8:
                #
                #     start = 100
                #     end = 150
                #     factor = 1.5
                #     amplitudeFactor = 2
                #     amplitudeAddition = 0.4
                #
                #     length = (end-start)
                #     newStart = start - int(length*(factor-1)/2)
                #     newEnd = end + int(length*(factor-1)/2)+1
                #
                #     noise = Heartbeat_Obs[:,:,newStart:newEnd,0]  - Heartbeat_State[:,:,newStart:newEnd,0]
                #
                #     if amplitudeFactor >0:
                #         upsample = librosa.core.resample(Heartbeat_State[:,:,start:end,0].numpy(),length, length*factor)
                #         distortion = (np.where(upsample < 0))
                #         # distortion = amplitudeFactor
                #         upsample[distortion] *= (1/amplitudeFactor)
                #         upsample[not distortion] *= amplitudeFactor
                #         distortionState = upsample
                #     else:
                #         distortionState = librosa.core.resample(Heartbeat_State[:, :, start:end, 0].numpy(), length,
                #                                                 length * factor)  + amplitudeAddition
                #
                #
                #     Heartbeat_Obs[:,:,newStart:newEnd,0] = torch.from_numpy(distortionState) + noise
                #     Heartbeat_State[:, :, newStart:newEnd,0] = torch.from_numpy(distortionState)
                #
                #
                #
                #     TestSet.dataset.noisy_dataset[n] = Heartbeat_Obs
                #     TestSet.dataset.centerd_data[n] = Heartbeat_State


                if n == 0:

                    self.InitFilter(Heartbeat_Obs, Heartbeat_State)

                    self.PerformEM(em_its=em_its, Observations=Heartbeat_Obs.squeeze(), T = self.ssModel.T,
                                   q_2=Initial_q_2, r_2=Initial_r_2, states=Heartbeat_State.squeeze()
                                   , ConvergenceThreshold=ConvergenceThreshold)

                    obs = self.KalmanSmoother.Smoothed_State_Means[0]
                else:
                    timerStart = time.time_ns()

                    self.KalmanSmoother.smooth(Heartbeat_Obs.squeeze(), T = self.ssModel.T)

                    timeSmoother = time.time_ns()
                    obs = self.KalmanSmoother.Smoothed_State_Means
                    # self.KalmanSmoother.emOnline(Heartbeat_Obs.squeeze(),self.ssModel.T,
                    #                              self.smoothing_window_Q,self.smoothing_window_R)


                self.SmoothedResults[n] = obs.squeeze()


                for c, ConsecutiveFilter in enumerate(self.ConsecutiveFilters):

                    if n % EM_rep == 0:
                        obs = self.KalmanSmoother.Smoothed_State_Means[0, :, c]
                        var = self.KalmanSmoother.Smoothed_State_Covariances[0,:,c,c]
                    else:
                        obs = self.KalmanSmoother.Smoothed_State_Means[:, :, c]
                        var = self.KalmanSmoother.Smoothed_State_Covariances[0,:,c,c]



                    ConsecutiveFilter.UpdateR(torch.eye(ConsecutiveFilter.m) * var)#.mean())

                    ConsecutiveFilter.UpdateRik(obs)



                    ConsecutiveFilter.UpdateOnline(obs)


                    self.FilteredResults[n,:,c] = ConsecutiveFilter.Filtered_State_Mean.squeeze()

                if n % EM_rep != 0:
                    timerEnd = time.time_ns()
                    timerWhole.append(timerEnd-timerStart)
                    timerSmoother.append(timeSmoother-timerStart)

            self.FilterLoss_dB = 10*torch.log10(loss_fn(self.FilteredResults,TestSet[:][1].squeeze()) )#/ loss_fn(torch.zeros_like(TestSet[:][1].squeeze()),TestSet[:][1].squeeze()))
            self.SmootherLoss_dB = 10*torch.log10(loss_fn(self.SmoothedResults,TestSet[:][1].squeeze()))#  / loss_fn(torch.zeros_like(TestSet[:][1].squeeze()),TestSet[:][1].squeeze()))

            self.comulativeLosses.append(self.SmootherLoss_dB)
            self.comulativeLosses.append(self.FilterLoss_dB)

            observations, states = TestSet[-Num_Plot_Samples:]

            observations = observations[..., channel].reshape(Num_Plot_Samples, -1, 1)
            states = states[..., channel].reshape(Num_Plot_Samples, -1, 1)

            states = TestSet.dataset.centerd_data[len(TestSet)-Num_Plot_Samples:len(TestSet)][...,0].mT
            observations = TestSet.dataset.noisy_dataset[len(TestSet)-Num_Plot_Samples:len(TestSet)][...,0].mT


            results = [self.SmoothedResults[-Num_Plot_Samples:, :, channel],
                       self.FilteredResults[-Num_Plot_Samples:, :, channel]]
            labels = ['Smoothed states within HB', 'Filtered State consecutive HB']

            self.PlotResults(observations, states, results, labels)
            inputSNR = 10*torch.log10(loss_fn(TestSet[:][1], TestSet[:][0]))

            print('Input SNR: {}[dB]'.format(inputSNR.item()))

            print('Output SNR: {}[dB]'.format((inputSNR-self.FilterLoss_dB).item()))

            print('Filtered Loss: {}[dB]'.format(self.FilterLoss_dB))
            print('Smoothed Loss: {}[dB]'.format(self.SmootherLoss_dB))

            if self.wandb:
                wandb.log({'FilteredLoss': self.FilterLoss_dB,
                           'SmoothedLoss': self.SmootherLoss_dB,
                           'Input SNR':inputSNR})

            return

        except ValueError:

            for n, Heartbeat_Obs in enumerate(tqdm(TestDataset)):

                if n % EM_rep == 0:

                    self.InitFilter(Heartbeat_Obs, None)

                    self.PerformEM(em_its=em_its, Observations=Heartbeat_Obs.squeeze(), T=self.ssModel.T,
                                   q_2=Initial_q_2, r_2=Initial_r_2, states=None
                                   , ConvergenceThreshold=ConvergenceThreshold)

                    obs = self.KalmanSmoother.Smoothed_State_Means[0, :, channel]
                else:
                    timerStart = time.time_ns()
                    self.KalmanSmoother.smooth(Heartbeat_Obs.squeeze(), T=self.ssModel.T)
                    obs = self.KalmanSmoother.Smoothed_State_Means[:, :, channel]

                self.SmoothedResults[n] = obs

                for c, ConsecutiveFilter in enumerate(self.ConsecutiveFilters):

                    if n % EM_rep == 0:
                        obs = self.KalmanSmoother.Smoothed_State_Means[0,:,c]
                    else:
                        obs = self.KalmanSmoother.Smoothed_State_Means[:,:,c]


                    ConsecutiveFilter.UpdateR(
                        torch.diag_embed(self.KalmanSmoother.Smoothed_State_Covariances[0, :,  c, c]))
                    ConsecutiveFilter.UpdateOnline(obs)
                    ConsecutiveFilter.UpdateRik(obs)

                if n % EM_rep != 0:
                    timerEnd = time.time_ns()

            self.StichedConsectutivlyFiltered = Stich(torch.cat( [ConsecutiveFilter.Filtered_State_Means[0] for ConsecutiveFilter in self.ConsecutiveFilters],dim=-1), self.Overlaps[self.PriorLearnSetsize-1:])

            results = [self.SmoothedResults[-Num_Plot_Samples:, :, channel],
                       self.ConsecutiveFilters[channel].Filtered_State_Means[0, -Num_Plot_Samples:]]
            labels = ['Smoothed states within HB', 'Filtered State consecutive HB']

            observations = TestSet[-Num_Plot_Samples:]
            observations = observations[..., channel].reshape(Num_Plot_Samples, -1, 1)
            self.PlotResults(observations, None, results, labels)




    def InitFilter(self,Input,Target):
        self.KalmanSmoother.InitMean(Input[:, 0, 0].unsqueeze(-1))

    def PerformEM(self,em_its, Observations, T,q_2, r_2,states,ConvergenceThreshold):

        self.EM_losses = self.KalmanSmoother.em(num_itts=em_its, Observations=Observations, T=T,
                                            q_2=q_2, r_2=r_2, states=states
                                            , ConvergenceThreshold=ConvergenceThreshold,
                                                smoothing_window_Q = self.smoothing_window_Q, smoothing_window_R= self.smoothing_window_R)

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

        legendFontSize = 15
        ticksSize = 16
        titleSize = 16
        labelSize = 16

        multi_figures = [plt.subplots(nrows=nrows,ncols=ncols, figsize = (16,9),dpi=120) for i in range(max(int(np.ceil(samples/(ncols*nrows))),1))]
        for fig,_ in multi_figures:
            fig.set_tight_layout(True)
            fig.suptitle('Filtered Signal Samples')

        channel = 0

        distinguihable_color = ['#fff017','#5EF1F2','#0075DC', '#00998F','#000075','#911eb4']



        if states == None:
            stateFlag = False
            states = [None for _ in range(samples)]
        else:
            stateFlag = True

        for j,(observation, state) in enumerate(zip(observations, states)):

            fig_single, ax_single = plt.subplots(figsize = (16,9),dpi=120)
            figSingleNoWindow, axSingleNoWindow = plt.subplots(figsize = (16,9),dpi=120)

            fig_multi ,ax_multi = multi_figures[int(j/(nrows * ncols))]


            current_axes = ax_multi[int(j%(nrows*ncols)/nrows), j % ncols]
            current_axes.tick_params(labelsize = 8)
            current_axes.xaxis.set_tick_params(labelsize=ticksSize)
            current_axes.yaxis.set_tick_params(labelsize=ticksSize)



            if state != None:
                ax_single.plot(t,state[...,channel].squeeze(), label = 'Ground Truth', color = 'g')
                axSingleNoWindow.plot(t, state[..., channel].squeeze(), label='Ground Truth', color='g')
                current_axes.plot(t,state[...,channel].squeeze(), label = 'Ground Truth', color = 'g')

            ax_single.plot(t,observation[...,channel].squeeze(), label = 'Observation', color = 'r', alpha = 0.4)
            axSingleNoWindow.plot(t,observation[...,channel].squeeze(), label = 'Observation', color = 'r', alpha = 0.4)

            current_axes.plot(t,observation[...,channel].squeeze(), label = 'Observation', color = 'r', alpha = 0.4)

            for i,(result, label) in enumerate(zip(results, labels)):
                # color = (max(0, i - 1) * 0.5 ** (i - 2), max(0, i) * 0.5 ** (i-1), max(0, i+1) * 0.5 ** i)
                color = distinguihable_color[i]

                ax_single.plot(t,result[j].squeeze(), label= label, color = color)
                axSingleNoWindow.plot(t,result[j].squeeze(), label= label, color = color)

                current_axes.plot(t,result[j].squeeze(), label = label, color = color)

            ax_single.legend(fontsize=1.5*legendFontSize)
            axSingleNoWindow.legend(fontsize=1.5*legendFontSize)

            current_axes.legend(fontsize= 1.5*legendFontSize)

            ax_single.set_xlabel('Time Steps',fontsize = 1.5*labelSize)
            ax_single.xaxis.set_tick_params(labelsize=1.5*ticksSize)
            ax_single.yaxis.set_tick_params(labelsize=1.5*ticksSize)

            axSingleNoWindow.set_xlabel('Time Steps',fontsize = 1.5*labelSize)
            axSingleNoWindow.xaxis.set_tick_params(labelsize=1.5*ticksSize)
            axSingleNoWindow.yaxis.set_tick_params(labelsize=1.5*ticksSize)


            current_axes.set_xlabel('Time Steps',fontsize = 1.5*labelSize)

            ax_single.set_ylabel('Amplitude [mV]',fontsize = 1.5*labelSize)
            axSingleNoWindow.set_ylabel('Amplitude [mV]',fontsize = 1.5*labelSize)

            current_axes.set_ylabel('Amplitude [mV]',fontsize = 1.5*labelSize)

            # ax_single.set_title('Filtered Signal Sample',fontsize = 1.5*titleSize)
            axSingleNoWindow.set_title('Filtered Signal Sample',fontsize = 1.5*titleSize)




            if self.Zoom:

                axins = ax_single.inset_axes([0.05, 0.5, 0.4, 0.4])

                if state != None:
                    axins.plot(t,state,color = 'g')

                for i, (result, label) in enumerate(zip(results, labels)):
                    # color = (max(0, i - 1) * 0.5 ** (i - 2), max(0, i) * 0.5 ** (i - 1), max(0, i + 1) * 0.5 ** i)
                    color = distinguihable_color[i]

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
            figSingleNoWindow.savefig(self.Logger.GetLocalSaveName('Sample_Plots',prefix= f'{prefix}Single_No_Window_{j}_'))

            if self.wandb:
                pass
                # wandb.log({'chart': fig_single})
            else:
                fig_single.show()

        for n,(multi_fig,_) in enumerate(multi_figures):
            multi_fig.savefig(self.Logger.GetLocalSaveName('Sample_Plots',prefix= f'{prefix}Multi_{n}_'))

        if self.wandb:
            pass
            # wandb.log({'chart':fig_multi})
        else:
            fig_multi.show()

        # Plot multiple HBs
        consecutive_beats = min(10, samples)

        StackedObservations = Stich(observations[-consecutive_beats:], self.Overlaps[-consecutive_beats:])

        if stateFlag:
            StackedStates = Stich(states[-consecutive_beats:], self.Overlaps[-consecutive_beats:])
            stackedYMin = torch.min(StackedStates)
            stackedYMax = torch.max(StackedStates)
        else:
            stackedYMin = torch.inf
            stackedYMax = -torch.inf

        Stackedresults = []

        smallestResultYAxis = torch.inf
        biggestResultYAxis = -torch.inf
        for result in results:
            Stackedresults.append(Stich(result[-consecutive_beats:], self.Overlaps[-consecutive_beats:]))
            yStackedMinResults = torch.min(result)
            yStackedMaxResults = torch.max(result)
            if yStackedMinResults < smallestResultYAxis:
                smallestResultYAxis = yStackedMinResults
            if yStackedMaxResults > biggestResultYAxis:
                biggestResultYAxis = yStackedMaxResults

        t_cons = np.arange(start=0, stop=consecutive_beats, step= consecutive_beats / len(StackedObservations))
        yAxisMin = min(stackedYMin.item(), smallestResultYAxis.item())
        yAxisMax = max(stackedYMax.item(), biggestResultYAxis.item())



        num_signal = 2 if stateFlag else 1

        fig_con, ax_cons = plt.subplots(nrows= num_signal + len(Stackedresults), ncols= 1, figsize = (16,9), dpi = 120)
        fig_con.set_tight_layout(True)

        ax_cons[0].plot(t_cons,StackedObservations.squeeze(), label = 'Observations', color = 'r', alpha = 0.4)

        ax_cons[0].set_xlabel('Time [s]',fontsize = labelSize)
        ax_cons[0].set_ylabel('Amplitude [mV]',fontsize = labelSize)
        title_cons = 'Observations'
        ax_cons[0].set_title(title_cons,fontsize = titleSize)
        ax_cons[0].xaxis.set_tick_params(labelsize=ticksSize)
        ax_cons[0].yaxis.set_tick_params(labelsize=ticksSize)

        if stateFlag:
            ax_cons[1].plot(t_cons,StackedStates.squeeze(), label = 'Ground Truth', color = 'g')

            ax_cons[1].set_xlabel('Time [s]',fontsize = labelSize)
            ax_cons[1].set_ylabel('Amplitude [mV]',fontsize = labelSize)
            title_cons = 'Ground Truth'
            ax_cons[1].set_title(title_cons,fontsize = titleSize)
            ax_cons[1].xaxis.set_tick_params(labelsize=ticksSize)
            ax_cons[1].yaxis.set_tick_params(labelsize=ticksSize)
            ax_cons[1].set_ylim([yAxisMin, yAxisMax])

        for j,(result, label) in enumerate(zip(Stackedresults, labels)):
            # color = (max(0, j - 1) * 0.5 ** (j - 2), max(0, j) * 0.5 ** (j - 1), max(0, j + 1) * 0.5 ** j)
            color = distinguihable_color[j]
            ax_cons[j+num_signal].plot(t_cons,result.squeeze(), color = color)
            ax_cons[j+num_signal].set_title(label,fontsize = titleSize)
            ax_cons[j+num_signal].set_xlabel('Time [s]',fontsize = labelSize)
            ax_cons[j+num_signal].set_ylabel('Amplitude [mV]',fontsize = labelSize)
            ax_cons[j+num_signal].xaxis.set_tick_params(labelsize=ticksSize)
            ax_cons[j+num_signal].yaxis.set_tick_params(labelsize=ticksSize)
            ax_cons[j+num_signal].set_ylim([yAxisMin, yAxisMax])

        fig_con.savefig(self.Logger.GetLocalSaveName('Sample_Plots',prefix= f'{prefix}Cons_'))

        if self.wandb:
            pass
        else:
            fig_con.show()

        1











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
                 Mode:str = 'All', AdditionalLogs: dict = {}, smoothing_window_Q = -1, smoothing_window_R = -1):

        super(Taylor_Pipeline, self).__init__(PriorModel,Logger,em_parameters,
                                                     Mode,AdditionalLogs,
                                              smoothing_window_Q = smoothing_window_Q, smoothing_window_R = smoothing_window_R)

########################################################################################################################
########################################################################################################################
################################################ Identity Model Pipeline ################################################
########################################################################################################################
########################################################################################################################

class Identity_Pipeline(EM_Pipeline):

    def __init__(self,PriorModel, Logger: LocalLogger, em_parameters:list = ('R','Q', 'Mu','Sigma'),
                 Mode:str = 'All', AdditionalLogs: dict = {}):

        super(Identity_Pipeline, self).__init__(PriorModel,Logger,em_parameters,
                                                     Mode,AdditionalLogs)
########################################################################################################################
########################################################################################################################
################################################ Identity Model Pipeline ################################################
########################################################################################################################
########################################################################################################################

class Prior_Pipeline(EM_Pipeline):

    def __init__(self,PriorModel, Logger: LocalLogger, em_parameters:list = ('R','Q', 'Mu','Sigma'),
                 Mode:str = 'All', AdditionalLogs: dict = {}):

        super(Prior_Pipeline, self).__init__(PriorModel,Logger,em_parameters,
                                                     Mode,AdditionalLogs)

    def InitFilter(self,Input,Target):
        self.KalmanSmoother.InitMean(Target[:, 0, 0].unsqueeze(-1))

    def FitPrior(self, PriorSet: Dataset):
        print('Fitting Prior')
        self.PriorLearnSetsize = len(PriorSet)
        try:
            observations,_ = next(iter(DataLoader(PriorSet,batch_size=len(PriorSet))))
            self.PriorModel.fit(observations.squeeze().mT)
            self.ssModel = self.PriorModel.GetSysModel()
        except ValueError:
            observations = next(iter(DataLoader(PriorSet, batch_size=len(PriorSet))))
            self.PriorModel.fit(observations.squeeze().mT)
            self.ssModel = self.PriorModel.GetSysModel()
        except NotImplementedError:

            self.ssModel = SystemModel(f = 'Identity', q = 1, h = 'Identity', r = 1,
                                       T = PriorSet.dataset.T,T_test= PriorSet.dataset.T,
                                       m= PriorSet.dataset.m, n= PriorSet.dataset.n)

        self.ssModel.InitSequence(torch.zeros((self.ssModel.m, 1)), torch.eye(self.ssModel.m))

        self.KalmanSmoother = KalmanSmoother(ssModel=self.ssModel, em_vars=self.em_parameters)
        self.KalmanSmoother.OnlyPrior = True
        self.KalmanSmoother.InitSequence()

        self.ssModel.GenerateSequence(self.ssModel.T)
        self.prior = self.ssModel.x

########################################################################################################################
########################################################################################################################
################################################ Rik Model Pipeline ################################################
########################################################################################################################
########################################################################################################################


class Rik_Pipeline(EM_Pipeline):

    def __init__(self, PriorModel, Logger: LocalLogger, em_parameters: list = ('R', 'Q', 'Mu', 'Sigma'),
                 Mode: str = 'All', AdditionalLogs: dict = {}, smoothing_window_Q = -1, smoothing_window_R = -1):

        self.Logs = {'Pipelines':'.pt',
                     'Sample_Plots':'.pdf',
                     }

        super(Rik_Pipeline, self).__init__(PriorModel, Logger, em_parameters,
                                                   Mode, AdditionalLogs,smoothing_window_Q,smoothing_window_R)


    def FitPrior(self, PriorSet: Dataset):


        self.ObservationCovariace =  self.PriorModel.fit(PriorSet)



        # self.ssModel.InitSequence(torch.zeros((self.ssModel.m, 1)), torch.eye(self.ssModel.m))



    def Run(self, Data: Dataset,PriorSamples:int = 100 , em_its:int = 10,
            Num_Plot_Samples:int = 10,ConvergenceThreshold:float = 1e-5, EM_rep = None,
            nResiduals = 1):

        try:


            def GetSubset(Data,SplitIndex):

                LowerIndices = torch.arange(0,SplitIndex,step=1)
                UpperIndices = torch.arange(SplitIndex,len(Data),step = 1)

                LowerSet = torch.utils.data.Subset(Data,LowerIndices)
                UpperSet = torch.utils.data.Subset(Data,UpperIndices)
                return LowerSet,UpperSet


            PriorDataset, TestDataset = GetSubset(Data,PriorSamples)

            self.FitPrior(PriorDataset)

            self.FilterConsecutive(TestDataset, Num_Plot_Samples, nResiduals)

            self.save()

        except:
            self.Logger.ForceClose()
            raise


    def FilterConsecutive(self,TestDataset,Num_Plot_Samples = 10, nResiduals = 1):

        TestLoader = DataLoader(TestDataset, batch_size=1)


        T = TestDataset.dataset.T
        channels = TestDataset.dataset.channels
        Samples = len(TestLoader)
        q_init = np.random.random()

        self.FilteredStates = torch.empty(Samples,T,channels)


        self.ConsecutiveSSModels = [SystemModel('Identity',q_init, 'Identity', self.ObservationCovariace[i],
                                              len(TestDataset), len(TestDataset), T,T) for i in range(channels)]


        self.Overlaps = TestDataset.dataset.Overlap

        timer = []

        for ConsecutiveSSModel in self.ConsecutiveSSModels:
            ConsecutiveSSModel.InitSequence()

        self.ConsecutiveFilters = [KalmanFilter(ConsecutiveSSModel, DiagonalMatrices= ['F','H','R','Q','Sigma'],nResiduals=nResiduals) for ConsecutiveSSModel in self.ConsecutiveSSModels]

        for ConsecutiveFilter in self.ConsecutiveFilters:
            ConsecutiveFilter.InitSequence()
            ConsecutiveFilter.InitOnline(len(TestLoader))

        for j, (observation, state) in enumerate(TestLoader):

            timeStart = time.time_ns()

            for channel in range(channels):

                self.ConsecutiveFilters[channel].UpdateRik(observation[...,channel].reshape(-1,1))

                self.ConsecutiveFilters[channel].UpdateOnline(observation[...,channel].reshape(-1,1))

                self.FilteredStates[j,:,channel] = self.ConsecutiveFilters[channel].Filtered_State_Mean.squeeze()

            timerEnd = time.time_ns()

            timer.append(timerEnd-timeStart)

        loss_fn = torch.nn.MSELoss(reduction = 'mean')


        loss = loss_fn(TestDataset[:][1], self.FilteredStates)

        self.FilterLoss_dB =(
            loss_fn(self.FilteredStates, TestDataset[:][1].squeeze()) )


        print(f'Loss {10*torch.log10(self.FilterLoss_dB)}[dB]')




        observations, states = TestDataset[-Num_Plot_Samples:]
        observations = observations[...,0].reshape(Num_Plot_Samples,-1,1)
        states = states[...,0].reshape(Num_Plot_Samples,-1,1)
        results = [self.FilteredStates[-Num_Plot_Samples:,:,0]]
        labels = ['Filtered State consecutive HB', 'Smoothed State consecutive HB']

        self.PlotResults(observations,states,results,labels)

########################################################################################################################
########################################################################################################################
################################################ Synthetic Model Pipeline ################################################
########################################################################################################################
########################################################################################################################



class Synthetic_Pipeline(EM_Pipeline):

    def __init__(self, PriorModel, Logger: LocalLogger, em_parameters: list = ('R', 'Q', 'Mu', 'Sigma'),
                 Mode: str = 'All', AdditionalLogs: dict = {}):

        self.Logs = {'Pipeline':'.pt',
                     'Sample_Plots':'.pdf',
                     }

        super(Synthetic_Pipeline, self).__init__(PriorModel, Logger, em_parameters,
                                                   Mode, AdditionalLogs)

    def FitPrior(self, PriorSet: Dataset):

        self.ssModel = self.PriorModel
        ini = torch.zeros((self.ssModel.m, 1))
        ini[0] = -torch.pi
        self.ssModel.InitSequence(ini, torch.eye(self.ssModel.m))

        self.KalmanSmoother = KalmanSmoother(ssModel=self.ssModel, em_vars=self.em_parameters)
        self.KalmanSmoother.InitSequence()

        self.ssModel.GenerateSequence(self.ssModel.T)
        self.prior = self.ssModel.x

    def InitFilter(self, Input, Target):
        ini = torch.zeros((self.ssModel.m, 1))
        ini[0] = -torch.pi
        ini[1:] = Input[0,0,0].unsqueeze(-1)
        self.KalmanSmoother.InitMean(ini)


########################################################################################################################
########################################################################################################################
################################################ AE Model Pipeline ################################################
########################################################################################################################
########################################################################################################################

class ReducedSystem(SystemModel):

    def __init__(self,NN,r,h,q,T,T_test,m,n):

        self.NN = NN

        super(ReducedSystem, self).__init__(self.f,r,h,q,T,T_test,m,n)

        self.FJacSet = True
        self.HJacSet = True



    def prime(self, data):

        self.data = self.NN(data.to(self.NN.dev)).squeeze().cpu()

        self.data_derivative = torch.cat((torch.diff(self.data,dim=0),torch.zeros(1,self.data.shape[-1])))

    def f(self,x,t):
        return self.data[t].reshape(-1,1)

    def getFJacobian(self,x,t):

        dF = torch.zeros((self.m,self.m))

        for i in range(self.m):
            dF[i,i] = self.data_derivative[t,i]
        return dF



class AE_Pipeline(EM_Pipeline):

    def __init__(self,dummy,Logger, em_parameters,Mode, AdditionalLogs = {},
                 smoothing_window_Q=0,smoothing_window_R=0):

        from NNs.AutoEncoder import AutoEncoder
        from SystemModels.Extended_sysmdl import SystemModel

        # Rik -6
        # PriorModel = torch.load(r"E:\MasterThesis\log\runs\AutoEncoder\22_09_13___11_44\Logs\Models\Models.pt")

        # Rik 0
        PriorModel = torch.load(r"E:\MasterThesis\log\runs\AutoEncoder\22_10_05___22_16\Logs\Models\Models.pt")

        # PhysioNet
        # PriorModel = torch.load(r"E:\MasterThesis\log\runs\AutoEncoder\22_10_03___23_10\Logs\Models\Models.pt")


        super(AE_Pipeline, self).__init__(PriorModel, Logger, em_parameters,
                                                 Mode, AdditionalLogs)

    def NN_forward(self, data, t):

        inference = self.PriorModel(data)

        return inference[0,:,t,:]



    def FitPrior(self, PriorSet: Dataset):

        T = PriorSet.dataset.T
        m = PriorSet.dataset.channels
        n = PriorSet.dataset.channels



        self.ssModel = ReducedSystem(self.PriorModel, 0, 'Identity', 0, T, T ,m,n)
        self.ssModel.InitSequence(torch.zeros((self.ssModel.m, 1)), torch.eye(self.ssModel.m))

        self.KalmanSmoother = KalmanSmoother(ssModel=self.ssModel, em_vars=self.em_parameters)
        self.KalmanSmoother.InitSequence()

    def PlotPrior(self):
        pass

    def ConsecutiveEM(self, TestSet, em_its,  Num_Plot_Samples, ConvergenceThreshold, EM_rep,**kwargs ):

        self.Overlaps = TestSet.dataset.Overlap

        nResiduals = kwargs['nResiduals'] if 'nResiduals' in kwargs.keys() else 1


        # self.EM_Data = TestSet

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
        self.FilteredResults = torch.empty((DataSet_length, self.ssModel.T, self.ssModel.m))
        self.NNResults = torch.empty((DataSet_length, self.ssModel.T, self.ssModel.m))


        TestDataset = DataLoader(TestSet, shuffle=False, batch_size=1)

        self.ConsecutiveSSModel = SystemModel(f = 'Identity', q = 0.01, h = 'Identity', r = 1,
                                              T = DataSet_length,T_test= DataSet_length,
                                              m= loader.num_samples, n=loader.num_samples )


        self.ConsecutiveFilters = [KalmanFilter(self.ConsecutiveSSModel, DiagonalMatrices=['F','H','Q','R','Sigma'],
                                                nResiduals= nResiduals) for _ in range(self.ssModel.m)]

        # self.ConsecutiveFilter = KalmanFilter(self.ConsecutiveSSModel, DiagonalMatrices=['F','H','Q','R','Sigma'])

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

        for n,ConsecutiveFilter in enumerate(self.ConsecutiveFilters):

            ConsecutiveFilter.InitSequence(torch.zeros((self.ssModel.T, 1)), torch.eye(ConsecutiveFilter.m))
            ConsecutiveFilter.InitOnline(DataSet_length)

        timerInference = []
        timerKF = []


        for n,(Heartbeat_Obs, Heartbeat_State) in enumerate(tqdm(TestDataset)):

            # if n == len(TestSet) - 3:
            #     Heartbeat_Obs[:, :, 30:50] += 0.5
            #     Heartbeat_State[:, :, 30:50] += 0.5
            #

            with torch.no_grad():
                timerStartinferenece = time.time_ns()
                NN_inference = self.ssModel.NN(Heartbeat_Obs.to(self.ssModel.NN.dev)).squeeze().cpu()
                timerEndinference = time.time_ns()

                timerInference.append(timerEndinference-timerStartinferenece)

            self.NNResults[n] = NN_inference


            # if n % EM_rep == 0:
            #
            #     self.InitFilter(Heartbeat_Obs, Heartbeat_State)
            #
            #     self.KalmanSmoother.ssModel.prime(Heartbeat_Obs)
            #
            #     self.PerformEM(em_its=em_its, Observations=Heartbeat_Obs.squeeze(), T = self.ssModel.T,
            #                    q_2=Initial_q_2, r_2=Initial_r_2, states=Heartbeat_State.squeeze()
            #                    , ConvergenceThreshold=ConvergenceThreshold)
            #
            #     obs = self.KalmanSmoother.Smoothed_State_Means[0]
            # else:
            #     self.KalmanSmoother.ssModel.prime(Heartbeat_Obs)
            #
            #     self.KalmanSmoother.smooth(Heartbeat_Obs.squeeze(), T = self.ssModel.T)
            #     obs = self.KalmanSmoother.Smoothed_State_Means
            #
            # self.SmoothedResults[n] = obs.squeeze()



            for c, ConsecutiveFilter in enumerate(self.ConsecutiveFilters):



                # if n % EM_rep == 0:
                #     obs = self.KalmanSmoother.Smoothed_State_Means[0, :, c]
                # else:
                #     obs = self.KalmanSmoother.Smoothed_State_Means[:, :, c]

                obs = NN_inference[...,c].reshape(-1,1)

                # Y_minus_i = torch.cat((NN_inference[..., :c], NN_inference[..., c + 1:]), dim=-1).squeeze()
                # y = NN_inference[..., c].squeeze()
                #
                # gamma = torch.linalg.pinv(Y_minus_i.T.mm(Y_minus_i)).mm(Y_minus_i.T).mm(y.unsqueeze(-1))
                #
                # w = y.unsqueeze(-1) - Y_minus_i.mm(gamma)
                #
                # var = w.var()

                var = (obs - Heartbeat_State).var()

                # ConsecutiveFilter.UpdateR(
                #     torch.diag_embed(self.KalmanSmoother.Smoothed_State_Covariances[0, :, c, c]))
                ConsecutiveFilter.UpdateR(
                    torch.eye(ConsecutiveFilter.m) *var)#* self.KalmanSmoother.Smoothed_State_Covariances[0, :, c,
                                                     #c].mean())
                ConsecutiveFilter.UpdateRik(obs)

                # ConsecutiveFilter.UpdateR(torch.eye(ConsecutiveFilter.m) * loss_fn(Heartbeat_State[...,c].squeeze(),self.SmoothedResults[n][...,c]))
                ConsecutiveFilter.UpdateOnline(obs)

                self.FilteredResults[n,:,c] = ConsecutiveFilter.Filtered_State_Mean.squeeze()

            timerEndKf = time.time_ns()
            timerKF.append(timerEndKf-timerStartinferenece)



        self.FilterLoss_dB = 10*torch.log10(loss_fn(self.FilteredResults,TestSet[:][1].squeeze()))
        self.SmootherLoss_dB = 10*torch.log10(loss_fn(self.SmoothedResults,TestSet[:][1].squeeze()))
        self.AELoss_dB = 10 * torch.log10(loss_fn(self.NNResults, TestSet[:][1].squeeze()))

        observations, states = TestSet[-Num_Plot_Samples:]
        observations = observations[..., channel].reshape(Num_Plot_Samples, -1, 1)

        states = states[..., channel].reshape(Num_Plot_Samples, -1, 1)

        results = [self.NNResults[-Num_Plot_Samples:,:,channel],
                   self.ConsecutiveFilters[channel].Filtered_State_Means[0, -Num_Plot_Samples:]
                   ]
        labels = [ 'AE output','Filtered State consecutive HB' ]

        self.PlotResults(observations, states, results, labels)
        print('Filtered Loss: {}[dB]'.format(self.FilterLoss_dB))
        print('AE Loss: {}[dB]'.format(self.AELoss_dB))


        return

    ########################################################################################################################
    ########################################################################################################################
    ################################################ Arima Model Pipeline ################################################
    ########################################################################################################################
    ########################################################################################################################


class Arima_Pipeline(EM_Pipeline):
    pass

    # def __init__(self,PriorModel, Logger: LocalLogger, em_parameters:list = ('R','Q', 'Mu','Sigma'),
    #          Mode:str = 'All', AdditionalLogs: dict = {}, smoothing_window_Q = -1, smoothing_window_R = -1):
    #
    #
    #
    #
    #     super(Arima_Pipeline, self).__init__(PriorModel,Logger,em_parameters,Mode,AdditionalLogs,smoothing_window_Q,smoothing_window_R)





