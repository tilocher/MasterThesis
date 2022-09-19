# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import sys
import os

from Base_Pipeline import Pipeline
import torch
import numpy as np
from matplotlib import pyplot as plt
import wandb
from log.BaseLogger import LocalLogger, WandbLogger
from torch.utils.data.dataloader import DataLoader
from utils import Stich

class ECG_AE_Pipeline(Pipeline):

    def __init__(self, logger: LocalLogger, **kwargs):

        log = {'SamplePlots': '.pdf'}

        if 'Segmented' in kwargs.keys():
            self.Segmented = kwargs['Segmented']
            log.update({'SegmentPlot':'.pdf'})
        else:
            self.Segmented = False


        super(ECG_AE_Pipeline, self).__init__('ECG_AutoEncoder', logger, AdditionalLogs=log,
                                              **kwargs)


    def InitModel(self, BatchSize, **kwargs):
        pass

    def Run_Inference(self, input, target, **kwargs):

        if self.Segmented:
            return self.Run_Inference_Segmented(input, target, **kwargs)
        else:
            return self.Run_Inference_Unsegmented(input, target, **kwargs)

    def Run_Inference_Unsegmented(self, input, target, **kwargs):

        model_output = self.model(input)

        return model_output, self.loss_fn(model_output, target)

    def Run_Inference_Segmented(self, input, target, **kwargs):

        loss = 0.
        outputs = []

        segment = kwargs['segment']

        # for SegmentInput, SegmentTarget in zip(input, target):

        SegmentedOutput = self.model(input[segment])
        # if self.loss_fn.reduction == 'none':
        #     loss += self.loss_fn(SegmentedOutput, SegmentTarget).mean(-2)
        # else:
        loss += self.loss_fn(SegmentedOutput, target[segment])

        # outputs.append(SegmentedOutput)

        return SegmentedOutput, loss

    def PlotResults(self, observations, states,results,labels,prefix:str = '',**kwargs):

        """
            Plot filtered samples as well as the observation and the state
            observations: The observed signal with shape (samples, Time, channels)
            states: The ground truth signal with shape (samples, Time, channels)
            """

        samples, T, channels = observations.shape

        t = np.arange(start=0, stop=1, step=1 / T)

        nrows = 2
        ncols = 2

        multi_figures = [plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 9), dpi=120) for i in
                         range(int(np.ceil(samples / (ncols * nrows))))]
        for fig, _ in multi_figures:
            fig.set_tight_layout(True)
            fig.suptitle('Filtered Signal Samples')

        channel = 0

        if states == None:
            stateFlag = False
            states = [None for _ in range(samples)]
        else:
            stateFlag = True

        for j, (observation, state) in enumerate(zip(observations, states)):

            fig_single, ax_single = plt.subplots(figsize=(16, 9), dpi=120)

            fig_multi, ax_multi = multi_figures[int(j / (nrows * ncols))]

            current_axes = ax_multi[int(j % (nrows * ncols) / nrows), j % ncols]

            if state != None:
                ax_single.plot(t, state[..., channel].squeeze(), label='Ground Truth', color='g')
                current_axes.plot(t, state[..., channel].squeeze(), label='Ground Truth', color='g')

            ax_single.plot(t, observation[..., channel].squeeze(), label='Observation', color='r', alpha=0.4)
            current_axes.plot(t, observation[..., channel].squeeze(), label='Observation', color='r', alpha=0.4)

            for i, (result, label) in enumerate(zip(results, labels)):
                color = (max(0, i - 1) * 0.5 ** (i - 2), max(0, i) * 0.5 ** (i - 1), max(0, i + 1) * 0.5 ** i)

                ax_single.plot(t, result[j,...,channel].squeeze(), label=label, color=color)
                current_axes.plot(t, result[j,...,channel].squeeze(), label=label, color=color)

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
                    axins.plot(t, state, color='g')

                for i, (result, label) in enumerate(zip(results, labels)):
                    color = (max(0, i - 1) * 0.5 ** (i - 2), max(0, i) * 0.5 ** (i - 1), max(0, i + 1) * 0.5 ** i)

                    axins.plot(t, result[j,...,channel].squeeze(), label=label, color=color)

                axins.get_xaxis().set_visible(False)
                axins.get_yaxis().set_visible(False)

                x1, x2, y1, y2 = 0.4, 0.6, ax_single.dataLim.intervaly[0], ax_single.dataLim.intervaly[1]
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.set_xticklabels([])
                axins.set_yticklabels([])
                axins.grid()

                ax_single.indicate_inset_zoom(axins, edgecolor="black")

            fig_single.savefig(self.Logger.GetLocalSaveName('Sample_Plots', prefix=f'{prefix}Single_{j}_'))
            if self.wandb:
                wandb.log({'chart': fig_single})
            else:
                fig_single.show()

        for n, (multi_fig, _) in enumerate(multi_figures):
            multi_fig.savefig(self.Logger.GetLocalSaveName('Sample_Plots', prefix=f'{prefix}Multi_{n}_'))

        if self.wandb:
            wandb.log({'chart': fig_multi})
        else:
            fig_multi.show()

        # Plot multiple HBs

        consecutive_beats = min(7, samples)

        StackedObservations = Stich(observations[-consecutive_beats:], self.Overlaps[-consecutive_beats:])

        if stateFlag:
            StackedStates = Stich(states[-consecutive_beats:], self.Overlaps[-consecutive_beats:])

        Stackedresults = []
        for result in results:
            Stackedresults.append(Stich(result[-consecutive_beats:], self.Overlaps[-consecutive_beats:]))

        t_cons = np.arange(start=0, stop=consecutive_beats, step=consecutive_beats / len(StackedObservations))

        num_signal = 2 if stateFlag else 1

        fig_con, ax_cons = plt.subplots(nrows=num_signal + len(Stackedresults), ncols=1, figsize=(16, 9), dpi=120)
        fig_con.set_tight_layout(True)

        ax_cons[0].plot(t_cons, StackedObservations[...,channel].squeeze(), label='Observations', color='r', alpha=0.4)

        ax_cons[0].set_xlabel('Time [s]')
        ax_cons[0].set_ylabel('Amplitude [mV]')
        title_cons = 'Observations'
        ax_cons[0].set_title(title_cons)

        if stateFlag:
            ax_cons[1].plot(t_cons, StackedStates[...,channel].squeeze(), label='Ground Truth', color='g')

            ax_cons[1].set_xlabel('Time [s]')
            ax_cons[1].set_ylabel('Amplitude [mV]')
            title_cons = 'Ground Truth'
            ax_cons[1].set_title(title_cons)

        for j, (result, label) in enumerate(zip(Stackedresults, labels)):
            color = (max(0, j - 1) * 0.5 ** (j - 2), max(0, j) * 0.5 ** (j - 1), max(0, j + 1) * 0.5 ** j)
            ax_cons[j + num_signal].plot(t_cons, result[...,channel].squeeze(), color=color)
            ax_cons[j + num_signal].set_title(label)
            ax_cons[j + num_signal].set_xlabel('Time [s]')
            ax_cons[j + num_signal].set_ylabel('Amplitude [mV]')

        fig_con.savefig(self.Logger.GetLocalSaveName('Sample_Plots', prefix=f'{prefix}Cons_'))
        fig_con.show()
        1

    def setModel(self, model):

            self.model = model
            parameters = {'LatentSpace': model.latent_space_dim,
                          'NumChannels': model.num_channels}

            self.Logger.SaveConfig(parameters)

    def TestWhole(self, Test_Dataset, Networks,**kwargs):

        N_T = len(Test_Dataset)

        self.Logger.SaveConfig({'Test Set Size': N_T})

        self.MSE_test_linear_arr = torch.empty([N_T], requires_grad=False)

        if hasattr(self.loss_fn, 'reduction'):
            self.loss_fn.reduction = 'none'


        Test_Dataloader = DataLoader(Test_Dataset, shuffle=False, batch_size= N_T)

        num_segments = len(Networks)

        with torch.no_grad():

            StichedOutput = []
            StichedLoss = []

            for segment in range(num_segments):
                self.model = Networks[segment]

                self.InitModel(N_T,**kwargs)

                for test_input, test_target in Test_Dataloader:

                    Inference_out, test_loss = self.Run_Inference(test_input,test_target,segment = segment)

                    StichedOutput.append(Inference_out)
                    StichedLoss.append(test_loss)

            StichedLoss = torch.cat(StichedLoss,dim=2)
            StichedOutput = torch.cat(StichedOutput,dim=2)

            self.MSE_test_linear_arr = torch.mean(StichedLoss,dim=[n for n in range(1,test_loss.ndim)])

            self.PlotResults(torch.cat(test_input,dim=2),torch.cat(test_target,dim=2),StichedOutput)


            self.MSE_test_linear_avg = self.MSE_test_linear_arr.mean()
            self.MSE_test_dB_avg = 10*torch.log10(self.MSE_test_linear_avg)

            if self.wandb:
                wandb.log({'Test Loss [dB]': self.MSE_test_dB_avg.item()})


            print(f'Test Loss: {self.MSE_test_dB_avg} [dB]')

        self.save()




