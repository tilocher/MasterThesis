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

    def PlotResults(self, test_input, test_target, predictions,**kwargs):

        # if self.Segmented:
        #     for seg,(inp,tar,pred) in enumerate(zip(test_input,test_target,predictions)):
        #
        #         inp  = inp.squeeze()
        #         tar = tar.squeeze()
        #         pred = pred.squeeze()
        #
        #         random_sample_index = 0
        #         random_channel = 0
        #
        #         plt.plot(inp[random_sample_index, :, random_channel].detach().cpu(), label='Observations',
        #                  alpha=0.4,
        #                  color='r')
        #         plt.plot(tar[random_sample_index, :, random_channel].detach().cpu(), label='Ground truth',
        #                  color='g')
        #         plt.plot(pred[random_sample_index, :, random_channel].detach().cpu(), label='Prediction',
        #                  color='b')
        #         plt.title(f'Test Sample from channel {random_channel}')
        #         plt.xlabel('Time Steps')
        #         plt.ylabel('Amplitude [mV]')
        #         plt.legend()
        #         plt.savefig(self.Logger.GetLocalSaveName('SegmentPlot', prefix=f'{seg}_'))
        #
        #         if self.wandb:
        #             wandb.log({'chart': plt})
        #             plt.clf()
        #
        #         else:
        #             plt.show()
        #
        #
        #     test_input = torch.cat(test_input, -2)
        #     test_target = torch.cat(test_target, -2)
        #     predictions = torch.cat(predictions,-2)

        num_samples = 5

        if 'segment' in kwargs.keys():
            segment = kwargs['segment']
            test_input = test_input[segment]
            test_target = test_target[segment]

        test_input = test_input.squeeze()
        test_target = test_target.squeeze()
        predictions = predictions.squeeze()

        for i in range(num_samples):

            # random_sample_index = np.random.randint(0,test_input.shape[0])
            # random_sample_index = np.random.randint(0,test_input.shape[0])

            # random_channel =  np.random.randint(0,2)

            fig, ax = plt.subplots(dpi=200)

            random_sample_index = i
            random_channel = 0

            ax.plot(test_input[random_sample_index, :, random_channel].detach().cpu(), label='Observations', alpha=0.4,
                     color='r')
            ax.plot(test_target[random_sample_index, :, random_channel].detach().cpu(), label='Ground truth',
                     color='g')
            ax.plot(predictions[random_sample_index, :, random_channel].detach().cpu(), label='Prediction', color='b')
            ax.set_title(f'Test Sample from channel {random_channel}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Amplitude [mV]')
            plt.legend()
            if not self.Segmented:

                axins = ax.inset_axes([0.05, 0.5, 0.4, 0.4])

                axins.plot(test_target[random_sample_index, :, random_channel].detach().cpu(), color='g')
                axins.plot(predictions[random_sample_index, :, random_channel].detach().cpu(), color='b')
                axins.get_xaxis().set_visible(False)
                axins.get_yaxis().set_visible(False)

                x1, x2, y1, y2 = int(0.4*360), int(0.6*360), torch.min(torch.min(test_target[random_sample_index, :, random_channel].detach().cpu()), torch.min(predictions[random_sample_index, :, random_channel].detach().cpu())).item(), \
                                 torch.max(torch.max(test_target[random_sample_index, :, random_channel].detach().cpu()), torch.max(predictions[random_sample_index, :, random_channel].detach().cpu())).item()
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.set_xticklabels([])
                axins.set_yticklabels([])
                axins.grid()

                ax.indicate_inset_zoom(axins, edgecolor="black")
            segment = 'segment_' + kwargs['segment'] if 'segment' in kwargs.keys() else ''
            plt.savefig(self.Logger.GetLocalSaveName('SamplePlots', prefix=f'{i}_{segment}'))

            if self.wandb:
                wandb.log({'chart': plt})
                plt.clf()

            else:
                plt.show()

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




