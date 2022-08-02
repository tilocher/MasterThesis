import torch

from Base_Pipeline import Pipeline

import numpy as np
from matplotlib import pyplot as plt
import wandb


class KNet_Pipeline(Pipeline):

    def __init__(self, Logger, unsupervised=False, gpu=True, **kwargs):

        super(KNet_Pipeline, self).__init__('KNet_Pipeline', Logger, gpu, **kwargs)

        self.unsupervised = unsupervised

    def InitTraining(self,Batch_size,**kwargs):

        pass

    def InitModel(self,Batch_size,**kwargs):

        self.model.batch_size = Batch_size
        self.model.init_hidden(Batch_size)

    def Run_Inference(self,input,target,**kwargs):


        T = input.shape[2]

        self.model.InitSequence(input[:,0,0,:].detach())


        model_output = torch.empty(self.model.batch_size,T,self.model.m,device=self.dev,requires_grad=False)
        y_pred = torch.empty(self.model.batch_size,T,self.model.n,device= self.dev, requires_grad= False)

        for t in range(T):

            model_output[:,t] = self.model(input[:,:, t].squeeze(), t)
            y_pred[:,t] = self.model.m1y.squeeze()

        if self.unsupervised:
            loss = self.loss_fn(y_pred, input.squeeze())
        else:
            loss = self.loss_fn(model_output, target.squeeze())

        return model_output, loss


    def PlotResults(self, test_input, test_target, predictions):

        num_samples = 5

        test_input = test_input.squeeze()
        test_target = test_target.squeeze()
        predictions = predictions.squeeze()

        np.random.seed(42)

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

            axins = ax.inset_axes([0.05, 0.5, 0.4, 0.4])

            axins.plot(test_target[random_sample_index, :, random_channel].detach().cpu(), color='g')
            axins.plot(predictions[random_sample_index, :, random_channel].detach().cpu(), color='b')
            axins.get_xaxis().set_visible(False)
            axins.get_yaxis().set_visible(False)

            x1, x2, y1, y2 = int(0.4 * 360), int(0.6 * 360), torch.min(
                torch.min(test_target[random_sample_index, :, random_channel].detach().cpu()),
                torch.min(predictions[random_sample_index, :, random_channel].detach().cpu())).item(), \
                             torch.max(torch.max(test_target[random_sample_index, :, random_channel].detach().cpu()),
                                       torch.max(
                                           predictions[random_sample_index, :, random_channel].detach().cpu())).item()
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticklabels([])
            axins.set_yticklabels([])
            axins.grid()

            ax.indicate_inset_zoom(axins, edgecolor="black")

            plt.savefig(self.Logger.GetLocalSaveName('SamplePlots', prefix=f'{i}_'))

            if self.wandb:
                wandb.log({'chart': plt})
                plt.clf()

            else:
                plt.show()
    def setModel(self, model):

        self.model = model
        # parameters = {'LatentSpace':model.latent_space_dim,
        #               'NumChannels': model.num_channels}
        #
        # self.Logger.SaveConfig(parameters)
