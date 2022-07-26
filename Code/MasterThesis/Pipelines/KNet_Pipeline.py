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

            random_sample_index = np.random.randint(0,test_input.shape[0])
            # random_sample_index = np.random.randint(0,test_input.shape[0])

            random_channel =  np.random.randint(0,2)

            plt.plot(test_input[random_sample_index,:,random_channel].detach().cpu(), label = 'Observations', alpha = 0.4, color = 'r')
            plt.plot(test_target[random_sample_index,:,random_channel].detach().cpu(), label = 'Ground truth',color = 'g')
            plt.plot(predictions[random_sample_index,:,random_channel].detach().cpu(), label = 'Prediction', color = 'b')
            plt.title(f'Test Sample from channel {random_channel}')
            plt.xlabel('Time Steps')
            plt.ylabel('Amplitude [mV]')
            plt.legend()
            plt.savefig(self.Logger.GetLocalSaveName('SamplePlots',prefix=f'{i}_'))
            if self.wandb:
                wandb.log({'chart':plt})
            else:
                plt.show()

    def setModel(self, model):

        self.model = model
        # parameters = {'LatentSpace':model.latent_space_dim,
        #               'NumChannels': model.num_channels}
        #
        # self.Logger.SaveConfig(parameters)