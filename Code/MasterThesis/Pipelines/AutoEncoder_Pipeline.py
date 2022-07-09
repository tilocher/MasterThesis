# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________

from Code.ECG_Modeling.Pipelines.Base_Pipeline_WB import Pipeline
from Code.ECG_Modeling.NN.AutoEncoder import AutoEncoder
import torch
import numpy as np
from matplotlib import pyplot as plt
import wandb

from Code.ECG_Modeling.Logger.WB_BaseLogger import WB_Logger


class ECG_AE_Pipeline(Pipeline):

    def __init__(self,debug = False, **kwargs):

        super(ECG_AE_Pipeline, self).__init__('ECG_AutoEncoder',debug = debug, **kwargs)





    def InitModel(self,**kwargs):
        pass

    def Run_Inference(self,input,target,**kwargs):

        model_output = self.model(input)

        return model_output, self.loss_fn(model_output,target)

    def PlotResults(self, test_input, test_target, predictions):

        num_samples = 5

        test_input = test_input.squeeze()
        test_target = test_target.squeeze()
        predictions = predictions.squeeze()

        np.random.seed(42)

        for i in range(num_samples):

            random_sample_index = np.random.randint(0,test_input.shape[0])
            random_channel = 0

            plt.plot(test_input[random_sample_index,:,random_channel].detach().cpu(), label = 'Observations', alpha = 0.4, color = 'r')
            plt.plot(test_target[random_sample_index,:,random_channel].detach().cpu(), label = 'Ground truth',color = 'g')
            plt.plot(predictions[random_sample_index,:,random_channel].detach().cpu(), label = 'Prediction', color = 'b')
            plt.legend()
            if not self.debug:
                wandb.log({'chart':plt})
            else:
                plt.show()

    def setModel(self, model):

        self.model = model
        parameters = {'LatentSpace':model.latent_space_dim,
                      'NumChannels': model.num_channels}

        self.UpdateHyperParameters(parameters)


if __name__ == '__main__':



    from Code.ECG_Modeling.DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH

    snr = 6

    ts = 360

    loader = PhyioNetLoader_MIT_NIH(2,1, 512 ,SNR_dB=snr,random_sample=False, gpu= True ,
                                    plot_sample=False, desired_shape= (1,512,2),roll=60)
    test_loader = PhyioNetLoader_MIT_NIH(2,1, 512 ,SNR_dB=snr,random_sample=False, gpu= True ,
                                         plot_sample=False, desired_shape= (1,512,2),roll=60)





    # LATENT_SPACE = wandb.config['LatentSpace']
    LATENT_SPACE = 50

    nnModel = AutoEncoder(num_channels= 2,
                        signal_length = 512,
                        conv_filters=(40, 20, 20, 20, 20, 40),
                        conv_kernels=((40,2), (40,2), (40,2), (40,2), (40,2), (40,2)),
                        conv_strides=((2,1), (2,1), (2,1), (2,1), (2,1), (2,1)),
                        conv_dilation = ((1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),
                        latent_space_dim = LATENT_SPACE)


    ECG_Pipeline = ECG_AE_Pipeline(hyperP= {'LatentSpace': LATENT_SPACE, 'NumChannels': 2}, debug= False)
    ECG_Pipeline.setModel(nnModel)
    ECG_Pipeline.setTrainingParams(weightDecay=1e-6)




    ECG_Pipeline.NNTrain(loader,epochs= 40)

    ECG_Pipeline.NNTest(test_loader)


