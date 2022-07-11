# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ["WANDB_DIR"] = os.path.abspath(r'C:\Users\Timur\Desktop\tmp')


from Code.MasterThesis.Pipelines.Base_Pipeline import Pipeline
from Code.MasterThesis.NNs.AutoEncoder import AutoEncoder
import torch
import numpy as np
from matplotlib import pyplot as plt
import wandb
from Code.MasterThesis.log.BaseLogger import Logger,LocalLogger

class ECG_AE_Pipeline(Pipeline):

    def __init__(self,wandb = False, **kwargs):

        super(ECG_AE_Pipeline, self).__init__('ECG_AutoEncoder',wandb = wandb, **kwargs)



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
            random_sample_index = np.random.randint(0,test_input.shape[0])

            random_channel = 0

            plt.plot(test_input[random_sample_index,:,random_channel].detach().cpu(), label = 'Observations', alpha = 0.4, color = 'r')
            plt.plot(test_target[random_sample_index,:,random_channel].detach().cpu(), label = 'Ground truth',color = 'g')
            plt.plot(predictions[random_sample_index,:,random_channel].detach().cpu(), label = 'Prediction', color = 'b')
            plt.legend()
            if self.wandb:
                wandb.log({'chart':plt})
            else:
                plt.show()

    def setModel(self, model):

        self.model = model
        parameters = {'LatentSpace':model.latent_space_dim,
                      'NumChannels': model.num_channels}

        self.UpdateHyperParameters(parameters)


if __name__ == '__main__':




    from Code.MasterThesis.DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH
    import yaml
    from yaml.loader import SafeLoader

    snr = 6

    ts = 360

    loader = PhyioNetLoader_MIT_NIH(2,1, 512 ,SNR_dB=snr,random_sample=False, gpu= True ,
                                    plot_sample=False, desired_shape= (1,512,2),roll=60)
    # test_loader = PhyioNetLoader_MIT_NIH(2,1, 512 ,SNR_dB=snr,random_sample=False, gpu= True ,
    #                                      plot_sample=False, desired_shape= (1,512,2),roll=0)


    N_train = int(0.8*len(loader))
    N_test = len(loader) - N_train
    dev = torch.device('cuda:0' if torch.cuda.is_available() and loader.gpu == 'gpu' else 'cpu')
    Train_Loader, Test_Loader = torch.utils.data.random_split(loader, [N_train, N_test],
                                                              generator=torch.Generator(device=dev))

    config = yaml.load(open( os.path.dirname(os.path.realpath(__file__)) + '\\..\\Configs\\ECG_AutoEncoder.yaml'),Loader= SafeLoader)

    WandBLogger = LocalLogger

    # config = WandBLogger.GetConfig()

    # LATENT_SPACE = wandb.config['LatentSpace']
    LATENT_SPACE = config['LatentSpace']

    nnModel = AutoEncoder(num_channels= 2,
                        signal_length = 512,
                        conv_filters=(40, 20, 20, 20, 20, 40),
                        conv_kernels=((40,2), (40,2), (40,2), (40,2), (40,2), (40,2)),
                        conv_strides=((2,1), (2,1), (2,1), (2,1), (2,1), (2,1)),
                        conv_dilation = ((1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),
                        latent_space_dim = LATENT_SPACE)


    ECG_Pipeline = ECG_AE_Pipeline(wandb= False)
    ECG_Pipeline.setModel(nnModel)
    ECG_Pipeline.setTrainingParams(weightDecay=config['L2'], n_Epochs = config['Epochs'], n_Batch = config['BatchSize'],
                                   learningRate = config['lr'], shuffle = True, split_ratio = 0.7)




    ECG_Pipeline.NNTrain(Train_Loader,epochs=50)

    ECG_Pipeline.NNTest(Test_Loader)


