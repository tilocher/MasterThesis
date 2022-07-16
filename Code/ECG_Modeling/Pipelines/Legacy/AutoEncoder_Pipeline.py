# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________

from Code.ECG_Modeling.Pipelines.Base_Pipeline import Pipeline
from Code.ECG_Modeling.NN.AutoEncoder import AutoEncoder


class ECG_AE_Pipeline(Pipeline):

    def __init__(self,**kwargs):

        super(ECG_AE_Pipeline, self).__init__('ECG_AutoEncoder',**kwargs)

    def setModel(self, model):

        hyperparameters = {'LatentSpace': model.latent_space_dim,
                           'Channels': model.num_channels,
                           'NumLayers': len(model.conv_filters)}

        self.HyperParameters.update(hyperparameters)

    def InitModel(self,**kwargs):
        pass

    def Run_Inference(self,input,target,**kwargs):

        model_output = self.model(input)

        return model_output, self.loss_fn(model_output,target)



if __name__ == '__main__':



    from Code.ECG_Modeling.DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH

    snr = 6

    ts = 360

    loader = PhyioNetLoader_MIT_NIH(1,1, 512 ,SNR_dB=snr,random_sample=False, gpu= True , plot_sample=False, desired_shape= (1,512,2))

    LATENT_SPACE = 25

    nnModel = AutoEncoder(num_channels= 2,
                        signal_length = 512,
                        conv_filters=(40, 20, 20, 20, 20, 40),
                        conv_kernels=((40,2), (40,2), (40,2), (40,2), (40,2), (40,2)),
                        conv_strides=((2,1), (2,1), (2,1), (2,1), (2,1), (2,1)),
                        conv_dilation = ((1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),
                        latent_space_dim = LATENT_SPACE)


    ECG_Pipeline = ECG_AE_Pipeline(hyperP= {'LatentSpace': LATENT_SPACE, 'NumChannels': 2})
    ECG_Pipeline.setModel(nnModel)
    ECG_Pipeline.setTrainingParams(weightDecay=1e-6)


    ECG_Pipeline.NNTrain(loader,epochs= 100)

    ECG_Pipeline.NNTest(loader)

