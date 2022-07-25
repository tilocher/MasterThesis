from Base_Pipeline import Pipeline


class KNet_Pipeline(Pipeline):

    def __init__(self,Logger, **kwargs):

        super(KNet_Pipeline, self).__init__('ECG_AutoEncoder',Logger, **kwargs)



    def InitModel(self,**kwargs):
        pass

    def Run_Inference(self,input,target,**kwargs):

        self.model(input[:, 0, 0].squeeze(), 0)

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
            # random_sample_index = np.random.randint(0,test_input.shape[0])

            random_channel =  np.random.randint(0,2)

            plt.plot(test_input[random_sample_index,:,random_channel].detach().cpu(), label = 'Observations', alpha = 0.4, color = 'r')
            plt.plot(test_target[random_sample_index,:,random_channel].detach().cpu(), label = 'Ground truth',color = 'g')
            plt.plot(predictions[random_sample_index,:,random_channel].detach().cpu(), label = 'Prediction', color = 'b')
            plt.title(f'Test Sample from channel {random_channel}')
            plt.xlabel('Time Steps')
            plt.ylabel('Amplitude [mV]')
            plt.legend()
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
