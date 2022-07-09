# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
from datetime import datetime as dt
import os

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from Code.ECG_Modeling.Logger.TensorBoard_BaseLogger import TensorBoard_BaseLogger
from tqdm import trange
import numpy as np

class Pipeline(TensorBoard_BaseLogger):

    def __init__(self, modelName, dev = 'gpu'):

        Logs = {'Models':'.pt', 'Pipelines':'.pt'}

        super(Pipeline, self).__init__(modelName,Logs=Logs)

        if torch.cuda.is_available() and dev == 'gpu':
            self.dev = torch.device("cuda:0")
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            print("using GPU!")
        else:
            self.dev = torch.device("cpu")
            print("using CPU!")

        self.Base_folder  = os.path.dirname(os.path.realpath(__file__))
        self.modelName = modelName
        self.Time = dt.now()

    def UpdateHyperParameters(self, HyperP):

        if hasattr(self,'HyperParameters'):
            self.HyperParameters.update(HyperP)
        else:
            self.HyperParameters = HyperP

    def save(self):
        torch.save(self, self.getSaveName('Pipelines'))

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model


    def setTrainingParams(self, n_Epochs = 100, n_Batch = 32, learningRate = 1e-3,
                          weightDecay = 1e-6, shuffle = True, split_ratio = 0.7, loss_fn = nn.MSELoss(reduction='mean')):

        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch # Number of Samples in Batch
        self.learningRate = learningRate # Learning Rate
        self.weightDecay = weightDecay # L2 Weight Regularization - Weight Decay
        self.shuffle = shuffle # If we want to shuffle the data for each batch
        self.Train_CV_split_ratio = split_ratio

        # MSE LOSS Function
        self.loss_fn = loss_fn

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

        HyperP = {'lr': self.learningRate,
                 'BatchSize': self.N_B,
                 'Shuffle': str(self.shuffle),
                 'Train\CV Split Ratio ': self.Train_CV_split_ratio,
                 'Loss Function': str(self.loss_fn),
                 'L2': self.weightDecay,
                 'Optimizer': 'ADAM'}


        self.UpdateHyperParameters(HyperP)



    def InitModel(self,**kwargs):
        raise NotImplementedError('Methode needs to be implemented outside of base-class')

    def Run_Inference(self,input,target,**kwargs):
        raise NotImplementedError('Methode needs to be implemented outside of base-class')

    def NNTrain(self, DataSet, epochs=None, **kwargs):

        start_time = dt.now()

        print(f'---Started training of {self.modelName}---')

        try:
            self._NNTrain(DataSet,epochs,**kwargs)
            self.writer.add_hparams(self.HyperParameters
               , {'hparam/CV Loss [dB]':self.MSE_cv_dB_epoch[self.MSE_cv_idx_opt]},
                                    run_name= self.getSaveName('run'))
        except:
            self.ForceClose()
            raise

        end_time = dt.now()

        training_time = end_time-start_time

        print(f'---Training of {self.modelName} finished---')

        print(f'Training took: {training_time.seconds//3600} hours,'
              f' {(training_time.seconds // 60) % 60 } minutes, {training_time.seconds % 60} seconds')


    def _NNTrain(self, DataSet, epochs=None, **kwargs):


        DataSet_length = len(DataSet)

        self.HyperParameters.update({'Dataset Samples':DataSet_length})

        N_train = int(self.Train_CV_split_ratio * DataSet_length)
        N_CV = DataSet_length - N_train



        self.MSE_cv_linear_epoch = torch.empty([self.N_Epochs], requires_grad=False).to(self.dev, non_blocking=True)
        self.MSE_cv_dB_epoch = torch.empty([self.N_Epochs], requires_grad=False).to(self.dev, non_blocking=True)

        self.MSE_train_linear_epoch = torch.empty([self.N_Epochs], requires_grad=False).to(self.dev, non_blocking=True)
        self.MSE_train_dB_epoch = torch.empty([self.N_Epochs], requires_grad=False).to(self.dev, non_blocking=True)

        ##############
        ### Epochs ###
        ##############

        MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        if epochs is None:
            N = self.N_Epochs
        else:
            N = epochs

        self.HyperParameters.update({'Epochs': N})

        sample_input, sample_target = DataSet[np.random.randint(0, len(DataSet))]

        self.writer.add_graph(self.model, sample_input.unsqueeze(0))

        Epoch_itter = trange(N)

        Train_Dataset, CV_Dataset = torch.utils.data.random_split(DataSet, [N_train, N_CV],
                                                                  generator=torch.Generator(device=self.dev))

        for ti in Epoch_itter:


            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()

            CV_Dataloader = DataLoader(CV_Dataset, shuffle=False, batch_size=N_CV)


            self.InitModel(**kwargs)


            for cv_input, cv_target in CV_Dataloader:

                Inference_out, cv_loss = self.Run_Inference(cv_input, cv_target, **kwargs)


                self.MSE_train_linear_epoch[ti] = cv_loss.detach()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(cv_loss).detach()

                Epoch_cv_loss_lin = cv_loss.item()

            if (self.MSE_cv_dB_epoch[ti] < MSE_cv_dB_opt):

                MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti

                torch.save(self.model, self.getSaveName('Models'))

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            Train_DataLoader = DataLoader(Train_Dataset, batch_size=self.N_B, shuffle=self.shuffle,
                                          generator=torch.Generator(device=self.dev))


            self.InitModel(**kwargs)


            MSE_train_linear_batch = torch.empty(Train_DataLoader.__len__(), device=self.dev, requires_grad=False)

            for j, (train_input, train_target) in enumerate(Train_DataLoader):


                Inference_out, train_loss = self.Run_Inference(train_input, train_target, **kwargs)


                MSE_train_linear_batch[j] = train_loss

                self.optimizer.zero_grad()

                train_loss.backward(retain_graph=False)

                self.optimizer.step()

            # Average
            self.MSE_train_linear_epoch[ti] = MSE_train_linear_batch.mean().detach()
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(MSE_train_linear_batch.mean()).detach()

            Epoch_train_loss_lin = self.MSE_train_linear_epoch[ti].item()

            Epoch_cv_loss_dB = 10 * np.log10(Epoch_cv_loss_lin)
            Epoch_train_loss_dB = 10 * np.log10(Epoch_train_loss_lin)

            self.writer.add_scalar('Training Loss', Epoch_train_loss_lin, ti)
            self.writer.add_scalar('CV Loss', Epoch_cv_loss_lin, ti)

            self.writer.add_scalar('Training Loss [dB]', Epoch_train_loss_dB, ti)
            self.writer.add_scalar('CV Loss [dB]', Epoch_cv_loss_dB, ti)

            # Update Description
            train_desc = str(round(Epoch_train_loss_dB, 4))
            cv_desc = str(round(Epoch_cv_loss_dB, 4))
            Epoch_itter.set_description(
                'Epoch training Loss: {} [dB], Epoch Val. Loss: {} [dB], Best Val. Loss: {} [dB]'.format(train_desc, cv_desc,
                                                                                                         self.MSE_cv_dB_epoch[self.MSE_cv_idx_opt]))


        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]


    def NNTest(self, DataSet, **kwargs):


        start_time = dt.now()

        print(f'---Started testing of {self.modelName}---')

        try:
            self._NNTest(DataSet,**kwargs)
            self.writer.add_hparams(self.HyperParameters
               , {'hparam/Test Loss [dB]':self.MSE_test_dB_avg},
                                    run_name= self.getSaveName('run'))
            self.writer.add_histogram('Linear Test Loss Distribution',self.MSE_test_linear_arr)
        except:
            self.ForceClose()
            raise

        end_time = dt.now()

        training_time = end_time-start_time

        print(f'---Testing of {self.modelName} finished---')

        print(f'Testing took: {training_time.seconds//3600} hours,'
              f' {(training_time.seconds // 60) % 60 } minutes, {training_time.seconds % 60} seconds')


    def _NNTest(self,Test_Dataset, **kwargs):

        N_T = len(Test_Dataset)

        self.MSE_test_linear_arr = torch.empty([N_T],requires_grad= False)

        if hasattr(self.loss_fn,'reduction'):
            self.loss_fn.reduction = 'none'


        Model = torch.load(self.getSaveName('Models'),map_location= self.dev)

        Model.eval()

        Test_Dataloader = DataLoader(Test_Dataset, shuffle=False, batch_size= N_T)

        with torch.no_grad():

            for test_input, test_target in Test_Dataloader:

                Inference_out, test_loss = self.Run_Inference(test_input, test_target, **kwargs)

                self.MSE_test_linear_arr = torch.mean(test_loss,dim=[n for n in range(1,test_loss.ndim)])


            self.MSE_test_linear_avg = self.MSE_test_linear_arr.mean()
            self.MSE_test_dB_avg = 10*torch.log10(self.MSE_test_linear_avg)

            self.writer.add_scalar('Test Loss [dB]', self.MSE_test_dB_avg.item())




