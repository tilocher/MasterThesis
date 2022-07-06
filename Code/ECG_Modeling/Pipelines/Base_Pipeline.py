# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import datetime
import time
import os

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm import trange
import numpy as np

from torch.utils.tensorboard import SummaryWriter

import shutil
import psutil

from tensorboard import program


class Pipeline():

    def __init__(self, modelName, dev = 'gpu'):

        if torch.cuda.is_available() and dev == 'gpu':
            self.dev = torch.device("cuda:0")
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            print("using GPU!")
        else:
            self.dev = torch.device("cpu")
            print("using CPU!")

        self.Base_folder  = os.path.dirname(os.path.realpath(__file__))
        self.modelName = modelName
        self.Time = time.time()

        self.ManageFiles()

        self.LaunchTensorBoard()

        self.writer = SummaryWriter(self.RunFolderName + self.RunFileName)




    def LaunchTensorBoard(self):

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.RunFolderName])
        url = tb.launch()
        print(f'Tensorboard listening on {url}')

        self.TensorBoard_Port = int(''.join([n for n in url if n.isdigit()]))




    def ManageFiles(self):

        self.RunFolderName = self.Base_folder +'\\runs\\'+ self.modelName

        if not self.modelName in os.listdir(self.Base_folder + '\\runs'):
            os.makedirs(self.RunFolderName)
            self.run_number = 0
        else:
            all_runs = os.listdir(self.RunFolderName)
            if len(all_runs) > 0:
                all_runs_sorted = sorted(all_runs, key=lambda x: int(x[4:]))
                self.run_number = int(all_runs_sorted[-1][4:]) + 1
            else:
                self.run_number = 0

        self.RunFolderName += '\\'
        self.RunFileName = 'run_{}'.format(self.run_number)


        self.modelFolderName = self.Base_folder + '\\Models\\{}\\'.format(self.modelName)
        if not self.modelName in os.listdir(self.Base_folder + '\\Models'):
            os.makedirs(self.modelFolderName)

        self.modelFileName = 'Model_run_{}.pt'.format(self.run_number)

        self.PipelineFolderName = self.Base_folder + '\\Pipelines\\{}\\'.format(self.modelName)
        if not self.modelName in os.listdir(self.Base_folder + '\\Pipelines'):
            os.makedirs(self.PipelineFolderName)

        self.PipelineFileName = 'Pipeline_run_{}.pt'.format(self.run_number)


    def save(self):
        torch.save(self, self.PipelineFolderName + self.PipelineFileName)

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
        self.split_ratio = split_ratio

        # MSE LOSS Function
        self.loss_fn = loss_fn

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

        self.writer.add_hparams(
            {'lr':self.learningRate,
             'BatchSize': self.N_B,
             'Shuffle': self.shuffle,
             'Train/CV Split Ratio ': self.split_ratio,
             'Loss Function': str(self.loss_fn),
             'L2': self.weightDecay,
             'Optimizer': 'ADAM'},{})

    def ForceExit(self):
        self.writer.close()
        force_close_port(self.TensorBoard_Port)
        if self.modelFileName in os.listdir(self.modelFolderName): os.remove(self.modelFolderName + self.modelFileName)
        if self.RunFileName in os.listdir(self.RunFolderName): shutil.rmtree(self.RunFolderName + self.RunFileName)
        if self.PipelineFileName in os.listdir(self.PipelineFolderName): os.remove(
            self.PipelineFolderName + self.PipelineFileName)

    def InitModel(self,**kwargs):
        raise NotImplementedError('Methode needs to be implemented outside of base-class')

    def Run_Inference(self,input,target,**kwargs):
        raise NotImplementedError('Methode needs to be implemented outside of base-class')

    def NNTrain(self, DataSet, epochs=None, **kwargs):

        try:
            self._NNTrain(DataSet,epochs,**kwargs)
        except:
            self.ForceExit()

    def _NNTrain(self, DataSet, epochs=None, **kwargs):

        DataSet_length = len(DataSet)

        N_train = int(self.split_ratio * DataSet_length)
        N_CV = DataSet_length - N_train


        print(okk)

        self.MSE_cv_linear_epoch = torch.empty([self.N_Epochs], requires_grad=False).to(self.dev, non_blocking=True)
        self.MSE_cv_dB_epoch = torch.empty([self.N_Epochs], requires_grad=False).to(self.dev, non_blocking=True)

        self.MSE_train_linear_epoch = torch.empty([self.N_Epochs], requires_grad=False).to(self.dev, non_blocking=True)
        self.MSE_train_dB_epoch = torch.empty([self.N_Epochs], requires_grad=False).to(self.dev, non_blocking=True)

        ##############
        ### Epochs ###
        ##############

        MSE_cv_dB_opt = 1000
        MSE_cv_idx_opt = 0

        if epochs is None:
            N = self.N_Epochs
        else:
            N = epochs

        sample_input, sample_target = DataSet[np.random.randint(0, len(DataSet))]

        self.writer.add_graph(self.model, sample_input.unsqueeze(0))

        Epoch_itter = trange(N)

        for ti in Epoch_itter:

            Train_Dataset, CV_Dataset = torch.utils.data.random_split(DataSet, [N_train, N_CV],
                                                                      generator=torch.Generator(device=self.dev))

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()

            CV_Dataloader = DataLoader(CV_Dataset, shuffle=False, batch_size=N_CV)

            try:
                self.InitModel(**kwargs)
            except:
                raise ModelInitError()

            for cv_input, cv_target in CV_Dataloader:
                try:
                    Inference_out, cv_loss = self.Run_Inference(cv_input, cv_target, **kwargs)
                except:
                    raise InferenceError()

                self.MSE_train_linear_epoch[ti] = cv_loss.detach()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(cv_loss).detach()

                Epoch_cv_loss_lin = cv_loss.item()

            if (self.MSE_cv_dB_epoch[ti] < MSE_cv_dB_opt):

                MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]

                torch.save(self.model, self.modelFolderName + self.modelFileName)

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            Train_DataLoader = DataLoader(Train_Dataset, batch_size=self.N_B, shuffle=self.shuffle,
                                          generator=torch.Generator(device=self.dev))
            try:
                self.InitModel(**kwargs)
            except:
                raise ModelInitError()

            MSE_train_linear_batch = torch.empty(Train_DataLoader.__len__(), device=self.dev, requires_grad=False)

            for j, (train_input, train_target) in enumerate(Train_DataLoader):

                try:
                    Inference_out, train_loss = self.Run_Inference(train_input, train_target, **kwargs)
                except:
                    raise InferenceError()

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
                'Epoch training Loss: {} [dB], Epoch Val. Loss: {} [dB]'.format(train_desc, cv_desc))

        # del Epoch_train_loss, Epoch_cv_loss, train_desc, cv_desc, train_loss, cv_loss

        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]



def force_close_port(port, process_name=None):
    """Terminate a process that is bound to a port.

    The process name can be set (eg. python), which will
    ignore any other process that doesn't start with it.
    """
    for proc in psutil.process_iter():
        for conn in proc.connections():
            if conn.laddr[1] == port:

                try:
                    proc.username()
                except psutil.AccessDenied:
                    pass
                else:
                    if process_name is None or proc.name().startswith(process_name):
                        try:
                            proc.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass


class ModelInitError(Exception):

    def __init__(self):
        super(ModelInitError, self).__init__('Error in the Model Initialization function')

class InferenceError(Exception):

    def __init__(self):
        super(InferenceError, self).__init__('Error in the Inference function')