import matplotlib.pyplot as plt
import torch

import DataLoaders
from DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH
import yaml
from yaml.loader import SafeLoader
from log.BaseLogger import WandbLogger,LocalLogger
from NNs.KalmanNet_nn import KalmanNetNN
from Pipelines.KNet_Pipeline import KNet_Pipeline
import os
import wandb
import copy
import numpy as np
from torch.utils.data.dataloader import DataLoader
from SystemModels.Taylor_model import Taylor_model




if __name__ == '__main__':


    config = yaml.load(open('Configs/KNet.yaml'), Loader=SafeLoader)

    UseWandb = config['wandb']


    Logger = LocalLogger('KNet', BaseConfig=config) if not UseWandb else \
        WandbLogger(name='KNet', BaseConfig=config)

    config = Logger.GetConfig()

    snr = config['snr']

    signal_length = config['signal_length']

    gpu = config['gpu']

    roll = config['roll']


    loader = PhyioNetLoader_MIT_NIH(num_sets= 2, num_beats= 1,
                                    num_samples= signal_length, SNR_dB= snr, gpu= gpu,
                                    desired_shape= (1, signal_length, 2), roll= roll)


    N_train = int(0.8 * len(loader))
    N_test = len(loader) - N_train
    dev = torch.device('cuda:0' if torch.cuda.is_available() and loader.gpu else 'cpu')
    torch.random.manual_seed(42)
    Train_Loader, Test_Loader = torch.utils.data.random_split(loader, [N_train, N_test],
                                                              generator=torch.Generator(device=dev))

    Test_Loader = copy.deepcopy(Test_Loader)
    Test_Loader.dataset.roll = 0

    TaylorModel = Taylor_model(taylor_order= config['TaylorOrder'],  window= config['Window'],
                                window_size= config['WindowSize'], window_parameters = config['WindowParameter'])

    DataSet_length = len(Train_Loader)

    TrainDataset = DataLoader(Train_Loader, shuffle=False, batch_size=DataSet_length)

    train_inputs, _ = next(iter(TrainDataset))

    TaylorModel.fit(train_inputs.squeeze().mT)

    ssModel = TaylorModel.GetSysModel(train_inputs.shape[-1],gpu= gpu)
    # self.ssModel.f = lambda x,t: x
    ssModel.InitSequence(torch.zeros((2, 1),device=dev), torch.eye(2,device=dev))


    nnModel = KalmanNetNN()
    nnModel.NNBuild(ssModel)
    nnModel.InitSequence(torch.zeros_like(Train_Loader[:][0]),None, 360)


    ECG_Pipeline = KNet_Pipeline(Logger= Logger)
    ECG_Pipeline.setModel(nnModel)
    ECG_Pipeline.setTrainingParams(weightDecay=config['L2'], n_Epochs=config['Epochs'], n_Batch=config['BatchSize'],
                                   learningRate=config['lr'], shuffle=True, split_ratio=0.7)

    ECG_Pipeline.NNTrain(Train_Loader)

    ECG_Pipeline.NNTest(Test_Loader)
