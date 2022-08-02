import matplotlib.pyplot as plt
import torch

import DataLoaders
from DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH
import yaml
from yaml.loader import SafeLoader
from log.BaseLogger import WandbLogger,LocalLogger
from NNs.RTSNet_nn import RTSNetNN
from Pipelines.RTSNet_Pipeline import RTSNet_Pipeline
import os
import wandb
import copy
import numpy as np
from torch.utils.data.dataloader import DataLoader
from SystemModels.Taylor_model import Taylor_model




if __name__ == '__main__':


    config = yaml.load(open('Configs/RTSNet.yaml'), Loader=SafeLoader)

    UseWandb = config['wandb']


    Logger = LocalLogger('RTSNet', BaseConfig=config) if not UseWandb else \
        WandbLogger(name='RTSNet',group= config['WandbGroup'], BaseConfig=config)

    config = Logger.GetConfig()

    snr = config['snr']

    signal_length = config['signal_length']

    gpu = config['gpu']

    roll = config['roll']

    num_sets = config['Number of MIT-BIH sets']

    num_heartbeats = config['NumberHeartbeats']


    loader = PhyioNetLoader_MIT_NIH(num_sets= num_sets, num_beats= num_heartbeats,
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

    ##############################################################################################
    ##############################################################################################
    ##############################################################################################

    ssModel.GenerateSequence(ssModel.T)

    sample = ssModel.x.T[:, 0]

    # plt.plot(self.ssModel.x.T, label = 'Learned Prior')
    t = np.arange(start=0, stop=sample.shape[0] / (360), step=1 / (360))
    fig, ax = plt.subplots(dpi=200)

    ax.plot(t, sample, label='Learned Prior', color='g')
    # ax.plot(t, noise[:, 0], label='Noisy observation', color='r', alpha=0.4)
    ax.grid()
    ax.legend()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude [mV]')

    axins = ax.inset_axes([0.05, 0.5, 0.4, 0.4])
    axins.plot(t, sample, color='g')
    axins.get_xaxis().set_visible(False)
    axins.get_yaxis().set_visible(False)

    x1, x2, y1, y2 = 0.4, 0.6, torch.min(sample).item(), torch.max(sample).item()
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.grid()

    ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.show()

    ##############################################################################################
    ##############################################################################################
    ##############################################################################################


    nnModel = RTSNetNN(gpu)
    nnModel.Build(ssModel)


    ECG_Pipeline = RTSNet_Pipeline(Logger= Logger,unsupervised=config['unsupervised'], gpu = gpu)
    ECG_Pipeline.setModel(nnModel)
    ECG_Pipeline.setTrainingParams(weightDecay=config['L2'], n_Epochs=config['Epochs'], n_Batch=config['BatchSize'],
                                   learningRate=config['lr'], shuffle=True, split_ratio=0.7)

    ECG_Pipeline.NNTrain(Train_Loader)

    ECG_Pipeline.NNTest(Test_Loader)
