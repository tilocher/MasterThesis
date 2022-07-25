import matplotlib.pyplot as plt
import torch

import DataLoaders
from DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH
import yaml
from yaml.loader import SafeLoader
from log.BaseLogger import WandbLogger,LocalLogger
from NNs.AutoEncoder import AutoEncoder
from Pipelines.AutoEncoder_Pipeline import ECG_AE_Pipeline
import os
import wandb
import copy
from Pipelines.EM_Pipeline import EM_Taylor_Pipeline
from SystemModels.Taylor_model import Taylor_model
import numpy as np

if __name__ == '__main__':

    config = yaml.load(open('Configs/EM_taylor.yaml'), Loader=SafeLoader)

    Logger = LocalLogger('EM_Taylor', BaseConfig=config) if not config['wandb'] else \
        WandbLogger(name='EM_Taylor', group='SNR_sweep', BaseConfig=config)

    config = Logger.GetConfig()

    snr = config['snr']

    signal_length = config['signal_length']

    UseWandb = config['wandb']




    loader = PhyioNetLoader_MIT_NIH(1, 1, signal_length, SNR_dB=snr, random_sample=False, gpu=False,
                                    plot_sample=False, desired_shape=(1, signal_length, 2), roll=0)
    np.random.seed(42)
    randint = np.random.randint(len(loader))

    noise,sample = loader[randint]
    sample = sample.squeeze()
    noise = noise.squeeze()

    # t = np.arange(start=0, stop=sample.shape[0] / (loader.fs), step=1 / (loader.fs))
    # fig, ax = plt.subplots(dpi = 200)
    #
    # ax.plot(t,sample[:,0], label = 'Noiseless signal', color = 'g')
    # ax.plot(t,noise[:,0],label = 'Noisy observation', color = 'r',alpha = 0.4)
    # ax.grid()
    # ax.legend()
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Amplitude [mV]')
    # ax.set_title('MIT-BIH Arrhythmia Database Preprocessed sample SNR: 15[dB]')
    #
    # axins = ax.inset_axes([0.05, 0.5, 0.4, 0.4])
    # axins.plot(t,sample[:,0], color = 'g')
    # axins.get_xaxis().set_visible(False)
    # axins.get_yaxis().set_visible(False)
    #
    # x1, x2, y1, y2 = 0.4, 0.6, -0.7, 0.9
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])
    # axins.grid()
    #
    # ax.indicate_inset_zoom(axins, edgecolor="black")
    #
    #
    # # plt.savefig('MIT-BIH_Arrhythmia_Database_sample_noise_15dB.pdf')
    # plt.show()

    N_train = int(0.8 * len(loader)) # int(0.05 * len(loader))
    N_test = len(loader) - N_train

    dev = torch.device('cpu')
    torch.random.manual_seed(42)

    Train_Loader, Test_Loader = torch.utils.data.random_split(loader, [N_train, N_test],
                                                              generator=torch.Generator())

    # N_test = 500

    # Test_Loader.indices = list(np.random.choice(Test_Loader.indices, 500, replace = False))

    Test_Loader.indices = Test_Loader.indices[:500]



    taylor_model = Taylor_model(taylor_order= config['TaylorOrder'],  window= config['Window'],
                                window_size= config['WindowSize'], window_parameters = config['WindowParameter'])



    EM_Pipe = EM_Taylor_Pipeline(taylor_model,Logger, em_parameters= ['mu','Sigma','R','Q'])

    EM_Pipe.TrainPrior(Train_Loader)

    EM_Pipe.TestEM(Test_Loader, em_its= config['EM_Its'])





