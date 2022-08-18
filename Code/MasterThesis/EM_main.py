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

    config = yaml.load(open('Configs/EM.yaml'), Loader=SafeLoader)

    Logger = LocalLogger('EM_algorithm', BaseConfig=config) if not config['wandb'] else \
        WandbLogger(name='EM_algorithm', group= config['wandbGroup'], BaseConfig=config)

    try:

        config = Logger.GetConfig()

        snr = config['snr']

        signal_length = config['signal_length']

        UseWandb = config['wandb']


        loader = PhyioNetLoader_MIT_NIH(1, 1, signal_length, SNR_dB=snr, random_sample=False, gpu=False,
                                        plot_sample=False, desired_shape=(1, signal_length, 2), roll=0)


        N_train = int(0.8 * len(loader))
        N_test = len(loader) - N_train

        dev = torch.device('cpu')
        torch.random.manual_seed(42)

        Train_Loader, Test_Loader = torch.utils.data.random_split(loader, [N_train, N_test],
                                                                  generator=torch.Generator())

        Test_Loader.indices = Test_Loader.indices[:500]



        taylor_model = Taylor_model(taylor_order= config['TaylorOrder'],  window= config['Window'],
                                    window_size= config['WindowSize'], window_parameters = config['WindowParameter'])

        Fit = config['Fit']
        Mode = config['Mode']

        EM_Pipe = EM_Taylor_Pipeline(taylor_model,Logger, em_parameters= config['EM_vars'], Fit = Fit, Mode = Mode)


        EM_Pipe.TrainPrior(Train_Loader)

        EM_Pipe.TestEM(Test_Loader, em_its=config['EM_Its'],ConvergenceThreshold=config['ConvergenceThreshold'])

    except:
        Logger.ForceClose()
        raise





