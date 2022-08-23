import matplotlib.pyplot as plt
import torch

import DataLoaders
from DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH
import yaml
from yaml.loader import SafeLoader
from log.BaseLogger import WandbLogger,LocalLogger

from SystemModels.Taylor_model import Taylor_model
import numpy as np
from Pipelines.EM_Pipelines import *

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





        taylor_model = Taylor_model(taylor_order= config['TaylorOrder'],  window= config['Window'],
                                    window_size= config['WindowSize'], window_parameters = config['WindowParameter'])

        Fit = config['Fit']
        Mode = config['Mode']

        EM_Pipe = eval('TaylorModel_Pipeline(taylor_model,Logger, em_parameters= config["EM_vars"], Mode = Mode)'.format(Fit))

        EM_Pipe.Run(loader, em_its=config['EM_Its'],PriorSamples=config['PriorSamples'],ConvergenceThreshold=config['ConvergenceThreshold'])

    except:
        Logger.ForceClose()
        raise





