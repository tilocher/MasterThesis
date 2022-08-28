import matplotlib.pyplot as plt
import torch

import DataLoaders
from DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH, PhyioNetLoader_AbdominalAndDirect
from DataLoaders.RikLoader import RikDataset
import yaml
from yaml.loader import SafeLoader
from log.BaseLogger import WandbLogger,LocalLogger

from SystemModels.Taylor_model import Taylor_model
import numpy as np
from Pipelines.EM_Pipelines import *

if __name__ == '__main__':

    config = yaml.load(open('Configs/EM.yaml'), Loader=SafeLoader)

    Fit = config['Fit']

    Logger = LocalLogger(f'EM_algorithm_{Fit}', BaseConfig=config) if not config['wandb'] else \
        WandbLogger(name='EM_algorithm', group= config['wandbGroup'], BaseConfig=config)

    try:

        config = Logger.GetConfig()

        snr = config['snr']

        signal_length = config['signal_length']

        UseWandb = config['wandb']


        Mode = config['Mode']

        # loader = PhyioNetLoader_MIT_NIH(1, 1, signal_length, SNR_dB=snr, random_sample=False, gpu=False,
        #                                 plot_sample=False, desired_shape=(1, signal_length, 2), roll=0)

        FetalLoader = PhyioNetLoader_AbdominalAndDirect(sample=0,desired_shape=(1, signal_length, 4))


        # loader = RikDataset()



        taylor_model = Taylor_model(taylor_order= config['TaylorOrder'],  window= config['Window'],
                                    window_size= config['WindowSize'], window_parameters = config['WindowParameter'])



        EM_Pipe = eval(f'{Fit}_Pipeline(taylor_model,Logger, em_parameters= config["EM_vars"], Mode = Mode)')

        EM_rep = config['EM_rep'] if 'EM_rep' in config.keys() else 10
        Num_Plot_Samples = config['Num_Plot_Samples'] if 'Num_Plot_Samples' in config.keys() else 10

        EM_Pipe.Run(FetalLoader, em_its=config['EM_Its'],PriorSamples=config['PriorSamples'],
                    ConvergenceThreshold=config['ConvergenceThreshold'],EM_rep=EM_rep, Num_Plot_Samples= Num_Plot_Samples)

        mECG = EM_Pipe.ConsecutiveFilter.Filtered_State_Means[0,]

    except:
        Logger.ForceClose()
        raise





