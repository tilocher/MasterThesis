import matplotlib.pyplot as plt
import torch

import DataLoaders
from DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH, PhyioNetLoader_AbdominalAndDirect,FECG_Loader
from DataLoaders.RikLoader import RikDataset
import yaml
from yaml.loader import SafeLoader
from log.BaseLogger import WandbLogger,LocalLogger

from SystemModels.Taylor_model import Taylor_model
import numpy as np
from Pipelines.EM_Pipelines import *
from SystemModels.PDE_Model import PDE_Model
import PriorModels

if __name__ == '__main__':

    config = yaml.load(open('Configs/EM.yaml'), Loader=SafeLoader)

    Prior = config['Prior']

    Logger = LocalLogger(f'EM_algorithm_{Prior}', BaseConfig=config) if not config['wandb'] else \
        WandbLogger(name='EM_algorithm', group= config['wandbGroup'], BaseConfig=config)

    try:

        config = Logger.GetConfig()

        snr = config['snr']

        signal_length = config['signal_length']

        UseWandb = config['wandb']


        Mode = config['Mode']

        smoothing_window_Q = config['smoothing_window_Q']
        smoothing_window_R = config['smoothing_window_R']

        prior = config['Prior']
        priorParams = config['Parameters']



        priorModel = eval(f'PriorModels.{prior}Prior(**priorParams)')

        loader = PhyioNetLoader_MIT_NIH(1, 1, signal_length, SNR_dB=snr, random_sample=False, gpu=False,
                                        plot_sample=False, desired_shape=(1, signal_length, 2), roll=0, offset= 0)
        # loader_missmatch = PhyioNetLoader_MIT_NIH(1, 1, signal_length, SNR_dB=snr, random_sample=False, gpu=False,
        #                                 plot_sample=False, desired_shape=(1, signal_length, 2), roll=0,offset = 0)
        # signal_length_ratio = 0.5
        # loader = RikDataset(desired_shape=(1, int(signal_length_ratio*500), 12),gpu = config['gpu'],
        #                     preprocess=config['preprocess'], snr_dB= snr, offset= 0, signal_length_ratio=signal_length_ratio)




        if config['Prior'] == 'Identity':
            class dummy():
                def fit(self,x):
                    raise NotImplementedError
            taylor_model = dummy()

        if config['Prior'] == 'Synthetic':
            taylor_model = PDE_Model(360,2)




        EM_Pipe = eval(f'{Prior}_Pipeline(priorModel,Logger, em_parameters= config["EM_vars"],'
                       f' Mode = Mode, smoothing_window_Q = smoothing_window_Q,smoothing_window_R = smoothing_window_R )')

        EM_rep = config['EM_rep'] if 'EM_rep' in config.keys() else 10
        Num_Plot_Samples = config['Num_Plot_Samples'] if 'Num_Plot_Samples' in config.keys() else 10

        # EM_Pipe.FitPrior(loader)

        EM_Pipe.Run(loader, em_its=config['EM_Its'],PriorSamples=config['PriorSamples'],
                    ConvergenceThreshold=config['ConvergenceThreshold'],EM_rep=EM_rep, Num_Plot_Samples= Num_Plot_Samples)



    except:
        Logger.ForceClose()
        raise





