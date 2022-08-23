import torch

from DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH
import yaml
from yaml.loader import SafeLoader
from log.BaseLogger import WandbLogger,LocalLogger

from Pipelines.EM_Pipelines import EM_Pipeline,ConstantModel_Pipeline
from SystemModels.ConstantModels import ConstantModel
import numpy as np

if __name__ == '__main__':

    config = yaml.load(open('Configs/EM_Const.yaml'), Loader=SafeLoader)

    Logger = LocalLogger('Const_EM', BaseConfig=config) if not config['wandb'] else \
        WandbLogger(name='Const_EM', group='Const_EM', BaseConfig=config)

    config = Logger.GetConfig()

    snr = config['snr']

    signal_length = config['signal_length']

    UseWandb = config['wandb']


    loader = PhyioNetLoader_MIT_NIH(1, 1, signal_length, SNR_dB=snr, random_sample=False, gpu=False,
                                    plot_sample=False, desired_shape=(1, signal_length, 2), roll=0,channels = 1)

    Cmodel = ConstantModel(config['Order'], 0,1, 1, 360, deltaT = 1)#1/loader.fs)# 7e-3)#1/1000)


    EM_Pipe = ConstantModel_Pipeline(Cmodel,Logger, em_parameters= config['EM_vars'],Mode=config['Mode'])
    EM_Pipe.Run(loader, PriorSamples= 2270 -10, em_its= config['EM_Its'], ConvergenceThreshold= config['ConvergenceThreshold'])






