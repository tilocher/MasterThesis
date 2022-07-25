import torch

from DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH
import yaml
from yaml.loader import SafeLoader
from log.BaseLogger import WandbLogger,LocalLogger

from Pipelines.EM_Pipeline import EM_Taylor_Pipeline
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
                                    plot_sample=False, desired_shape=(1, signal_length, 2), roll=0)
    np.random.seed(42)
    randint = np.random.randint(len(loader))

    noise,sample = loader[randint]
    sample = sample.squeeze()
    noise = noise.squeeze()


    N_train = int(0.8 * len(loader)) # int(0.05 * len(loader))
    N_test = len(loader) - N_train

    dev = torch.device('cpu')
    torch.random.manual_seed(42)

    Train_Loader, Test_Loader = torch.utils.data.random_split(loader, [N_train, N_test],
                                                              generator=torch.Generator())

    # N_test = 500

    # Test_Loader.indices = list(np.random.choice(Test_Loader.indices, 500, replace = False))

    Test_Loader.indices = Test_Loader.indices[:500]

    Cmodel = ConstantModel(config['Order'], 0,1, 1, 360, deltaT = 1)#1/loader.fs)# 7e-3)#1/1000)


    EM_Pipe = EM_Taylor_Pipeline(Cmodel,Logger, em_parameters= ['mu','Sigma','R','Q'])

    EM_Pipe.TrainPrior(Train_Loader)

    EM_Pipe.TestEM(Test_Loader, em_its= config['EM_Its'])





