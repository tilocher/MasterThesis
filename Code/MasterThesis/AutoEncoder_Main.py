import torch
from DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH
import yaml
from yaml.loader import SafeLoader
from log.BaseLogger import WandbLogger,LocalLogger
from NNs.AutoEncoder import AutoEncoder
from Pipelines.AutoEncoder_Pipeline import ECG_AE_Pipeline
import copy
import numpy as np




if __name__ == '__main__':


    config = yaml.load(open('Configs/ECG_AutoEncoder.yaml'), Loader=SafeLoader)

    UseWandb = config['wandb']


    Logger = LocalLogger('AutoEncoder_SNR_sweep_roll', BaseConfig= config) if not UseWandb else\
                    WandbLogger(name= 'AutoEncoder_SNR_sweep_roll',group='AutoEncoder_SNR_sweep_roll', BaseConfig= config)

    config = Logger.GetConfig()

    snr = config['snr']

    signal_length = config['signal_length']

    gpu = config['gpu']

    roll = config['roll']

    loader = PhyioNetLoader_MIT_NIH(num_sets= 4, num_beats= 1,
                                    num_samples= signal_length, SNR_dB= snr, gpu= gpu,
                                    desired_shape= (1, signal_length, 2), roll= roll)


    N_train = int(0.8 * len(loader))
    # N_train = 100
    N_test = len(loader) - N_train
    dev = torch.device('cuda:0' if torch.cuda.is_available() and loader.gpu else 'cpu')

    torch.random.manual_seed(42)
    Train_Loader, Test_Loader = torch.utils.data.random_split(loader, [N_train, N_test],
                                                              generator=torch.Generator())



    # N_test = 500

    # Test_Loader.indices = list(np.random.choice(Test_Loader.indices, 500, replace = False))

    Test_Loader = copy.deepcopy(Test_Loader)
    Test_Loader.dataset.roll = 0


    LATENT_SPACE = config['LatentSpace']

    nnModel = AutoEncoder(num_channels=2,
                          signal_length=signal_length,
                          conv_filters=(40, 20, 20, 20, 20, 40),
                          conv_kernels=((40, 2), (40, 2), (40, 2), (40, 2), (40, 2), (40, 2)),
                          conv_strides=((2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1)),
                          conv_dilation=((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)),
                          latent_space_dim=LATENT_SPACE)

    ECG_Pipeline = ECG_AE_Pipeline(Logger= Logger)
    ECG_Pipeline.setModel(nnModel)
    ECG_Pipeline.setTrainingParams(weightDecay=config['L2'], n_Epochs=config['Epochs'], n_Batch=config['BatchSize'],
                                   learningRate=config['lr'], shuffle=True, split_ratio=0.7)

    ECG_Pipeline.NNTrain(Train_Loader)

    ECG_Pipeline.NNTest(Test_Loader)
