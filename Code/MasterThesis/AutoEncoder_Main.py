import torch
from DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH
import yaml
from yaml.loader import SafeLoader
from log.BaseLogger import WandbLogger,LocalLogger
from NNs.AutoEncoder import AutoEncoder
from Pipelines.AutoEncoder_Pipeline import ECG_AE_Pipeline
import copy
import numpy as np
from utils import GetSubset

from DataLoaders.RikLoader import RikDataset


if __name__ == '__main__':


    config = yaml.load(open('Configs/ECG_AutoEncoder.yaml'), Loader=SafeLoader)

    UseWandb = config['wandb']


    Logger = LocalLogger(config['LoggerName'], BaseConfig= config) if not UseWandb else\
                    WandbLogger(name= config['LoggerName'],group=config['wandbGroup'], BaseConfig= config)

    config = Logger.GetConfig()

    snr = config['snr']

    signal_length = config['signal_length']

    gpu = config['gpu']

    roll = config['roll']

    Segmented = config['Segmented']

    # loader = PhyioNetLoader_MIT_NIH(num_sets= 47, num_beats= 1,
    #                                 num_samples= signal_length, SNR_dB= snr, gpu= gpu,
    #                                 desired_shape= (1, signal_length, 2), roll= roll,offset=1)
    #
    # loaderTest = PhyioNetLoader_MIT_NIH(num_sets=1, num_beats=1,
    #                                 num_samples=signal_length, SNR_dB=snr, gpu=gpu,
    #                                 desired_shape=(1, signal_length, 2), roll=roll, offset= 0)

    signal_length_ratio = 0.5
    loader = RikDataset(num_files=130,desired_shape=(1, int(signal_length_ratio * 500), 12), gpu=config['gpu'],
                        preprocess=config['preprocess'], snr_dB=snr, offset=1, signal_length_ratio=signal_length_ratio)

    if Segmented:
        loader.SplitToSegments()

    N_train = int(0.99 * len(loader))
    # N_train = 32
    # N_test = len(loader) - N_train
    dev = torch.device('cuda:0' if torch.cuda.is_available() and loader.gpu else 'cpu')

    torch.random.manual_seed(42)


    Train_Loader,Test_Loader = GetSubset(loader, N_train)

    # # Train_Loader = loader
    # Test_Loader,_ = GetSubset(loaderTest,len(loaderTest))

    # Train_Loader, Test_Loader = torch.utils.data.random_split(loader, [N_train, N_test],
    #                                                           generator=torch.Generator())

    # Test_Loader,_ = torch.utils.data.random_split(Test_Loader, [10, len(Test_Loader)-10],
    #                                                           generator=torch.Generator())

    # N_test = 500

    # Test_Loader.indices = list(np.random.choice(Test_Loader.indices, 500, replace = False))
    Test_Loader = copy.deepcopy(Test_Loader)
    Test_Loader.dataset.roll = 0


    LATENT_SPACE = config['LatentSpace']


    if Segmented:
        num_segments = len(Train_Loader[0][0])

    else:
        num_segments = 1

    models = []
    Pipelines = []

    for i in range(num_segments):

        nnModel = AutoEncoder(num_channels=12,
                              signal_length=signal_length,
                              conv_filters=(40, 20, 20, 20, 20, 40),
                              conv_kernels=((40, 2), (40, 2), (40, 2), (40, 2), (40, 2), (40, 2)),
                              conv_strides=((2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1)),
                              conv_dilation=((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)),
                              latent_space_dim=LATENT_SPACE)

        ECG_Pipeline = ECG_AE_Pipeline(logger= Logger, Segmented= Segmented)
        ECG_Pipeline.setModel(nnModel)
        ECG_Pipeline.setTrainingParams(weightDecay=config['L2'], n_Epochs=config['Epochs'], n_Batch=config['BatchSize'],
                                       learningRate=config['lr'],
                                       shuffle=True, split_ratio=0.7, loss_fn=torch.nn.MSELoss(reduction='mean'))

        ECG_Pipeline.NNTrain(Train_Loader,segment = i)

        models.append(ECG_Pipeline.model)


        ECG_Pipeline.NNTest(Test_Loader, segment = i)

        Pipelines.append(ECG_Pipeline)

    # ECG_Pipeline.TestWhole(Test_Loader,models)