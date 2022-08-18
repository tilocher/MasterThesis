from sklearn.mixture import GaussianMixture
from DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH

if __name__ == '__main__':
    import torch

    from log.BaseLogger import LocalLogger, WandbLogger
    import yaml
    from yaml.loader import SafeLoader

    config = yaml.load(open('../Configs/EM.yaml'), Loader=SafeLoader)

    Logger = LocalLogger('EM_Taylor', BaseConfig=config) if not config['wandb'] else \
        WandbLogger(name='EM_Taylor', group='SNR_sweep', BaseConfig=config)

    config = Logger.GetConfig()

    snr = config['snr']

    signal_length = config['signal_length']

    UseWandb = config['wandb']

    loader = PhyioNetLoader_MIT_NIH(1, 1, signal_length, SNR_dB=snr, random_sample=False, gpu=False,
                                    plot_sample=False, desired_shape=(1, signal_length, 2), roll=0)

    N_train = int(0.8 * len(loader))  # int(0.05 * len(loader))
    N_test = len(loader) - N_train

    dev = torch.device('cpu')
    torch.random.manual_seed(42)

    Train_Loader, Test_Loader = torch.utils.data.random_split(loader, [N_train, N_test],
                                                              generator=torch.Generator())
    import numpy as np
    from matplotlib import pyplot as plt


    signal = Train_Loader[0][0]

    a = GaussianMixture(5).fit(signal[:, :, 0].T+torch.min(signal[:,:,0]) +1e-6)
    t = np.linspace(-10,1,300)
    logprob = a.score_samples(t.reshape(-1,1))
    pred = np.exp(logprob)
    plt.plot(pred)
    # plt.plot(signal[:,:,0].squeeze())
    plt.show()

    1