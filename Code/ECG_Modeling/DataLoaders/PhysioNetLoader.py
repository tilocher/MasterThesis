# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import glob
# Load libraries
import numpy as np
import wfdb
import mne
import torch

import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader, Dataset


class PhyioNetLoader_AbdominalAndDirect(Dataset):

    def __init__(self):
        super(PhyioNetLoader_AbdominalAndDirect, self).__init__()

        edf_files = glob.glob('..\\Datasets\\PhysioNet\\abdominal-and-direct-fetal\\*.edf')
        self.datasets = [mne.io.read_raw_edf(edf_file) for edf_file in edf_files]

        test = mne.io.read_raw_edf('..\\Datasets\\PhysioNet\\abdominal-and-direct-fetal\\ANNOTATORS.edf')

    def PlotSample(self, length):
        plt.plot(self.datasets[0].get_data()[1:, :2000].mean(0), label='mean')
        plt.plot(self.datasets[0].get_data()[0, :2000], label='gt')
        plt.legend()
        plt.show()


class PhyioNetLoader_MIT_NIH(Dataset):
    '''
        @article{goldberger2000physiobank,
        title={PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals},
        author={Goldberger, Ary L and Amaral, Luis AN and Glass, Leon and Hausdorff, Jeffrey M and Ivanov, Plamen Ch and Mark, Roger G and Mietus, Joseph E and Moody, George B and Peng, Chung-Kang and Stanley, H Eugene},
        journal={circulation},
        volume={101},
        number={23},
        pages={e215--e220},
        year={2000},
        publisher={Am Heart Assoc}
        }
        from: https://archive.physionet.org/physiobank/database/nstdb/?C=N;O=D
        and: https://archive.physionet.org/physiobank/database/mitdb/
    '''

    def __init__(self, num_sets: int, sample_len_sec: float, SNR_dB: float):
        super(PhyioNetLoader_MIT_NIH, self).__init__()

        self.sample_len_sec = sample_len_sec

        self.SNR_dB = SNR_dB

        header_files = glob.glob('..\\Datasets\\PhysioNet\\MIT-BIH_Arrhythmia_Database\\*.hea')

        rand_sample = np.random.choice(header_files, num_sets, replace=False)

        self.files = [wfdb.rdrecord(header_file[:-4]) for header_file in rand_sample]

        self.fs = self.files[0].fs

        self.dataset = torch.tensor([file.p_signal for file in self.files]).mT

        self.SplitToSeconds(sample_len_sec)

        self.AddGaussianNoise(SNR_dB)

        self.PlotSample()

    def SplitToSeconds(self, sample_len_sec: float) -> torch.Tensor:
        """
        Split the dataset into sample_len_sec long segments
        :param sample_len_sec: The length in seconds of the desired signal
        :return: The split dataset
        """

        samples = int((sample_len_sec * self.fs))

        split_data = torch.split(self.dataset, samples, dim=-1)[:-1]

        split_data = self.split_dataset = torch.concat(split_data, dim=0)

        return split_data

    def __getitem__(self, item: int) -> tuple:
        """
        Get the specified sample of noisy and clean data from the dataset
        :param item: index of the samples to get
        :return: A tuple of noisy and clean samples
        """
        return self.noisy_dataset[item], self.split_dataset[item]

    def __len__(self) -> int:
        """
        Get the size of the dataset
        :return: Size of the dataset
        """
        return self.split_dataset.shape[0]

    def AddGaussianNoise(self, SNR_dB: float) -> None:
        """
        Add white gaussian noise to the split dataset
        :param SNR_dB: Signal to noise ration in decibel
        :return:
        """

        signals = self.split_dataset
        signal_power_dB = 10 * torch.log10(signals.var(-1) + signals.mean(-1) ** 2)

        noise_power_dB = signal_power_dB - SNR_dB
        noise_power = 10 ** (noise_power_dB / 20).unsqueeze(-1)

        noise = np.random.normal(np.zeros_like(signals), noise_power.repeat(1,1,  signals.shape[-1]), signals.shape)
        noise_power_num = 10 * np.log10(noise.var(-1) + noise.mean(-1) ** 2)
        print('SNR of actual signal', round((signal_power_dB - noise_power_num).mean().item(), 3), '[dB]')

        noisy_sample = signals + noise

        self.noisy_dataset = noisy_sample

    def PlotSample(self) -> None:
        """
        Plot a random sample of a random channel with noise
        :return: None
        """

        # Randomly select sample and channel
        sample = np.random.randint(0, self.split_dataset.shape[0])
        channel = np.random.randint(0, self.split_dataset.shape[1])

        # Get the sample
        sample_signal = self.split_dataset[sample, channel,:]
        noisy_sample = self.noisy_dataset[sample, channel,:]

        # Time axis
        t = np.arange(start=0, stop=sample_signal.shape[0] / (self.fs), step=1 / (self.fs))

        # Plotting
        plt.plot(t, sample_signal[:], 'g', label='Ground Truth')
        plt.plot(t, noisy_sample[:], 'r', label='Noisy Signal', alpha=0.4)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [mV]')
        plt.title('MIT-BIH Dataset sample with additive GWN: SNR {} [dB]'.format(round(self.SNR_dB, 2)))
        plt.legend()
        plt.savefig('..\\Plots\\MIT-BIH_sample_plot_snr_{}dB.pdf'.format(round(self.SNR_dB, 2)))
        plt.show()


    def GetData(self):
        return self.noisy_dataset, self.split_dataset

if __name__ == '__main__':
    dataset = PhyioNetLoader_MIT_NIH(4, 2, 10)
