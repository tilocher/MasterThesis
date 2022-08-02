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
import os


class PhyioNetLoader_AbdominalAndDirect(Dataset):

    def __init__(self):
        super(PhyioNetLoader_AbdominalAndDirect, self).__init__()

        self.file_location = os.path.dirname(os.path.realpath(__file__))

        edf_files = glob.glob(self.file_location+'/../Datasets/PhysioNet/abdominal-and-direct-fetal/*.edf')
        self.files = [mne.io.read_raw_edf(edf_file) for edf_file in edf_files]

        self.dataset = torch.tensor(np.array([file.get_data() for file in self.files]), dtype=torch.float32).mT
        # self.labels = torch.tensor(np.array([file.sample for file in self.annotation]), dtype=torch.float32)[0, 1:]

        # test = mne.io.read_raw_edf(self.file_location+'/../Datasets/PhysioNet/abdominal-and-direct-fetal/ANNOTATORS.edf')

    def PlotSample(self, length):
        plt.plot(self.dataset[0, 11000:15000,1:].squeeze(), label='mean')
        # plt.plot(self.datasets[0].get_data()[0, :2000], label='gt')
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

    def __init__(self, num_sets: int, num_beats: int, num_samples: int, SNR_dB: float, random_sample = False,
                 gpu = True, plot_sample = False, desired_shape = None, roll = 0):
        super(PhyioNetLoader_MIT_NIH, self).__init__()
        self.snr_dB = None
        self.snr = None
        torch.manual_seed(42)
        assert isinstance(num_samples,int), 'Number of samples must be an integer'
        assert isinstance(num_samples, int), 'Number of heartbeats must be an integer'

        self.num_beats = num_beats

        self.SNR_dB = SNR_dB

        self.num_samples = num_samples

        self.file_location = os.path.dirname(os.path.realpath(__file__))

        self.gpu = gpu

        self.roll = roll

        self.dev = torch.device('cuda:0' if torch.cuda.is_available() and gpu else 'cpu')

        folderName = self.file_location + '/../Datasets/PhysioNet/MIT-BIH_Arrhythmia_Database/'


        header_files = glob.glob(folderName +'*.hea')
        annotation_files = glob.glob(folderName + '*.atr')

        if not random_sample:
            sample_header = header_files[:num_sets]
            sample_annotation = annotation_files[:num_sets]
        else:
            sample_header = np.random.choice(header_files, num_sets, replace=False)
            sample_annotation = annotation_files[:num_sets]

        self.files = [wfdb.rdrecord(header_file[:-4]) for header_file in sample_header]
        self.annotation = [wfdb.rdann(annotation_file[:-4], 'atr' ) for annotation_file in sample_annotation]

        self.fs = self.files[0].fs
        self.num_channels = self.files[0].n_sig

        self.dataset = torch.tensor(np.array([file.p_signal for file in self.files]),dtype= torch.float32,device=
                                    torch.device('cpu')).mT


        # self.labels = torch.tensor(np.array([file.sample for file in self.annotation]),dtype= torch.float32)[0,1:]

        self.labels = [file.sample for file in self.annotation]

        shape = str(desired_shape) if desired_shape != None else ''

        CenteredDataFileName = f'CenteredData_snr_{SNR_dB}_shape_{shape}_samples_{num_samples}_sets_{num_sets}.pt'
        NoisyDataFileName = f'NoisyData_snr_{SNR_dB}_shape_{shape}_samples_{num_samples}_sets_{num_sets}.pt'

        if CenteredDataFileName in os.listdir(folderName):
            self.centerd_data = torch.load(folderName + CenteredDataFileName).to(self.dev)
            center_flag = False
        else:
            self.Center()
            center_flag = True

        if NoisyDataFileName in os.listdir(folderName):
            self.noisy_dataset = torch.load(folderName + NoisyDataFileName).to(self.dev)
            noisy_flag = False
        else:
            self.AddGaussianNoise(SNR_dB)
            noisy_flag = True

        if plot_sample:
            self.PlotSample()

        if desired_shape != None:
            intermediate = self.centerd_data
            intermediate_noisy = self.noisy_dataset

            while len(desired_shape) + 1 != len(intermediate.shape):
                intermediate = intermediate.unsqueeze(-1)
                intermediate_noisy = intermediate_noisy.unsqueeze(-1)

            permutation = [intermediate.shape[1:].index(x) + 1 for x in desired_shape]

            permutation = [0] + permutation

            self.centerd_data = intermediate.permute(permutation)
            self.noisy_dataset = intermediate_noisy.permute(permutation)

        if center_flag:
            torch.save(self.centerd_data, folderName + CenteredDataFileName)

        if noisy_flag:
            torch.save(self.noisy_dataset, folderName +  NoisyDataFileName)



    def Center(self):
        full_data = []
        for j,label in enumerate(self.labels):

            beat_indices_last = label[self.num_beats-1::self.num_beats]
            beat_indices_first = label[0::self.num_beats]
            intermediate = []


            last_index = 0
            num_waveforms = 0

            self.num_pad = []

            for i, index in enumerate(beat_indices_last):

                if self.num_beats == 1:
                    middle = int(last_index + int((index - last_index)/1))
                else:
                    middle = int( last_index + (index - beat_indices_first[i]) / 2)

                lower_index = middle - int(self.num_samples/2)
                upper_index = middle + int(self.num_samples/2)

                if lower_index >= 0 and upper_index < self.dataset.shape[-1]:

                    if self.num_samples > self.fs*self.num_beats:
                        num_pad = int((self.num_samples-self.fs)/2)
                        self.num_pad.append([num_pad,num_pad])
                        data = torch.nn.functional.pad(self.dataset[j,:,int(index)-int(self.fs/2):int(index) + int(self.fs/2)], (num_pad,num_pad),'replicate')
                        intermediate.append(data)

                    else:
                        intermediate.append(self.dataset[j,:,lower_index:upper_index])
                    num_waveforms+=1
                last_index = index

            centered_data  = torch.stack(intermediate,dim=1)
            permutation = [centered_data.shape.index(x) for x in (num_waveforms,self.num_channels,self.num_samples)]
            centered_data  = centered_data.permute(permutation)

            full_data.append(centered_data)

        full_data = self.centerd_data =  torch.cat(full_data,dim=0)
        return full_data


    def __getitem__(self, item: int) -> tuple:
        """
        Get the specified sample of noisy and clean data from the dataset
        :param item: index of the samples to get
        :return: A tuple of noisy and clean samples
        """

        if not 'segmented' in self.__dict__:

            if self.roll == 0:

                return self.noisy_dataset[item], self.centerd_data[item]

            else:
                shift = int(torch.randint(low=-self.roll, high=self.roll, size=(1,)))
                return torch.roll(self.noisy_dataset[item],shift,dims=1), torch.roll(self.centerd_data[item],shift,dims=1)

        else:
            return self._GetSegmented(item)


    def __len__(self) -> int:
        """
        Get the size of the dataset
        :return: Size of the dataset
        """
        return self.centerd_data.shape[0]

    def AddGaussianNoise(self, SNR_dB: float) -> None:
        """
        Add white gaussian noise to the split dataset
        :param SNR_dB: Signal to noise ration in decibel
        :return:
        """

        signals = self.centerd_data
        signal_power_dB = 10 * torch.log10(signals.var(-1) + signals.mean(-1) ** 2)

        noise_power_dB = signal_power_dB - SNR_dB
        noise_power = 10 ** (noise_power_dB / 20).unsqueeze(-1)

        noise = torch.normal(torch.zeros_like(signals), noise_power.repeat(1,1,  signals.shape[-1]))
        noise_power_num = 10 * np.log10(noise.var(-1) + noise.mean(-1) ** 2)
        print('SNR of actual signal', round( ( signal_power_dB - noise_power_num).mean().item(), 3), '[dB]')

        noisy_sample = signals + noise

        noisy_sample = noisy_sample.to(self.dev)
        self.centerd_data = self.centerd_data.to(self.dev)

        self.noisy_dataset = noisy_sample

    def PlotSample(self) -> None:
        """
        Plot a random sample of a random channel with noise
        :return: None
        """

        # Randomly select sample and channel
        sample = np.random.randint(0, self.centerd_data.shape[0])
        channel = np.random.randint(0, self.centerd_data.shape[1])
        sample = np.random.randint(0, self.centerd_data.shape[0])


        # Get the sample
        sample_signal = self.centerd_data[sample, channel,:]
        noisy_sample = self.noisy_dataset[sample, channel,:]

        # Time axis
        t = np.arange(start=0, stop=sample_signal.shape[0] / (self.fs), step=1 / (self.fs))

        # Plotting
        plt.plot(t, sample_signal[:].detach().cpu(), 'g', label='Ground Truth')
        plt.plot(t, noisy_sample[:].detach().cpu(), 'r', label='Noisy Signal', alpha=0.4)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [mV]')
        plt.title('MIT-BIH Dataset sample with additive GWN: SNR {} [dB]'.format(round(self.SNR_dB, 2)))
        plt.legend()
        # plt.savefig(self.file_location + '/../Plots/MIT-BIH-samples/MIT-BIH_sample_plot_snr_{}dB.pdf'.format(round(self.SNR_dB, 2)))
        plt.show()


    def GetData(self,num_batches):
        return self.noisy_dataset[:num_batches], self.centerd_data[:num_batches]

    def GetRolledData(self,num_batches,max_roll = 10):

        shift_obs = torch.empty((num_batches, self.noisy_dataset.shape[-2], self.noisy_dataset.shape[-1]))
        shift_state = torch.empty((num_batches, self.centerd_data.shape[-2],self.centerd_data.shape[-1]))

        for t in range(num_batches):
            shift = int(torch.randint(low=-max_roll, high=max_roll, size= (1,)))
            shift_obs[t] = torch.roll(self.noisy_dataset[t], shift)
            shift_state[t] = torch.roll(self.centerd_data[t], shift)

        return shift_obs, shift_state

    def SplitToSegments(self):

        self.segments = 0,160,200,self.fs

        self.SegmentedData = []
        self.SegmentedObservations = []


        for segment_start,segment_end in zip(self.segments[:-1],self.segments[1:]):
            self.SegmentedData.append(self.centerd_data[:,:,segment_start:segment_end])
            self.SegmentedObservations.append(self.noisy_dataset[:,:,segment_start:segment_end])


        self.segmented = True

    def _GetSegmented(self,key):

        states = [segment[key] for segment in self.SegmentedData]

        noise = [segment[key] for segment in self.SegmentedObservations]

        return noise,states













if __name__ == '__main__':
    dataset = PhyioNetLoader_AbdominalAndDirect()
    dataset.PlotSample(100)
