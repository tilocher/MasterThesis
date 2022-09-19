import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import scipy
import torch
from Filters.QRSDetectorOffline import QRSDetectorOffline
from ecgdetectors import Detectors


class RikDataset(Dataset):

    def __init__(self,desired_shape, signal_length_ratio = 0.7, num_files = 1, gpu = False, snr_dB = 0, preprocess = False, offset = 0 ):
        super(RikDataset, self).__init__()

        self.file_location = os.path.dirname(os.path.realpath(__file__))
        self.num_files = num_files
        self.num_beats = 1
        self.fs  = 500
        self.num_samples = self.T = int(self.fs * signal_length_ratio)
        self.preprocessed = preprocess
        self.num_channels = 12

        self.channels = self.m = self.n = 12

        mat_files = glob.glob(self.file_location + '/../Datasets/Rik/simulatedFromAdult_500Hz/traindata/*.mat')
        noise_files = glob.glob(self.file_location + '/../Datasets/Rik/simulatedFromAdult_500Hz/*.mat' )


        self.noise = []
        self.dataset = []
        self.labels = []

        detectors = Detectors(self.fs)

        self.gpu = gpu


        if gpu and torch.cuda.is_available():
            self.dev = torch.device('cuda:0')
        else:
            self.dev = torch.device('cpu')



        for mat_file in mat_files[offset:num_files+offset]:
            raw_data = loadmat(mat_file)['ECG']

            raw_data = torch.from_numpy(raw_data)
            peak_detection_data = raw_data[0]
            labels = detectors.wqrs_detector(peak_detection_data.numpy())

            self.dataset.append(torch.tensor(raw_data.clone().detach()[:,:8000], dtype=torch.float32))
            self.labels.append(labels)

        self.dataset = torch.stack(self.dataset)


        if preprocess:
            self.preprocess()

        self.Center()
        self.AddGaussianNoise(snr_dB)


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

        1

    def preprocess(self):
        # Keep unprocessed data
        self.noisy_unprocessed_data = self.noisy_dataset
        self.unprocessed_data = self.dataset
        nyq = 0.5 * self.fs
        order = 6
        cutoff = 15

        normal_cutoff = cutoff / nyq
        f1 = 0.5 / self.fs
        f2 = 90/ self.fs

        b, a = signal.butter(order, [f1 * 2, f2 * 2], btype='bandpass')
        self.dataset = torch.tensor(signal.lfilter(b, a, self.dataset),dtype= torch.float32)
        self.noisy_dataset = torch.tensor(signal.lfilter(b, a, self.noisy_dataset),dtype = torch.float32)





    # def AddGaussianNoise(self, SNR_dB: float) -> None:
    #     """
    #     Add white gaussian noise to the split dataset
    #     :param SNR_dB: Signal to noise ration in decibel
    #     :return:
    #     """
    #     np.random.seed(39)
    #
    #     signals = self.dataset
    #     self.noisy_dataset = torch.empty(signals.shape)
    #
    #     signal_power_dB = 10 * torch.log10(signals.var(-1) + signals.mean(-1) ** 2)
    #
    #     noise_power_dB = signal_power_dB - SNR_dB
    #     noise_power = 10 ** (noise_power_dB / 20).unsqueeze(-1)
    #
    #
    #     for i,signal in enumerate(signals):
    #         for j,channel in enumerate(signal):
    #
    #             noise = torch.normal(torch.zeros_like(channel),noise_power[i,j])
    #             noise_power_num = 10 * np.log10(noise.var() + noise.mean() ** 2)
    #
    #             noisy_signal = signals[i,j] + noise
    #
    #             self.noisy_dataset[i,j] = noisy_signal

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

        noise = torch.normal(torch.zeros_like(signals), noise_power.repeat(1, 1, signals.shape[-1]))
        noise_power_num = 10 * np.log10(noise.var(-1) + noise.mean(-1) ** 2)
        print('SNR of actual signal', round((signal_power_dB - noise_power_num).mean().item(), 3), '[dB]')

        noisy_sample = signals + noise

        noisy_sample = noisy_sample.to(self.dev)
        self.centerd_data = self.centerd_data.to(self.dev)

        self.noisy_dataset = noisy_sample



    # def Center(self):
    #     full_data = []
    #     full_data_noisy = []
    #     for j,label in enumerate(self.labels):
    #
    #         beat_indices_last = label[self.num_beats-1::self.num_beats]
    #         beat_indices_first = label[0::self.num_beats]
    #         intermediate = []
    #         intermediate_noisy = []
    #
    #
    #         last_index = 0
    #         num_waveforms = 0
    #         end_index = 0
    #
    #         self.num_pad = []
    #         self.Overlap = []
    #
    #         for i, index in enumerate(beat_indices_last):
    #
    #             if self.num_beats == 1:
    #                 middle = int(last_index + int((index - last_index)/1))
    #             else:
    #                 middle = int( last_index + (index - beat_indices_first[i]) / 2)
    #
    #             lower_index = middle - int(self.num_samples/2)
    #             upper_index = middle + int(self.num_samples/2)
    #
    #             if lower_index >= 0 and upper_index < self.dataset.shape[-1]:
    #
    #                 if self.num_samples > self.fs*self.num_beats:
    #                     num_pad = int((self.num_samples-self.fs)/2)
    #                     self.num_pad.append([num_pad,num_pad])
    #                     data = torch.nn.functional.pad(self.dataset[j,:,int(index)-int(self.fs/2):int(index) + int(self.fs/2)], (num_pad,num_pad),'replicate')
    #                     intermediate.append(data)
    #
    #                 else:
    #                     intermediate.append(self.dataset[j,:,lower_index:upper_index])
    #                     intermediate_noisy.append(self.noisy_dataset[j,:,lower_index:upper_index])
    #
    #                     self.Overlap.append(max(end_index-lower_index, 0))
    #
    #                 num_waveforms+=1
    #             last_index = index
    #             end_index = upper_index
    #
    #         centered_data  = torch.stack(intermediate,dim=0)
    #         centered_data_noisy = torch.stack(intermediate_noisy,dim=0)
    #         permutation = [centered_data.shape.index(x) for x in (num_waveforms,self.channels,self.num_samples)]
    #         centered_data  = centered_data.permute(permutation)
    #         centered_data_noisy = centered_data_noisy.permute(permutation)
    #
    #         full_data.append(centered_data)
    #         full_data_noisy.append(centered_data_noisy)
    #
    #     self.centerd_data =  torch.cat(full_data,dim=0).to(self.dev)
    #     self.noisy_dataset = torch.cat(full_data_noisy,dim = 0).to(self.dev)
    #
    #     return self.centerd_data , self.noisy_dataset

    def Center(self):
        full_data = []
        for j,label in enumerate(self.labels):

            beat_indices_last = label[self.num_beats-1::self.num_beats]
            beat_indices_first = label[0::self.num_beats]
            intermediate = []


            last_index = 0
            num_waveforms = 0
            end_index = 0

            self.num_pad = []
            self.Overlap = []

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
                        self.Overlap.append(max(end_index-lower_index, 0))

                    num_waveforms+=1
                last_index = index
                end_index = upper_index

            centered_data  = torch.stack(intermediate,dim=1)
            permutation = [centered_data.shape.index(x) for x in (num_waveforms,self.num_channels,self.num_samples)]
            centered_data  = centered_data.permute(permutation)

            full_data.append(centered_data)

        full_data = self.centerd_data =  torch.cat(full_data,dim=0)
        return full_data

    def __getitem__(self, item):

        return self.noisy_dataset[item] , self.centerd_data[item]


    def __len__(self):

        return self.centerd_data.shape[0]



if __name__ == '__main__':

    data = RikDataset()