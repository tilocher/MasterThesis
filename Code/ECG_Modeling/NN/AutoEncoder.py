# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import torch
from torch import nn
import numpy as np
import time

def DataLoader(N):
    import h5py

    # N = 20
    input_length = 512
    data = np.empty((6, 1))
    lenghts_of_files = []
    lenghts_of_mod_files = []
    num_peaks_ls = []
    N_peaks = 1000
    start_time = time.time()

    file_list = ["%02d" % x for x in range(N + 1)]

    for i in file_list:
        with h5py.File('..\\Datasets\\data_Mehdi\\meas0' + str(i) + '.data', 'r') as hdf:
            interm_data = np.empty((6, 1))
            ls = list(hdf.keys())
            new_data = hdf.get('fetalSignal')
            new_data = np.array(new_data)
            new_data = new_data.T
            lenghts_of_files.append(new_data.shape[1])
            print("Length of file", i, "=", new_data.shape[1])
            print(new_data.shape)
            new_peaks = hdf.get('fRpeaks')
            new_peaks = [x for x in new_peaks if str(x) != 'nan']
            new_peaks = np.array(new_peaks)
            new_peaks = new_peaks.T
            new_peaks = new_peaks.squeeze().astype(int)
            # Added this one line below
            new_peaks = new_peaks[:N_peaks]
            # num_peaks = new_peaks.shape[0] changed to
            num_peaks = N_peaks
            if new_peaks.shape[0] < N_peaks:
                num_peaks = new_peaks.shape[0]
                for j in range(new_peaks.shape[0]):
                    peak_indice = int(new_peaks[j])
                    if peak_indice - 235 > 0 and peak_indice + 236 < new_data.shape[1]:
                        concat = new_data[:, peak_indice - 235:peak_indice + 236]
                        concat = np.pad(concat, ((0, 0), (20, 21)), 'edge')
                        interm_data = np.concatenate((interm_data, concat), axis=1)
                    else:
                        num_peaks = num_peaks - 1
            else:
                for j in range(new_peaks.shape[0]):
                    peak_indice = int(new_peaks[j])
                    if peak_indice - 235 > 0 and peak_indice + 236 < new_data.shape[1]:
                        concat = new_data[:, peak_indice - 235:peak_indice + 236]
                        concat = np.pad(concat, ((0, 0), (20, 21)), 'edge')
                        interm_data = np.concatenate((interm_data, concat), axis=1)
                    else:
                        num_peaks = num_peaks - 1

            num_peaks_ls.append(num_peaks)
            print("Number of peaks in file", i, "=", num_peaks)

            interm_data = interm_data[:, 1:]
            print("Data of file", i, "is of shape =", interm_data.shape)
            lenghts_of_mod_files.append(interm_data.shape[1])
            data = np.concatenate((data, interm_data), axis=1)
            print("NEXT")

    data = data[:, 1:]



    print("--- Data loading time is %s seconds ---" % (time.time() - start_time))

    return data


def average_ecg(data):

    from scipy import signal

    N_avg = 30

    x = data.shape[0]
    y = data.shape[1]

    augmented_data = np.concatenate((data,data[:,-512*N_avg:-512]), axis=1)
    one_positions = np.arange(0,512*N_avg,512)
    window = np.zeros(512*N_avg)
    window[one_positions] = 1

    data_avg = np.zeros((x,y))
    for i in range(6):
        intermed = signal.convolve(data[i,:],window)/N_avg
        data_avg[i] = intermed[512*(N_avg-1):512*(N_avg-1)+y]

    return data_avg


def BatchData(array, input_length = 512):
    '''
      Modify array shape from (6, n of samples) -----> (n of heart beats, 6, 512, 1)
      n of heart beats = n of samples/512.
      512 = number of samples per hear beat.
    '''
    if array.ndim != 2:
        raise Exception("Sorry, the number of dimension of the given array must be 2.")
    if array.shape[0] != 6:
        raise Exception("Sorry, the array must contain 6 channels.")

    dim = array.shape
    print("Modify Shape func : the given array is of dim =", dim)
    n_hbeats = int(dim[1] / 512)
    array = np.stack(np.split(array, np.arange(input_length, dim[1], input_length), axis=1))
    array = array.reshape(n_hbeats, 6 * input_length, 1)
    array = array.reshape(array.shape[0], 6, 512, 1)


    return torch.tensor(array.mean(1).reshape(-1,1,input_length),dtype= torch.float32)


class AutoEncoder(torch.nn.Module):

    def __init__(self,input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        super(AutoEncoder, self).__init__()
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim

        self.encoder = None
        self.decoder = None
        self.model = None

        self._encoder_output_dim = None
        self._num_conv_layers = len(conv_filters)
        self._model_input = None

        self.Build()


    def Build(self):

        self.Build_Encoder()
        # self.Build_Decoder()

    def Build_Encoder(self):
        self.Encoder = Encoder(self.input_shape,self.conv_filters,self.conv_kernels,
                               self.conv_strides,self.latent_space_dim)

class Encoder(torch.nn.Module):

    def __init__(self,input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):

        super(Encoder, self).__init__()
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self._num_conv_layers = len(conv_filters)

        self.Build()

    def Build(self):


        self.ConvLayers = nn.ModuleList([
            nn.Conv2d(in_channels= 6 ,out_channels= 6,
                      kernel_size= self.conv_kernels[i], stride= self.conv_strides[i],
                      ) for i in range(self._num_conv_layers)
        ])

    def forward(self,x):

        for Conv in self.ConvLayers:
            x = Conv(x)
        return x


if __name__ == '__main__':

    # data = DataLoader(1)
    # data = np.nan_to_num(data)
    #
    # data_avg = average_ecg(data)
    #
    # data = BatchData(data)[200:203]
    # data_avg = BatchData(data_avg)[200:203]
    conv_filters = (40, 20, 20, 20, 20, 40),
    conv_kernels = ([6, 40], [6, 40], [6, 40], [6, 40], [6, 40], [6, 40]),
    conv_strides = ([1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2])

    from keras.layers import  Conv2D
    conv_layer = Conv2D(
        filters=40,
        kernel_size=[6,40],
        strides=conv_strides[0],
        padding="same",
        name=f"encoder_conv_layer_{0}"
    )
    LATENT_SPACE = 25

    AE = AutoEncoder(input_shape=(6, 512, 1),
        conv_filters=(40, 20, 20, 20, 20, 40),
        conv_kernels=([6,40], [6,40], [6,40], [6,40], [6,40], [6,40]),
        conv_strides=([1,2], [1,2], [1,2], [1,2], [1,2], [1,2]),
        latent_space_dim = LATENT_SPACE)

    AE.Encoder(torch.randn((32, 1, 6,512 )))

    print('ola')