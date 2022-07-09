# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import torch
from torch import nn
from numpy import ceil,log,power
import time



def next_power_of_two(x):
    return int(power(2, ceil(log(x) / log(2))))

class AutoEncoder(torch.nn.Module):

    def __init__(self,signal_length,
                 num_channels,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 conv_dilation,
                 latent_space_dim):

        if torch.cuda.is_available():
            self.dev = torch.device("cuda:0")
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            self.dev = torch.device("cpu")
            print("using CPU!")


        super(AutoEncoder, self).__init__()
        self.signal_length = signal_length
        self.num_channels = num_channels
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.conv_dilation = conv_dilation
        self.latent_space_dim = latent_space_dim

        self.encoder = None
        self.decoder = None
        self.model = None

        self._encoder_output_dim = None
        self._num_conv_layers = len(conv_filters)
        self._model_input = None

        self.Build()


    def Build(self):

        self.Encoder = Encoder(self.signal_length, self.num_channels, self.conv_filters, self.conv_kernels,
                               self.conv_strides, self.conv_dilation, self.latent_space_dim, self.dev)
        self.Decoder = Decoder(self.Encoder.Linear_dimension,self.Encoder.Last_conv_dim, self.conv_filters, self.conv_kernels,
                               self.conv_strides, self.conv_dilation, self.latent_space_dim, self.dev)

    def forward(self,x):

        x = self.Encoder(x)

        x = self.Decoder(x)

        return x

class Encoder(torch.nn.Module):

    def __init__(self,signal_length,
                 num_channels,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 conv_dilation,
                 latent_space_dim,
                 dev):

        self.dev = dev

        super(Encoder, self).__init__()

        self.signal_length = signal_length
        self.num_channels = num_channels
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.conv_dilation = conv_dilation
        self.latent_space_dim = latent_space_dim


        self.Build()


    # Build the encoder network
    def Build(self):

        # Initialize Batchnorm layer and relu and flatten
        self.BatchNormLayers = nn.ModuleList([nn.BatchNorm2d(filters,device=self.dev) for filters in self.conv_filters])
        self.ReLU = nn.ModuleList([nn.ReLU() for _ in self.conv_filters])
        self.Flatten = nn.Flatten()


        # Create modules for conv layers and padding layers
        self.ConvLayers = nn.ModuleList()
        self.PaddingLayers = nn.ModuleList()

        # Init necessary dimensions
        input_size = self.signal_length
        in_channels = (1,) + self.conv_filters[:-1]


        for in_channel,out_channel,kernel,stride,dilation in\
                zip(in_channels,self.conv_filters,self.conv_kernels, self.conv_strides,self.conv_dilation):

            # Each layer should shrink the signal length to the next power of two
            target_size = next_power_of_two(input_size / 2)

            # Calculate necessary padding
            pad = int(((target_size - 1) * stride[0] - input_size + dilation[0] * (
                        kernel[0] - 1) + 1)/2)
            padding = (1,0,pad,pad)

            # Init Torch layers
            Pad = nn.ConstantPad2d(padding,0)

            Conv = nn.Conv2d(in_channels= in_channel, out_channels = out_channel,
                          kernel_size= kernel, stride= stride,dilation= dilation,device= self.dev)

            # Add to the Module list
            self.ConvLayers.append(Conv)
            self.PaddingLayers.append(Pad)

            # Update the input sizes
            input_size = target_size

        self.Last_conv_dim = (out_channel,target_size,self.num_channels)
        self.Linear_dimension = int(target_size * out_channel * self.num_channels)
        self.LinearLayer = nn.Linear(self.Linear_dimension, self.latent_space_dim,device=self.dev)

    # Define Forward pass of the encoder
    def forward(self,x):

        x = nn.functional.normalize(x,dim = 2)

        for Conv,Pad,BNorm, Relu in zip(self.ConvLayers,self.PaddingLayers,self.BatchNormLayers,self.ReLU):

            x = Pad(x)

            x = Conv(x)

            x = Relu(x)

            x = BNorm(x)

        x = self.Flatten(x)

        x = self.LinearLayer(x)

        return x


class Decoder(nn.Module):


    def __init__(self, Linear_size,
                 conv_dim,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 conv_dilation,
                 latent_space_dim ,
                 dev):

        self.dev = dev


        super(Decoder, self).__init__()

        self.Linear_size = Linear_size
        self.conv_dim = conv_dim
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.conv_dilation = conv_dilation
        self.latent_space_dim = latent_space_dim

        self.Build()

    # Build the Decoder network
    def Build(self):

        # Init necessary dimensions
        input_size = self.conv_dim[-2]
        in_channels = (self.conv_filters[-1],) + self.conv_filters[:-1]
        out_channels = self.conv_filters[:-1] + (1,)

        # Initialize Batchnorm layer and relu and flatten
        self.BatchNormLayers = nn.ModuleList([nn.BatchNorm2d(filters,device=self.dev) for filters in out_channels])
        self.ReLU = nn.ModuleList([nn.ReLU() for _ in self.conv_filters])
        self.Flatten = nn.Flatten()
        self.LinearLayer = nn.Linear(self.latent_space_dim,self.Linear_size,device=self.dev)

        # Create modules for conv layers and padding layers
        self.DeconvLayers = nn.ModuleList()
        self.PaddingLayers = nn.ModuleList()


        for in_channel, out_channel, kernel, stride, dilation in \
                zip(in_channels, out_channels, self.conv_kernels, self.conv_strides, self.conv_dilation):

            # Each layer should shrink the signal length to the next power of two
            target_size = next_power_of_two(input_size * 2)

            # Calculate necessary padding
            pad = int(((input_size-1)*stride[0]- target_size + dilation[0]*(kernel[0]-1) +1 )/2)
            padding = (-1, 0, -pad, -pad)

            # Init Torch layers
            Pad = nn.ConstantPad2d(padding, 0)

            DeConv = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel,
                             kernel_size=kernel, stride=stride, dilation=dilation,device=self.dev)

            # Add to the Module list
            self.DeconvLayers.append(DeConv)
            self.PaddingLayers.append(Pad)


            # Update the input sizes
            input_size = target_size

    # Define Forward pass of the decoder

    def forward(self, x):

        x = self.LinearLayer(x)

        x = x.reshape((-1,)+ self.conv_dim)

        for Conv, Pad, BNorm, Relu in zip(self.DeconvLayers, self.PaddingLayers, self.BatchNormLayers, self.ReLU):

            x = Conv(x)

            x = Pad(x)

            x = Relu(x)

            x = BNorm(x)

        return x


# class AE_Pipeline()

if __name__ == '__main__':

    # data = DataLoader(1)
    # data = np.nan_to_num(data)
    #
    # data_avg = average_ecg(data)
    #
    # data = BatchData(data)[200:203]
    # data_avg = BatchData(data_avg)[200:203]
    # conv_filters = (40, 20, 20, 20, 20, 40),
    # conv_kernels = ([6, 40], [6, 40], [6, 40], [6, 40], [6, 40], [6, 40]),
    # conv_strides = ([1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2])
    #
    # from keras.layers import  Conv2D
    # conv_layer = Conv2D(
    #     filters=40,
    #     kernel_size=[6,40],
    #     strides=conv_strides[0],
    #     padding="same",
    #     name=f"encoder_conv_layer_{0}"
    # )
    # LATENT_SPACE = 25
    #
    # AE = AutoEncoder(num_channels= 2,
    #                     signal_length = 512,
    #                     conv_filters=(40, 20, 20, 20, 20, 40),
    #                     conv_kernels=((40,2), (40,2), (40,2), (40,2), (40,2), (40,2)),
    #                     conv_strides=((2,1), (2,1), (2,1), (2,1), (2,1), (2,1)),
    #                     conv_dilation = ((1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),
    #                     latent_space_dim = LATENT_SPACE)
    #
    # AE(torch.randn((32,1,512,2)))




    from Code.ECG_Modeling.DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH

    snr = 6

    ts = 360

    loader = PhyioNetLoader_MIT_NIH(1,1, 1*ts,SNR_dB=snr,random_sample=False)
    split = torch.utils.data.random_split(loader,[int(0.7*len(loader)) ,  len(loader)- int(0.7*len(loader))])

    num_batches = 1000

    obs,state = loader.GetData(num_batches)



