# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import numpy as np
import torch
import torch.nn as nn



def Conv2DOutDim(Conv_layer, Input):

    if len(Input.shape) == 4:

        input_shape = Input.shape[1:]

    elif len(Input.shape) > 4 or len(Input.shape) < 3:

        ValueError('Shape is either to big or too small')

    else:

        input_shape = Input.shape

    C_in = input_shape[0]
    H_in = input_shape[1]
    W_in = input_shape[2]

    padding = Conv_layer.padding
    dilation = Conv_layer.dilation
    kernel_size = Conv_layer.kernel_size
    stride = Conv_layer.stride

    H_out = np.floor((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) -1 ) / stride[0] + 1)
    W_out = np.floor((W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) -1 ) / stride[1] + 1)

    return int(H_out), int(W_out)

class InceptionNetwork(nn.Module):

    def __init__(self,device):
        super(InceptionNetwork, self).__init__()
        in_channels_1 = 1
        out_channels_1 = 1

        in_channels_2 = out_channels_1
        out_channels_2 = 1

        self.conv1 = nn.Conv2d(in_channels=in_channels_1, out_channels=out_channels_1, kernel_size=(1, 3), dilation=(1,1),device=device)
        self.conv2 = nn.Conv2d(in_channels=in_channels_1, out_channels=out_channels_1, kernel_size=(1, 3), dilation=(1,4),device=device)
        self.conv3 = nn.Conv2d(in_channels=in_channels_1, out_channels=out_channels_1, kernel_size=(1, 3), dilation=(1,8),device=device)

        self.conv4 = nn.Conv2d(in_channels=in_channels_2, out_channels=out_channels_2, kernel_size=(1, 3), dilation=(1,1),device=device)
        self.conv5 = nn.Conv2d(in_channels=in_channels_2, out_channels=out_channels_2, kernel_size=(1, 3), dilation=(1,4),device=device)
        self.conv6 = nn.Conv2d(in_channels=in_channels_2, out_channels=out_channels_2, kernel_size=(1, 3), dilation=(1,8),device=device)

        self.out_conv = nn.Conv2d(in_channels=in_channels_2, out_channels=out_channels_2, kernel_size=(1, 3), dilation=(1,1),device=device)

        self.Prelu = nn.PReLU()
        self.BatchNorm = nn.BatchNorm2d(out_channels_1)
        self.MPool = nn.MaxPool2d((1, 4))

    def forward(self,x):

        x1 = self.Prelu(self.conv1(x))
        x2 = self.Prelu(self.conv2(x))
        x3 = self.Prelu(self.conv3(x))

        x1 = self.BatchNorm(self.conv4(x1))
        x2 = self.BatchNorm(self.conv5(x2))
        x3 = self.BatchNorm(self.conv6(x3))

        x = torch.cat((x,x1,x2,x3),dim=-1)

        x = self.Prelu(self.out_conv(x))
        x = self.MPool(x)
        return x

class ECG_Net(nn.Module):

    def __init__(self,batch_size, device):
        super(ECG_Net, self).__init__()

        self.device = device

        self.Incpetions = torch.nn.ModuleList([InceptionNetwork(self.device).to(self.device) for i in range(4)])

        self.batch_size = batch_size
        self.dropout = nn.Dropout(0.2)
        self.Flatten = nn.Flatten()
        self.Relu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()


    def InitLSTM(self, batch_size,sample):

        sample_out = self.try_conv(sample)
        lstm_size = sample_out.shape[-1]
        num_channels = sample_out.shape[-2]
        num_layers = self.num_layers = 3
        hidden_size = self.hidden_size = 1000

        self.LSTM = nn.LSTM(input_size=lstm_size, hidden_size=hidden_size, batch_first=True,
                            num_layers= num_layers, device=self.device)
        self.h0 = torch.randn((num_layers, batch_size, hidden_size), device=self.device)
        self.r0 = torch.randn((num_layers, batch_size, hidden_size), device=self.device)
        self.hidden = nn.Linear(hidden_size * num_channels, hidden_size, device=self.device)

        self.outlayer = nn.Linear(hidden_size , 5, device=self.device)

    def try_conv(self,sample):

       x = sample

       for Incep in self.Incpetions:

            x = Incep(x)

       return x


    def ResetLSTM(self,batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        self.h0 = hidden.data
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        self.r0 = hidden.data



    def forward(self,x):

        x = self.try_conv(x).squeeze()
        x = self.dropout(x)
        x,(self.h0, self.r0) = self.LSTM(x,(self.h0,self.r0))
        x = self.Flatten(x)
        x =  self.hidden(x)
        x = self.Relu(x)
        x = self.outlayer(x)
        x = self.Sigmoid(x)
        return x


