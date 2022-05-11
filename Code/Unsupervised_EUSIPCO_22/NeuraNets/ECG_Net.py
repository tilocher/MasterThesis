# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import torch
import torch.nn as nn

class ECG_Net(nn.Module):

    def __init__(self,batch_size):
        super(ECG_Net, self).__init__()

        in_channels_1 = 1
        out_channels_1 = 5

        #

        in_channels_2 = out_channels_1
        out_channels_2 = out_channels_1

        self.conv1 = nn.Conv2d(in_channels= in_channels_1, out_channels= out_channels_1,kernel_size=(1,3), dilation= 1)
        self.conv2 = nn.Conv2d(in_channels= in_channels_1, out_channels= out_channels_1,kernel_size=(1,3), dilation= 2)
        self.conv3 = nn.Conv2d(in_channels= in_channels_1, out_channels= out_channels_1,kernel_size=(1,3), dilation= 4)

        self.conv4 = nn.Conv2d(in_channels= in_channels_2, out_channels=out_channels_2, kernel_size=(1, 3), dilation=1)
        self.conv5 = nn.Conv2d(in_channels= in_channels_2, out_channels=out_channels_2, kernel_size=(1, 3), dilation=2)
        self.conv6 = nn.Conv2d(in_channels= in_channels_2, out_channels=out_channels_2, kernel_size=(1, 3), dilation=4)

        self.batch_size = batch_size
        self.dropout = nn.Dropout(0.2)
        self.Prelu = nn.PReLU()
        self.BatchNorm = nn.BatchNorm2d(out_channels_1)
        self.Flatten = nn.Flatten()
        self.MPool = nn.MaxPool2d((4,10))

        self.LSTM = nn.LSTM(input_size=896, hidden_size= 1000, batch_first= True)
        self.h0 = torch.randn((1,batch_size,1000))
        self.r0 = torch.randn((1,batch_size, 1000))

        self.outlayer = nn.Linear(5000,13)

    def forward(self,x):

        # x1 = nn.functional.pad(self.conv1(x), [int(self.conv1.dilation[0]),int(self.conv1.dilation[0])])
        # x2 = nn.functional.pad(self.conv2(x), [int(self.conv2.dilation[0]),int(self.conv2.dilation[0])])
        # x3 = nn.functional.pad(self.conv3(x), [int(self.conv3.dilation[0]),int(self.conv3.dilation[0]) ])

        x1 = self.Prelu(self.conv1(x))
        x2 = self.Prelu(self.conv2(x))
        x3 = self.Prelu(self.conv3(x))

        x4 = self.BatchNorm(self.conv4(x1))
        x5 = self.BatchNorm(self.conv5(x2))
        x6 = self.BatchNorm(self.conv6(x3))

        x4 = self.MPool(x4)
        x5 = self.MPool(x5)
        x6 = self.MPool(x6)

        x4 = self.Flatten(x4)
        x5 = self.Flatten(x5)
        x6 = self.Flatten(x6)

        x = torch.cat((x4,x5,x6),dim=1)
        x = x.reshape(self.batch_size,5,-1)

        x,(self.h0, self.r0) = self.LSTM(x,(self.h0,self.r0))
        x = self.Flatten(x)
        x = self.Prelu(x)
        x = self.outlayer(x)
        return x

        # conv_out = x1 + x2 + x3 + x
        1

