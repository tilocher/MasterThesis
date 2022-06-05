# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from SystemModels.ECG_model import ECG_signal
import mne
from NeuraNets.ECG_Net import ECG_Net
from matplotlib import pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)

class ECG_Dataset(Dataset):
    def __init__(self, ecg_data,size):
        super(ECG_Dataset, self).__init__()

        data = ecg_data[0]
        for i in range(len(ecg_data)):
            if i != 0:
                data = torch.cat((data,ecg_data[i]), dim= 0)

        self.size = size
        target = torch.tensor(data)
        # flat_target = torch.flatten(target)
        self.target = torch.chunk(target,size,-1)


    def __len__(self):
        return  self.target.__len__()

    def __getitem__ (self, indx):
        return self.target[indx]


dataset = mne.io.read_raw_edf(r'C:\Users\Timur\Desktop\MasterThesis\Code\Unsupervised_EUSIPCO_22\Datasets\abdominal-and-direct-fetal-ecg-database-1.0.0\r01.edf')
# training_target = dataset.get_data()[1:]
training_target = dataset.get_data()[0]



ecg_data = ECG_Dataset([training_target], 100)





ECG_DataLoader = DataLoader(ecg_data, shuffle=True, batch_size=16)

Net = ECG_Net(ECG_DataLoader.batch_size)
Epochs = 10

loss_fn = torch.nn.L1Loss(reduction='sum')
def costum_loss(input,target):

    fft_input = torch.fft.fft(input)
    fft_target = torch.fft.fft(target)

    return loss_fn(fft_input,fft_target)
optimizer = torch.optim.Adam(Net.parameters(), lr = 1e-3)

# torch.autograd.set_detect_anomaly(True)

for e in range(Epochs):

    epoch_loss = 0.
    counter = 0

    for target in ECG_DataLoader:
        Net.zero_grad()
        Loss = 0.
        if target.shape[0] == ECG_DataLoader.batch_size:
            ssModel = ECG_signal(ECG_DataLoader.batch_size)

            a = Net(target.reshape(ECG_DataLoader.batch_size,1,4,-1))

            init_cond = target[:,0]#.mean(-1)

            generated = ssModel.GenerateBatch(init_cond.unsqueeze(-1), target.shape[-1], a)

            Loss = loss_fn(generated[:,2],target)#.mean(1))
            # Loss = costum_loss(generated[:, 2], target)  # .mean(1))
            Loss.backward()

            # torch.nn.utils.clip_grad_norm(Net.parameters(), 1)

            optimizer.step()
            # for name, param in Net.named_parameters():
            #     print(name, torch.isfinite(param.grad).all())

            epoch_loss += Loss
            counter += 1
            print('Batch Loss:', 10*torch.log10(Loss).item(), '[dB]')

            del ssModel, Loss,a , init_cond, target

            1

    print('Mean epoch loss:', 10*torch.log10(epoch_loss/counter).item(), '[dB]')
    # plt.plot(generated[0,1].detach().numpy())
    # plt.plot(generated[0,0].detach().numpy())
    plt.plot(generated[0,2].detach().numpy())
    plt.plot(target[0,:])

    plt.show()













