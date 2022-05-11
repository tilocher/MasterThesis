# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from SystemModels.ECG_model import ECG_signal
import mne
from NeuraNets.ECG_Net import ECG_Net

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
training_target = dataset.get_data()[1:]


ecg_data = ECG_Dataset([training_target], 100)




ECG_DataLoader = DataLoader(ecg_data, shuffle=True, batch_size=16)

Net = ECG_Net(ECG_DataLoader.batch_size)
Epochs = 10

for e in range(Epochs):


    for target in ECG_DataLoader:

        a = Net(target.reshape(ECG_DataLoader.batch_size,1,4,-1))
        1










