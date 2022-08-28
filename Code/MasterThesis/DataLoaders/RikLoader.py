from scipy.io import loadmat
from torch.utils.data import Dataset
import os
import glob

class RikDataset(Dataset):

    def __init__(self):
        super(RikDataset, self).__init__()

        self.file_location = os.path.dirname(os.path.realpath(__file__))

        mat_files = glob.glob(self.file_location + '/../Datasets/Rik/simulatedFromAdult_500Hz/traindata/*.mat')

        test = loadmat(mat_files[0])
        1
