# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

from Code.Unsupervised_EUSIPCO_22.NeuraNets.ECG_Net import ECG_Net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class SynteticECGDataset(Dataset):

    def __init__(self,input,target, time_slice):

        self.input = input.to(device)
        self.target = target[:,0].to(device)

        self.time_slice = time_slice

    def __getitem__(self, item):

        inp,tar = self.input[item].unsqueeze(0), self.target[item]

        return inp,tar

    def __len__(self):
        return len(self.input)

class ECGPipeline():

    def __init__(self, train_path_input,train_path_target
                 , cv_path_input, cv_path_target,
                 test_path_input,test_path_target):

        self.train_path_input = train_path_input
        self.cv_path_input = cv_path_input
        self.test_path_input = test_path_input

        self.train_path_target = train_path_target
        self.cv_path_target = cv_path_target
        self.test_path_target = test_path_target



    def run(self, batch_size = 32):

        # Load datasets
        train_input= torch.from_numpy(np.load(self.train_path_input)).float()
        cv_input = torch.from_numpy(np.load(self.cv_path_input)).float()
        test_input = torch.from_numpy(np.load(self.test_path_input)).float()

        train_target = torch.from_numpy(np.load(self.train_path_target)).float()
        cv_target = torch.from_numpy(np.load(self.cv_path_target)).float()
        test_target = torch.from_numpy(np.load(self.test_path_target)).float()


        # Create a Dataloader for each train, cv and test set
        time_steps = 500
        Train_DataSet = DataLoader(SynteticECGDataset(train_input,train_target, time_steps), batch_size= batch_size,shuffle= True)
        CV_DataSet = DataLoader(SynteticECGDataset(cv_input,cv_target,time_steps), batch_size= batch_size,shuffle= True)
        Test_DataSet = DataLoader(SynteticECGDataset(test_input,test_target, time_steps), batch_size= batch_size,shuffle= True)



        # Create Network
        sample = next(iter(Train_DataSet))[0].shape
        Network = ECG_Net(batch_size,device)
        Network.InitLSTM(batch_size, torch.randn((sample[0], sample[1], sample[2], Train_DataSet.dataset.time_slice ),device=device))

        # Start Epochs, and init metrics:
        Epochs = 50
        stride = 60

        loss_fn = torch.nn.BCELoss(reduction = 'mean')
        optimizer = torch.optim.Adam(Network.parameters(), lr= 1e-2, weight_decay= 1e-6)

        Epoch_training_loss = torch.empty(Epochs)
        Epoch_cv_loss = torch.empty(Epochs)
        Epoch_training_loss_dB = torch.empty(Epochs)
        Epoch_cv_loss_dB = torch.empty(Epochs)
        Epoch_training_accuracy = torch.empty(Epochs)
        Epoch_cv_accuracy = torch.empty(Epochs)

        Epoch_range = trange(Epochs, desc= 'Epoch, training acc: {} , cv acc: {}', leave = True)
        for e in Epoch_range:
            # Training Loop
            epoch_acc = []
            Epoch_loss = 0

            for train_inp, train_tar in Train_DataSet:
                # if not train_inp.shape[0] == batch_size:
                #     continue

                Network.ResetLSTM(train_inp.shape[0])

                window_start = 0
                window_end = Train_DataSet.dataset.time_slice

                correct = 0
                total = 0

                Loss = 0.

                while window_end < train_inp.shape[-1]:

                    prediction = Network(train_inp[:,:,:,window_start:window_end])
                    Loss += loss_fn(prediction,train_tar[:,:,window_end])
                    correct += torch.sum(prediction.argmax(-1) == train_tar[:,:,window_end].argmax(-1))
                    total += torch.numel(prediction)

                    window_start += stride
                    window_end += stride

                Loss.backward()
                optimizer.zero_grad()
                optimizer.step()

                Epoch_loss += Loss

                acc = correct/total
                Epoch_range.set_description('Epoch: {}, Training acc: {}, Training Loss: {}'.format(e,acc,Loss))
                epoch_acc.append(acc)

            with torch.no_grad():

                for cv_inp, cv_tar in CV_DataSet:
                    # if not cv_inp.shape[0] == batch_size:
                    #     continue

                    Network.ResetLSTM(cv_inp.shape[0])

                    window_start = 0
                    window_end = Train_DataSet.dataset.time_slice

                    correct = 0
                    total = 0

                    Loss = 0.

                    while window_end < cv_inp.shape[-1]:
                        prediction = Network(cv_inp[:, :, :, window_start:window_end])
                        Loss += loss_fn(prediction, cv_tar[:, :, window_end])
                        correct += torch.sum(prediction.argmax(-1) == cv_tar[:, :, window_end].argmax(-1))
                        total += torch.numel(prediction)

                        window_start += stride
                        window_end += stride


if __name__ == '__main__':


    path_inp = r'C:\Users\Timur\Desktop\MasterThesis\Code\Unsupervised_EUSIPCO_22\Datasets\Syntetic\train_input_100.npy'
    path_tar = r'C:\Users\Timur\Desktop\MasterThesis\Code\Unsupervised_EUSIPCO_22\Datasets\Syntetic\train_label_valued_binary100.npy'

    # tar = np.load(path_tar)
    # N_inx = np.where(tar  == 'N')
    # N_inx = np.where(tar == 'P')
    # Q_inx = np.where(tar == 'Q')
    # R_inx = np.where(tar == 'R')
    # S_inx = np.where(tar == 'S')
    # T_inx = np.where(tar == 'T')
    #
    # valued_tar = np.empty(tar.shape)
    # valued_tar[N_inx] = 0
    # valued_tar[Q_inx] = 1
    # valued_tar[R_inx] = 2
    # valued_tar[S_inx] = 3
    # valued_tar[T_inx] = 4
    # np.save(f'..//Datasets//Syntetic//train_label_valued{100}.npy', valued_tar)
    # 1

    # tar = np.load(path_tar)
    # non_zero = np.where(tar > 0)
    # tar[non_zero] = 1
    # np.save(f'..//Datasets//Syntetic//train_label_valued_binary{100}.npy', tar)
    #
    # 1

    pipe = ECGPipeline(path_inp,path_tar,path_inp,path_tar,path_inp,path_tar)
    pipe.run()