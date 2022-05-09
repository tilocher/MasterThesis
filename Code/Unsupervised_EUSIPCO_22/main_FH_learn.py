# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________


import os
import sys
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
import torch.nn as nn
from SystemModels.Linear_sysmdl import SystemModel
from SystemModels.Linear_sysmdl import SystemModel
from Extended_data import DataGen, DataLoader, DataLoader_GPU, Decimate_and_perturbate_Data, Short_Traj_Split
from filing_paths import BaseFolder
from Extended_data import N_E, N_CV, N_T, F, H, F_rotated, H_rotated, T, T_test, m1_0, m2_0, m, n
from NeuraNets.FH_Net import FH_Net
from datetime import datetime
from Pipelines.Pipeline_FH import Pipeline_FH

# if torch.cuda.is_available():
#     dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     print("Running on the GPU")
#
# else:
#dev = torch.device("cpu")
print("Running on the CPU")

print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
path_results = 'RTSNet/'

####################
### Design Model ###
####################
r_dB = 0
r2 = torch.tensor([10 ** (r_dB / 10)])
vdB = -20  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)
print("1/r2 [dB]: ", 10 * torch.log10(1 / r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1 / q2[0]))

# True model
r = torch.sqrt(r2)
q = torch.sqrt(q2)
sys_model = SystemModel(F, q, H, r, T, T_test)
sys_model.InitSequence(m1_0, m2_0)

# Mismatched model
sys_model_partialh = SystemModel(F, q, H_rotated, r, T, T_test)
sys_model_partialh.InitSequence(m1_0, m2_0)

###################################
### Data Loader (Generate Data) ###
###################################
dataFolderName = r'' + BaseFolder + '\\Simulations\\Linear_canonical' + '\\'
dataFileName = '{}x{}_rq{}{}_T{}.pt'.format(m, n, r_dB, (r_dB + vdB), T)


if not dataFileName in os.listdir(dataFolderName):
    print("Start Data Gen")
    DataGen(sys_model, dataFolderName + dataFileName, T, T_test, randomInit=False)

print("Data Load")
[train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(
     dataFolderName + dataFileName)
print("trainset size:", train_target.size())
print("cvset size:", cv_target.size())
print("testset size:", test_target.size())

### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

model = FH_Net()
model.Build(m,n)

TrainedNetFolder = 'TrainedNets\\'
path_results = TrainedNetFolder +  'FHNet/'

Pipeline = Pipeline_FH(strTime,TrainedNetFolder+'FHNet','FHNet')
Pipeline.setModel(model)
Pipeline.setssModel(sys_model)
Pipeline.setTrainingParams(10,20,1e-3,1e-6)
Pipeline.NNTrain(sys_model,cv_input,cv_target,train_input,train_target,path_results)




