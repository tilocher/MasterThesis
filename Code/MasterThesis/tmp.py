import numpy as np
import torch

from matplotlib import  pyplot as plt
from DataLoaders.PhysioNetLoader import PhyioNetLoader_MIT_NIH
import pandas

if __name__ == '__main__':
    fig = plt.figure(frameon=False, dpi = 200)
    # fig.set_size_inches(10, )
    markersize = 20

    loader = PhyioNetLoader_MIT_NIH(1,2,650,0,gpu=False,plot_sample=False)

    sample = loader[7]
    # t = np.arange(start=0, stop=sample_signal.shape[0] / (self.fs), step=1 / (self.fs))

    # ax = plt.Axes(fig, [0., 0., 1., 1.])

    # ax = plt.axes(projection = '3d')
    ax = plt.axes()
    # ax.set_axis_off()
    fig.add_axes(ax)
    # plt.plot(0,1,'*',color = '#FF3333',markersize = markersize)
    # plt.plot(-0.1, 0.5, '*', color='#CC99FF',markersize = markersize)
    # plt.plot(0.11, 0.5, '*', color='#CC99FF',markersize = markersize)
    # plt.plot(-0.2, 0.1, '*', color='#FF99CC',markersize = markersize)
    # plt.plot(0.2, 0.1, '*', color='#FF99CC',markersize = markersize)
    # plt.xlim([-0.4,0.4])


    y = torch.linspace(0,1,sample[1].T[:,0].shape[0])
    x = torch.zeros_like(y)

    # x = np.sin(t)
    # y = np.cos(t)

    # ax.view_init(-20,-70)

    torch.random.manual_seed(69)

    noise = 0.04*torch.randn(size=sample[1].T[:,0].shape)
    jump = 4

    plt.plot(sample[1].T[::jump,0] + noise[::jump],'o--',color='#1e76b4')
    # ax.plot3D(x,y,sample[1].T[:,0])
    # plt.axis([0,1,0.3,0.9])

    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.savefig('beat_sideways.svg',transparent = True)

    plt.show()
    def conv(x):
        if isinstance(x,bytes):
            return 0
        else:
            return x



    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ###################### Plot snr vs test loss result em #############################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    # #
    # a = pandas.read_csv(r"C:\Users\Timur\Downloads\wandb_export_2022-07-19T18_48_36.202+02_00.csv",delimiter=',')
    # a = a.values[:,1:3]
    # perm = []
    #
    # y1 = a[:7,1]
    # y2 = a[7:14,1]
    # y3 = a[14:,1]
    #
    # x1 = a[:7,0]
    # x1[[2,5]] = x1[[5,2]]
    # x1[[3,5]] = x1[[5,3]]
    #
    # y1[[2, 5]] = y1[[5, 2]]
    # y1[[3, 5]] = y1[[5, 3]]
    # x2 = a[7:14, 0]
    # x3 = a[14:,0]
    #
    # noise_floor = [-4.35,-7.35,-10.35,-13.35,-16.35,-20.35, -30.35]
    # plt.plot(x1,noise_floor,'*--', label = 'Noise Floor',color = 'r')
    # plt.plot(x1, y1, '*--', label='Taylor + EM', color='blue')
    # plt.plot(x2, y2, '*--', label='$ f= \mathbb{I} + EM$', color='orange')
    # plt.plot(x3, y3, '*--', label='Taylor Prior', color='c')
    # plt.grid()
    # plt.legend()
    # plt.xlabel('SNR [dB]')
    # plt.ylabel('MSE Loss [dB]')
    # plt.title('MSE loss for different SNRs for different models')
    # # plt.savefig('EM_losses_plot_taylor_em.pdf')
    #
    #
    # plt.show()

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    #
    # a = pandas.read_csv(r"C:\Users\Timur\Downloads\wandb_export_2022-07-19T20_54_06.770+02_00.csv",delimiter=',')
    # a = a.values[:, 1:3]
    # plt.plot(a[:,0],a[:,1])
    # plt.show()

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ########################### Plot EM results ##############################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################


    # import torch
    #
    # em_taylor = torch.load(r"C:\Users\Timur\Desktop\MasterThesis\Code\MasterThesis\log\runs\EM_Taylor\20_07___10_55\Logs\Pipelines\Pipelines.pt")
    #
    # prior = torch.load(r"C:\Users\Timur\Desktop\MasterThesis\Code\MasterThesis\log\runs\EM_Taylor\19_07___21_07\Logs\Pipelines\Pipelines.pt")
    #
    # em_id = torch.load(r"C:\Users\Timur\Desktop\MasterThesis\Code\MasterThesis\log\runs\EM_Taylor\19_07___21_55\Logs\Pipelines\Pipelines.pt")
    #
    # np.random.seed(42)
    #
    #
    # sampleNR = np.random.randint(0,len(prior.TestLoader))
    # sampleNR = 0
    #
    # sample = em_id.KalmanSmoother.Smoothed_State_Means[sampleNR,:,0,0]
    #
    # # plt.plot(em_taylor.KalmanSmoother.Filtered_State_Means[sample,:,0,0], color = 'b', label = 'EM + Taylor')
    # # plt.plot(prior.KalmanSmoother.Filtered_State_Means[sample,:,0,0])
    # # plt.plot(em_id.KalmanSmoother.Filtered_State_Means[sample,:,0,0])
    #
    #
    # obs,state = prior.TestLoader[sampleNR]
    #
    # obs = obs[0,:,0]
    # state = state[0,:,0]
    #
    # # plt.plot(obs[0,:,0],alpha = 0.4)
    # # plt.plot(state[0,:,0],color = 'g', label = 'Ground Truth')
    # plt.grid()
    # # plt.legend()
    #
    # t = np.arange(start=0, stop=sample.shape[0] / (360), step=1 / (360))
    # fig, ax = plt.subplots(dpi=200)
    #
    # ax.plot(t, sample, label= '$\mathbb{I}$ + em', color='b')
    # ax.plot(t, state, label= 'Ground Truth', color='g', alpha=0.8)
    # # ax.plot(t, obs, label='Observations', color='r', alpha=0.3)
    #
    # ax.grid()
    # ax.legend()
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Amplitude [mV]')
    #
    # axins = ax.inset_axes([0.05, 0.5, 0.4, 0.4])
    # axins.plot(t, sample, color='b')
    # axins.plot(t, state, color='g',alpha = 0.8)
    # # axins.plot(t, obs, label='Observations', color='r', alpha=0.8)
    # axins.get_xaxis().set_visible(False)
    # axins.get_yaxis().set_visible(False)
    #
    # # x1, x2, y1, y2 = 0.4, 0.6, torch.min(sample).item(), \
    # #                  torch.max(torch.max(sample),torch.max(state)).item()
    #
    # x1, x2, y1, y2 = 0.4, 0.6, torch.min(obs).item(), \
    #                  torch.max(obs).item()
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])
    # axins.grid()
    #
    # ax.indicate_inset_zoom(axins, edgecolor="black")
    #
    # plt.title('Filtered ECG Sample for a SNR of 6[dB]')
    # # plt.savefig('Filtered_em.pdf')
    #
    # plt.show()
    # 1

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ################# Plot AE results ##################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

#
    # import torch
    #
    # from log.BaseLogger import LocalLogger,WandbLogger
    # import yaml
    # from yaml.loader import SafeLoader
    #
    # config = yaml.load(open('../Configs/EM_taylor.yaml'), Loader=SafeLoader)
    #
    # Logger = LocalLogger('EM_Taylor', BaseConfig=config) if not config['wandb'] else \
    #     WandbLogger(name='EM_Taylor', group='SNR_sweep', BaseConfig=config)
    #
    # config = Logger.GetConfig()
    #
    # snr = config['snr']
    #
    # signal_length = config['signal_length']
    #
    # UseWandb = config['wandb']
    #
    # loader = PhyioNetLoader_MIT_NIH(1, 1, signal_length, SNR_dB=snr, random_sample=False, gpu=False,
    #                                 plot_sample=False, desired_shape=(1, signal_length, 2), roll=0)
    # np.random.seed(42)
    #
    #
    # N_train = int(0.8 * len(loader)) # int(0.05 * len(loader))
    # N_test = len(loader) - N_train
    #
    # dev = torch.device('cpu')
    # torch.random.manual_seed(42)
    #
    # Train_Loader, Test_Loader = torch.utils.data.random_split(loader, [N_train, N_test],
    #                                                           generator=torch.Generator())
    #
    # # N_test = 500
    #
    # # Test_Loader.indices = list(np.random.choice(Test_Loader.indices, 500, replace = False))
    #
    # Test_Loader.indices = Test_Loader.indices[:500]
    #
    # roll = torch.load(r"C:\Users\Timur\Desktop\MasterThesis\Code\MasterThesis\log\runs\AutoEncoder_SNR_sweep_roll\19_07___22_30\Logs\Pipelines\Pipelines.pt")
    #
    # no_roll = torch.load(r"C:\Users\Timur\Desktop\MasterThesis\Code\MasterThesis\log\runs\AutoEncoder_SNR_sweep_roll\19_07___22_19\Logs\Pipelines\Pipelines.pt")
    #
    # # em_id = torch.load(r"C:\Users\Timur\Desktop\MasterThesis\Code\MasterThesis\log\runs\EM_Taylor\19_07___21_55\Logs\Pipelines\Pipelines.pt")
    #
    # prediction = no_roll.model(Test_Loader[:][0].to(torch.device('cuda:0')))
    #
    # sampleNR = 40
    # sample = prediction[40,0,:,0].detach().cpu()
    # # sample = roll.KalmanSmoother.Filtered_State_Means[sampleNR,:,0,0]
    #
    # # plt.plot(em_taylor.KalmanSmoother.Filtered_State_Means[sample,:,0,0], color = 'b', label = 'EM + Taylor')
    # # plt.plot(prior.KalmanSmoother.Filtered_State_Means[sample,:,0,0])
    # # plt.plot(em_id.KalmanSmoother.Filtered_State_Means[sample,:,0,0])
    #
    #
    # obs,state = Test_Loader[sampleNR]
    #
    # obs = obs[0,:,0]
    # state = state[0,:,0]
    #
    # # plt.plot(obs[0,:,0],alpha = 0.4)
    # # plt.plot(state[0,:,0],color = 'g', label = 'Ground Truth')
    # plt.grid()
    # # plt.legend()
    #
    # t = np.arange(start=0, stop=sample.shape[0] / (360), step=1 / (360))
    # fig, ax = plt.subplots(dpi=200)
    #
    # ax.plot(t, sample, label= 'Auto Encoder no shifts', color='b')
    # ax.plot(t, state, label= 'Ground Truth', color='g', alpha=0.8)
    # # ax.plot(t, obs, label='Observations', color='r', alpha=0.8)
    #
    # ax.grid()
    # ax.legend()
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Amplitude [mV]')
    #
    # axins = ax.inset_axes([0.05, 0.5, 0.4, 0.4])
    # axins.plot(t, sample, color='b')
    # axins.plot(t, state, color='g',alpha = 0.8)
    # # axins.plot(t, obs, label='Observations', color='r', alpha=0.8)
    # axins.get_xaxis().set_visible(False)
    # axins.get_yaxis().set_visible(False)
    #
    # x1, x2, y1, y2 = 0.4, 0.6, torch.min(sample).item(), \
    #                  torch.max(torch.max(sample),torch.max(state)).item()
    #
    # # x1, x2, y1, y2 = 0.4, 0.6, torch.min(obs).item(), \
    # #                  torch.max(obs).item()
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])
    # axins.grid()
    #
    # ax.indicate_inset_zoom(axins, edgecolor="black")
    #
    # plt.title('Filtered ECG Sample for a SNR of 6[dB]')
    # plt.savefig('Filtered_ae_no_roll.pdf')
    #
    # plt.show()
    # 1
####################################################################################################################
####################################################################################################################
####################################################################################################################
################# AE snr ##################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################



    #
    # #
    # a = pandas.read_csv(r"C:\Users\Timur\Downloads\wandb_export_2022-07-19T18_48_36.202+02_00.csv",delimiter=',')
    # a = a.values[:,1:3]
    # perm = []
    #
    # y1 = a[:7,1]
    # y2 = a[7:14,1]
    # y3 = a[14:,1]
    #
    # x1 = a[:7,0]
    # x1[[2,5]] = x1[[5,2]]
    # x1[[3,5]] = x1[[5,3]]
    #
    # y1[[2, 5]] = y1[[5, 2]]
    # y1[[3, 5]] = y1[[5, 3]]
    # x2 = a[7:14, 0]
    # x3 = a[14:,0]
    #
    # roll = pandas.read_csv(r"C:\Users\Timur\Downloads\wandb_export_2022-07-20T00_43_08.235+02_00.csv", delimiter=',')
    # roll = roll.values[:, 1:3]
    # no_roll = pandas.read_csv(r"C:\Users\Timur\Downloads\wandb_export_2022-07-20T00_43_22.183+02_00.csv", delimiter=',')
    # no_roll = no_roll.values[:, 1:3]
    # #
    # noise_floor = [-4.35,-7.35,-10.35,-13.35,-16.35,-20.35, -30.35]
    #
    # plt.plot(roll[:,0],noise_floor,'*--',label = 'Noise Floor',color = 'r')
    #
    # plt.plot(x1,y1,'*--', label = 'Taylor + EM',color = 'blue')
    # plt.plot(x2,y2,'*--', label = '$ f= \mathbb{I} + EM$',color = 'orange')
    # plt.plot(x3,y3,'*--', label = 'Taylor Prior',color = 'c')
    # #
    #
    #
    #
    #
    # plt.plot(roll[:,0],roll[:,1],'*--', label = 'AE with shift=20',color = 'yellow')
    # plt.plot(no_roll[:,0],no_roll[:,1],'*--', label = 'AE without roll',color='purple')
    # plt.grid()
    # plt.legend()
    # plt.xlabel('SNR [dB]')
    # plt.ylabel('MSE Loss [dB]')
    # plt.title('MSE loss for Auto Encoder')
    # # plt.savefig('All_losses.pdf')
    # plt.savefig('losses_AE.pdf')
    #
    #
    # plt.show()


    # arr_taylor = [-7.32,-10.54,-13.66,-16.71, -19.72,-23.63,-32.30]
    # arr_id = [-7.29, -10.51, -13.61, -16.63, -19.56, -23.29, -30.37]
    #
    # x = [-6,-3,0,3,6,10,20]
    #
    #
    # plt.plot(x,arr_taylor,'*--', label = 'Taylor + RTS')
    # plt.plot(x,arr_id,'*--', label = '$ f= \mathbb{I}$ + RTS')
    # plt.grid()
    # plt.legend()
    # plt.xlabel('SNR [dB]')
    # plt.ylabel('MSE Loss [dB]')
    # plt.title('MSE loss for different SNRs for different models')
    # plt.savefig('ERTS_losses_plot.pdf')
    #
    #
    # plt.show()

    # noise_floor = [-4.35,-7.35,-10.35,-13.35,-16.35,-20.35, -30.35]


####################################################################################################################
####################################################################################################################
####################################################################################################################
################# Segmenet ##################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################


    # import torch
    #
    # from log.BaseLogger import LocalLogger,WandbLogger
    # import yaml
    # from yaml.loader import SafeLoader
    #
    # config = yaml.load(open('Configs/EM.yaml'), Loader=SafeLoader)
    #
    # Logger = LocalLogger('EM_Taylor', BaseConfig=config) if not config['wandb'] else \
    #     WandbLogger(name='EM_Taylor', group='SNR_sweep', BaseConfig=config)
    #
    # config = Logger.GetConfig()
    #
    # snr = config['snr']
    #
    # signal_length = config['signal_length']
    #
    # UseWandb = config['wandb']
    #
    # loader = PhyioNetLoader_MIT_NIH(1, 1, signal_length, SNR_dB=snr, random_sample=False, gpu=False,
    #                                 plot_sample=False, desired_shape=(1, signal_length, 2), roll=0)
    # np.random.seed(42)
    #
    #
    # N_train = int(0.8 * len(loader)) # int(0.05 * len(loader))
    # N_test = len(loader) - N_train
    #
    # dev = torch.device('cpu')
    # torch.random.manual_seed(42)
    #
    # Train_Loader, Test_Loader = torch.utils.data.random_split(loader, [N_train, N_test],
    #                                                           generator=torch.Generator())
    #
    # # N_test = 500
    #
    # # Test_Loader.indices = list(np.random.choice(Test_Loader.indices, 500, replace = False))
    # #
    # # Test_Loader.indices = Test_Loader.indices[:500]
    # #
    # # roll = torch.load(r"C:\Users\Timur\Desktop\MasterThesis\Code\MasterThesis\log\runs\AutoEncoder_SNR_sweep_roll\19_07___22_30\Logs\Pipelines\Pipelines.pt")
    # #
    # # no_roll = torch.load(r"C:\Users\Timur\Desktop\MasterThesis\Code\MasterThesis\log\runs\AutoEncoder_SNR_sweep_roll\19_07___22_19\Logs\Pipelines\Pipelines.pt")
    #
    # # em_id = torch.load(r"C:\Users\Timur\Desktop\MasterThesis\Code\MasterThesis\log\runs\EM_Taylor\19_07___21_55\Logs\Pipelines\Pipelines.pt")
    #
    # # prediction = no_roll.model(Test_Loader[:][0].to(torch.device('cuda:0')))
    #
    # sampleNR = 40
    # # sample = prediction[40,0,:,0].detach().cpu()
    # # sample = roll.KalmanSmoother.Filtered_State_Means[sampleNR,:,0,0]
    #
    # # plt.plot(em_taylor.KalmanSmoother.Filtered_State_Means[sample,:,0,0], color = 'b', label = 'EM + Taylor')
    # # plt.plot(prior.KalmanSmoother.Filtered_State_Means[sample,:,0,0])
    # # plt.plot(em_id.KalmanSmoother.Filtered_State_Means[sample,:,0,0])
    #
    #
    # obs,state = Test_Loader[sampleNR]
    # #
    # # obs = obs[0,:,0]
    # state = state[0,:,0]
    # segments = [60,144,240,-1]
    #
    # #segments = [0,60,110,170,230,-1]
    #
    # print(int(0.6*360))
    # #
    # # # plt.plot(obs[0,:,0],alpha = 0.4)
    # # # plt.plot(state[0,:,0],color = 'g', label = 'Ground Truth')
    # # plt.grid()
    # # # plt.legend()
    # #
    # t = np.arange(start=0, stop=state.shape[0] / (360), step=1 / (360))
    # fig, ax = plt.subplots(dpi=200)
    # #
    # # # ax.plot(t, sample, label= 'Auto Encoder no shifts', color='b')
    # ax.plot(t, state, color='g', alpha=0.8)
    # # # ax.plot(t, obs, label='Observations', color='r', alpha=0.8)
    # ax.vlines([t[i] for i in segments], torch.min(state), torch.max(state), colors = 'r')
    # #
    # ax.grid()
    # ax.legend()
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Amplitude [mV]')
    #
    #
    # # axins = ax.inset_axes([0.05, 0.5, 0.4, 0.4])
    # # axins.plot(t, sample, color='b')
    # # axins.plot(t, state, color='g',alpha = 0.8)
    # # # axins.plot(t, obs, label='Observations', color='r', alpha=0.8)
    # # axins.get_xaxis().set_visible(False)
    # # axins.get_yaxis().set_visible(False)
    # #
    # # x1, x2, y1, y2 = 0.4, 0.6, torch.min(sample).item(), \
    # #                  torch.max(torch.max(sample),torch.max(state)).item()
    # #
    # # # x1, x2, y1, y2 = 0.4, 0.6, torch.min(obs).item(), \
    # # #                  torch.max(obs).item()
    # # axins.set_xlim(x1, x2)
    # # axins.set_ylim(y1, y2)
    # # axins.set_xticklabels([])
    # # axins.set_yticklabels([])
    # # axins.grid()
    # #
    # # ax.indicate_inset_zoom(axins, edgecolor="black")
    # #
    # plt.title('Segmented ECG Signal')
    # #plt.savefig('Sample_Plots/5_Segment.pdf')
    #
    # plt.show()
    # 1

    # -21.63 Knet