import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from Pipelines.EM_Pipelines import Rik_Pipeline, EM_Pipeline
from scipy.interpolate import interp2d




def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

def interpolate(x,y, steps):

    interpolated = []

    for xs,ys in zip(x,y):

        inter = np.linspace(xs,ys,steps)

        interpolated.append(inter)


    return np.array(interpolated)






if __name__ == '__main__':
    t = (np.linspace(1,250,250)-1).astype(int)

    Pipeline = torch.load(r"E:\MasterThesis\log\runs\EM_algorithm_Taylor\22_09_11___17_45\Logs\Pipelines\Pipelines.pt")

    observations, states = Pipeline.EM_Data[:]

    observation_1 = observations[0,0,:,0]
    observation_1_2 = observations[:2,0,:,0].reshape(-1)

    states_1 = states[0, 0, :, 0]
    states_1_2 = states[:2, 0, :, 0].reshape(-1)

    Filter_out_1 = Pipeline.FilteredResults[0,:,0]
    Filter_out_1_2 = Pipeline.FilteredResults[:2, :, 0].reshape(-1)
    steps = 10


    Smoother_out_1 = Pipeline.SmoothedResults[0,:,0]
    Smoother_out_1_2 = Pipeline.SmoothedResults[:2, :, 0].reshape(-1)
    Smoother_out_1_2_to_filter = interpolate(Smoother_out_1_2,Filter_out_1_2,steps)


    prior = Pipeline.prior[0]
    prior_12 = torch.cat((prior,prior))

    prior_time = 20
    prior_to_smooth = interpolate(prior,Smoother_out_1,steps)
    prior_to_smooth_1_2 = interpolate(prior_12,Smoother_out_1_2,steps)


    fig, ax = plt.subplots()
    xdata, ydata = [], []

    x_obs,y_obs = t, observation_1

    obs_plt, = plt.plot(x_obs,y_obs, color = 'r',alpha =0.3, label = 'Observation')
    state_plt = plt.plot(x_obs,states_1, color = 'g', label = 'Ground Truth')[0]
    ln, = plt.plot([], [], 'bx', label = 'prior')
    vline_plot = plt.vlines(1.5,0,0)
    plt.xlabel('time steps')
    plt.ylabel('Amplitude [mV]')
    plt.grid()
    plt.legend()
    # plt.

    def init():
        ax.set_xlim(0, 250 )
        ax.set_ylim(-1.5, 1.5)
        return ln,obs_plt,state_plt

    def update(frame):




        if frame < prior_time:
            xdata = np.linspace(1, 250, 250) - 1
            ydata = prior
            ln.set_data(xdata, ydata)

            return ln,obs_plt,state_plt


        if frame >= prior_time and frame  < steps + prior_time:
            xdata = np.linspace(1, 250, 250) - 1
            ydata = prior_to_smooth[:, frame-prior_time]
            ln.set_label('Smoothed States')
            plt.legend()

            ln.set_data(xdata, ydata)
            return ln,obs_plt,state_plt


        elif frame >= steps + prior_time:
            xdata = np.linspace(1, min(250 + 10*(frame - prior_time),500), min(250 + 10*(frame - prior_time),500)) - 1
            plt.clf()

            obs_plt.set_data(xdata,observation_1_2[:min(250 + 10*(frame - prior_time),500)])
            ln.set_data(xdata, Smoother_out_1_2[:min(250 + 10*(frame - prior_time),500)])
            state_plt.set_data(xdata, states_1_2[:min(250 + 10*(frame - prior_time),500)])

            ax.set_xlim(0, min(250 + 10*(frame - prior_time),500))
            ax.set_ylim(-1.5,1.5)
            plt.legend()

            return (ln,obs_plt,state_plt,vline_plot)

        else:
            print('sdflksdnflksdnhflkkhnfdlk')

            xdata = np.linspace(1, 500,500) - 1

            ln.set_data(xdata, Smoother_out_1_2_to_filter[:,frame- (prior_time-steps)])

            # ax.set_xlim(0, 250 + 10 * (frame - prior_time))
            ax.set_ylim(-1.5, 1.5)
            plt.legend()

            return (ln,obs_plt,state_plt,vline_plot)






    frames = (np.linspace(1,2*steps +25 + prior_time , 2*steps + 25 + prior_time)-1).astype(int)


    ani = FuncAnimation(fig, update, frames=frames,
                        init_func=init, blit=True)
    plt.show()



    1