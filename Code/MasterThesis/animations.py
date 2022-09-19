import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from Pipelines.EM_Pipelines import Rik_Pipeline, EM_Pipeline



def init():
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1, 1)
    return ln,


def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

if __name__ == '__main__':


    Pipeline = torch.load(r"E:\MasterThesis\log\runs\EM_algorithm_Taylor\22_09_11___17_45\Logs\Pipelines\Pipelines.pt")

    observations, states = Pipeline.EM_Data[:]

    Filter_out = Pipeline.FilteredResults
    Smoother_out = Pipeline.SmoothedResults


    prior = Pipeline.prior

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'ro')


    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 128),
                        init_func=init, blit=True)
    plt.show()



    1