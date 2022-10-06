import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':


    # Get Data
    obs = np.loadtxt('stackedObs.csv')
    states = np.loadtxt('stackedState.csv')
    smoother = np.loadtxt('stackedSmoother.csv')
    filters = np.loadtxt('stackedFilter.csv')
    ae = np.loadtxt('stackedAE.csv')
    rik = np.loadtxt('stackedRik.csv')
    a = np.array(1*[0.05224688])
    rik = np.concatenate((a,rik))

    # b = np.array(17*[0.003057])
    # ae = np.concatenate((b,ae))


    ### Plot parameters
    markerSize = 20
    lineWidthPrimary = 3
    lineWidthSecondary = 2
    fontSize = 22
    axisFont = 16
    legendFont = 14
    tickFont = 16

    ### Colors
    yellow = '#fff017'
    red = '#ff0000'
    cyan = '#00FFFF'
    blue = '#0000ff'
    green = '#00ff00'
    grey = '#bebebe'

    # Plotting
    T = len(obs)
    t_cons = np.arange(start=0, stop=7, step=7 / len(obs))

    fig_con, ax_cons = plt.subplots(nrows=4, ncols=1, figsize=(16, 9), dpi=120)
    fig_con.set_tight_layout(True)

    ax_cons[0].plot(t_cons, obs.squeeze(), label='Observations', color=red, alpha=0.4)

    # ax_cons[0].set_xlabel('Time [s]', fontsize=fontSize)
    ax_cons[0].set_ylabel('Amplitude [mV]', fontsize=axisFont)
    title_cons = 'Observations'
    ax_cons[0].set_title(title_cons, fontsize=fontSize)
    ax_cons[0].xaxis.set_tick_params(labelsize=fontSize)
    ax_cons[0].yaxis.set_tick_params(labelsize=fontSize)
    ax_cons[0].set_ylim(-0.5,1)


    ax_cons[1].plot(t_cons, states.squeeze(), label='Observations', color=green, linewidth=lineWidthPrimary,
             markersize=markerSize,
             linestyle='-')
    # ax_cons[1].set_xlabel('Time [s]', fontsize=fontSize)
    ax_cons[1].set_ylabel('Amplitude [mV]', fontsize=axisFont)
    title_cons = 'Ground Truth'
    ax_cons[1].set_title(title_cons, fontsize=fontSize)
    ax_cons[1].xaxis.set_tick_params(labelsize=fontSize)
    ax_cons[1].yaxis.set_tick_params(labelsize=fontSize)
    ax_cons[1].set_ylim(-0.5,1)



    # ax_cons[2].plot(t_cons, smoother.squeeze(), label='KS-intra', color=cyan,linewidth=lineWidthPrimary,
    #          markersize=markerSize,
    #          linestyle='--')
    ax_cons[2].plot(t_cons, rik.squeeze(), label='KF-inter', color=grey,linewidth=lineWidthPrimary,
             markersize=markerSize,
             linestyle='--')
    ax_cons[2].plot(t_cons, ae[:-1].squeeze(), label='AE-intra', color=yellow,linewidth=lineWidthPrimary,
             markersize=markerSize,
             linestyle='--')

    # ax_cons[2].set_xlabel('Time [s]', fontsize=fontSize)
    ax_cons[2].set_ylabel('Amplitude [mV]', fontsize=axisFont)
    title_cons = 'Comparative Results'
    ax_cons[2].set_title(title_cons, fontsize=fontSize)
    ax_cons[2].xaxis.set_tick_params(labelsize=fontSize)
    ax_cons[2].yaxis.set_tick_params(labelsize=fontSize)
    ax_cons[2].legend(fontsize = legendFont)
    ax_cons[2].set_ylim(-0.5,1)



    ax_cons[3].plot(t_cons, filters.squeeze(), label='AE-intra', color=blue,linewidth=lineWidthPrimary,
             markersize=markerSize,
             linestyle='-')

    ax_cons[3].set_xlabel('Time [s]', fontsize=fontSize)
    ax_cons[3].set_ylabel('Amplitude [mV]', fontsize=axisFont)
    title_cons = 'HKF'
    ax_cons[3].set_title(title_cons, fontsize=fontSize)
    ax_cons[3].xaxis.set_tick_params(labelsize=fontSize)
    ax_cons[3].yaxis.set_tick_params(labelsize=fontSize)
    ax_cons[3].set_ylim(-0.5,1)


    plt.savefig('consecutiveResultsProp.pdf')
    plt.show()