import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    ### Load data
    observation = np.loadtxt('Obs.csv')
    state = np.loadtxt('State.csv')
    aeResult = np.loadtxt('aeResults.csv')
    smoothedResults = np.loadtxt('SmoothedEM.csv')
    filiterdResults = np.loadtxt('FilteredEM.csv')
    rikResults = np.loadtxt('rikResults.csv')


    ### Set up time
    t = np.linspace(0,360,360)
    t = t/360

    ### Create axes and figure
    fig,ax = plt.subplots(figsize = (16,9),dpi = 400)

    ### Plot parameters
    markerSize = 20
    lineWidthPrimary = 3
    lineWidthSecondary = 2
    fontSize = 22
    axisFont = 20
    legendFont = 20
    tickFont = 16

    ### Colors
    yellow = '#fff017'
    red = '#ff0000'
    cyan = '#00FFFF'
    blue = '#0000ff'
    green = '#00ff00'
    grey = '#bebebe'

    ### Plot observations
    plt.plot(t, observation,
             color=red,
             alpha=0.3,
             label='Observations',
             linewidth=lineWidthSecondary,
             markersize=markerSize,
             linestyle='-')

    ### Plot states
    plt.plot(t, state,
             color=green,
             label='Ground Truth',
             linewidth=lineWidthPrimary,
             markersize=markerSize,
             linestyle='-')

    ### Plot AE output
    plt.plot(t, aeResult,
             color=yellow,
             label='AE-intra',
             linewidth=lineWidthPrimary,
             markersize=markerSize,
             linestyle='--')

    ### Plot Rik estimate
    plt.plot(t, rikResults,
             color=grey,
             label='KF-inter',
             linewidth=lineWidthSecondary,
             markersize=markerSize,
             linestyle='--')

    ### Plot smoothed estimate
    plt.plot(t, smoothedResults,
             color=cyan,
             label='KS-intra',
             linewidth=lineWidthSecondary,
             markersize=markerSize,
             linestyle='--')

    ### Plot filterd estimate
    plt.plot(t, filiterdResults,
             color=blue,
             label='HKF',
             linewidth=lineWidthPrimary,
             markersize=markerSize,
             linestyle='-')

    ### Insert Zoomed in plot
    axins = ax.inset_axes([0.0, 0.6, 0.4, 0.4])
    axins.get_xaxis().set_visible(False)
    axins.get_yaxis().set_visible(False)

    ### Plot inside zoom
    ### Plot states
    axins.plot(t, state,
               color=green,
               label='Ground Truth',
               linewidth=lineWidthPrimary,
               markersize=markerSize,
               linestyle='-')

    ### Plot AE output
    axins.plot(t, aeResult,
               color=yellow,
               linewidth=lineWidthSecondary,
               markersize=markerSize,
               linestyle='--')

    ### Plot Rik output
    axins.plot(t, rikResults,
               color=grey,
               linewidth=lineWidthSecondary,
               markersize=markerSize,
               linestyle='--')

    ### Plot smoothed estimate
    axins.plot(t, smoothedResults,
               color=cyan,
               linewidth=lineWidthSecondary,
               markersize=markerSize,
               linestyle='--')

    ### Plot filterd estimate
    axins.plot(t, filiterdResults,
               color=blue,
               linewidth=lineWidthPrimary,
               markersize=markerSize,
               linestyle='-')

    x1, x2, y1, y2 = 0.4, 0.6, ax.dataLim.intervaly[0], ax.dataLim.intervaly[1]
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.grid()

    ax.indicate_inset_zoom(axins, edgecolor="black")

    ### Update axis parameters
    plt.legend(fontsize=legendFont, loc='upper right')
    plt.xticks(fontsize=tickFont)
    plt.yticks(fontsize=tickFont)
    plt.grid()
    plt.xlabel('Time [s]', fontsize=axisFont)
    plt.ylabel('Amplitude [mV]', fontsize=axisFont)
    plt.savefig('singleHeartbeatPhysioNet.pdf')
    plt.show()







