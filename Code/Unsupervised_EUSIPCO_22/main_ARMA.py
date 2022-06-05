# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import mne
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    EKG = mne.io.read_raw_edf(r'C:\Users\Timur\Desktop\MasterThesis\Code\Unsupervised_EUSIPCO_22\Datasets\abdominal-and-direct-fetal-ecg-database-1.0.0\r01.edf')
    EKG_data = EKG.get_data()
    EKG_mean = EKG_data[1:].mean(0)[0:1000]

    # model = arch_model(EKG_mean*1e5, vol='GARCH', p=15, q=15)

    model = ARIMA(EKG_mean*1e5,order=(5,1,3))
    model_fit = model.fit()
    print(model_fit.summary())
    # forecast = model_fit.forecast(horizon=1, start=0, method='simulation')
    # plt.plot(forecast.simulations.values[:,3])

    # model_fit.plot()
    # print(model.state_names)
    # print(model_fit.summary())

    # ssA = np.eye(5)
    # ssA[:,0] = model_fit.arparams
    #
    # ssB = -model_fit.maparams.T

    plt.plot(model_fit.fittedvalues)
    # plt.plot(EKG_mean)
    # plt.plot(model_fit.filtered_state[0])
    plt.show()
    1