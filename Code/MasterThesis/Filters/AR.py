import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.ar_model import AutoReg

class AR():


    def __init__(self, order):

        self.order = order


    def fit(self,x):
        self.autoreg = AutoReg(x.numpy(),lags = self.order-1,period=360)
        self.AR_model_coeffs, self.sigma = yule_walker(x,self.order)


        self.autoreg_fit = self.autoreg.fit()

        a = self.autoreg.predict(self.AR_model_coeffs,start=0,end=1000)

        from matplotlib import pyplot as plt

        plt.plot(a)
        plt.show()
        1





if __name__ == '__main__':


    from DataLoaders.PhysioNetLoader import *

    signal_length = 360
    snr = 0

    loader = PhyioNetLoader_MIT_NIH(1, 1, signal_length, SNR_dB=snr, random_sample=False, gpu=False,
                                    plot_sample=False, desired_shape=(1, signal_length, 2), roll=0)


    ar = AR(5)

    inp = loader[:20][0]

    inp = inp.squeeze().reshape(-1,2)


    ar.fit(inp[:,0])
    1