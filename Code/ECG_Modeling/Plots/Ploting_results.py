# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
from matplotlib import pyplot as plt
import numpy as np


def EM_results_ploting_taylor_model():


    SNR= np.array([-6.,-3.,0.,3.,6.,10.,15.,20.])

    mse_q_2_002  = [-6.,-3.,0.,3.,6.,10.,15.,20.]
    mse_q_2_0002  = [-16.38,-17.11,-18.49,-19.93,-21.79,-25.41,-28.70,-32.65]
    mse_q_2_00002 = [-16.16, -16.49, -17.06, -17.25, -18.06, -18.53, -18.84 , -19.16]

    a = np.polyfit(SNR ,mse_q_2_0002 ,deg = 1)
    b = np.polyfit(SNR ,mse_q_2_00002 ,deg = 1)


    plt.plot(SNR ,mse_q_2_0002 , '^', label = '$q^2 = -27 dB$ ',color = 'c')
    plt.plot(SNR , SNR *a[0] + a[1], 'c--', alpha = 0.5, label = 'Linear best fit')
    plt.plot(SNR, mse_q_2_00002, '^', label='$q^2 = -37 dB$ ', color='orange')
    plt.plot(SNR, SNR * b[0] + b[1], '--', alpha=0.5, label='Linear best fit', color= 'orange')
    plt.grid()
    plt.legend()
    plt.xlabel('SNR [dB]')
    plt.ylabel('MSE [dB]')
    plt.title('MSE per SNR for MIT-BIH dataset with ADWGN with constant ss-Model')
    plt.savefig('MIT-BIH-MSE.pdf')
    plt.show()

if __name__ == '__main__':

    EM_results_ploting_taylor_model()