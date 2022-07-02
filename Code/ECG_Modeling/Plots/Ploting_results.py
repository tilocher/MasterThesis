# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
from matplotlib import pyplot as plt
import numpy as np


def EM_results_ploting_taylor_model():


    SNR= np.array([-6.,-3.,0.,3.,6.,10.,15.,20.])

    noise_floor = [-4.83,-7.819,-10.823 ,-13.827, -16.82, -20.822, -25.813, -30.824]

    mse_q_2_002  = [-6.,-3.,0.,3.,6.,10.,15.,20.]
    mse_q_2_00002_taylor  = [-16.38,-17.11,-18.49,-19.93,-21.79,-25.41,-28.70,-32.65]
    # mse_q_2_00002_taylor = [-16.16, -16.49, -17.06, -17.25, -18.06, -18.53, -18.84 , -19.16]
    mse_q_2_00002_constA = [-11.37230, -14.04923, -15.88955, -18.76115, -20.36785, -25.09783, -29.27905,-34.402176]
    mse_q_2_00002_constV = [-14.00839, -16.52352, -17.93246, -20.63577, -22.96424, -25.76315, -29.40805,-33.50840]
    mse_q_2_00002_taylor = [-16.51745, -17.97296, -18.62591, -20.31408, -22.11643, -24.41924, -29.12135, -33.02374]
    mse_q_2_00002_ECGMODEL = [-16.52153, -17.97947, -18.63157, -20.31903,-22.12518, -24.43914, -29.14500, -32.99022]




    plt.plot(SNR ,noise_floor , '--', label = 'Noise Floor',color = 'r')

    plt.plot(SNR, mse_q_2_00002_constA, '^', label='Constant Acceleration', color='orange')
    plt.plot(SNR, mse_q_2_00002_constV, '^', label='Constant velocity', color='b')
    plt.plot(SNR ,mse_q_2_00002_taylor , '^', label = 'Taylor',color = 'c')
    plt.plot(SNR, mse_q_2_00002_ECGMODEL, '^', label='ECG Model', color='pink')


    plt.grid()
    plt.legend()
    plt.xlabel('SNR [dB]')
    plt.ylabel('MSE [dB]')
    plt.title('MSE per SNR for MIT-BIH dataset with ADWGN with constant ss-Model')
    plt.savefig('MSE_plots\\MIT-BIH-MSE.pdf')
    plt.show()

if __name__ == '__main__':

    EM_results_ploting_taylor_model()