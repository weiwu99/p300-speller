import numpy as np
import scipy.io as sio

from timeit import default_timer as timer
import basics


if __name__ == '__main__':    
    #%% Load data
    A01 = sio.loadmat('A01.mat')
    A02 = sio.loadmat('A02.mat')
    A03 = sio.loadmat('A03.mat')
    A04 = sio.loadmat('A04.mat')
    A05 = sio.loadmat('A05.mat')
    A06 = sio.loadmat('A06.mat')
    A07 = sio.loadmat('A07.mat')
    A08 = sio.loadmat('A08.mat')

    start = timer() 
    C_reg = np.logspace(-6, 0, num=7)

    basics.ssc_CV(C_reg = C_reg, clf_type = 'LogReg', dataset = A08, train_trial_num = 20)
    # dsc.dsc_CV(C_reg = C_reg, clf_type = 'LDA', dataset = A04, train_trial_num = 20)



