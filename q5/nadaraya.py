from ossaudiodev import SOUND_MIXER_LINE1
import scipy
import numpy as np
import matplotlib.pyplot as plt

class NadarayaKernel:

    def __init__(self, bandwidth = 1):
        self.data = None
        self.yi = None
        self.bandwidth = bandwidth

    def fit(self, data):
        self.data = np.array(data.drop(['yield'], axis=1))
        self.yi = np.array(data['yield'])

    
    def kernel(self, x, xi):
        d = x.shape[0]
        # print(x.shape, xi.shape)
        # print((x - xi).shape, self.bandwidth.shape)
        # print(np.matmul((x - xi), self.bandwidth))
        value = 1 - (1/3 * np.matmul((x - xi), (x - xi).T))
        # print('Yay', value)
        return (1 / (np.linalg.det(self.bandwidth) ** d)) * max(0, value)
    
    def evaluate(self, x):
        sum1 = 0
        sum2 = 0
        for i, xi in enumerate(self.data):
            kernel = self.kernel(x, xi)
            sum1 += kernel * self.yi[i]
            sum2 += kernel
        # print(sum1, sum2)
        return 0 if sum2 == 0 else sum1 / sum2

