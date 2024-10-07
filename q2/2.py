import numpy as np
import matplotlib.pyplot as plt

# Custom Epanechnikov KDE class
class EpanechnikovKDE:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        """Fit the KDE model with the given data."""
        self.data = data

    def epanechnikov_kernel(self, x, xi):
        """Epanechnikov kernel function."""
        diff = (x - xi)/self.bandwidth
        if np.linalg.norm(diff) <= 1:
            return 0.75 * (1 - np.dot(diff, diff))
        return 0

    def evaluate(self, x):
        """Evaluate the KDE at point x."""
        value = 0
        for xi in self.data:
            value += self.epanechnikov_kernel(x, xi)
        return value / (len(self.data) * self.bandwidth)

# Load the data from the NPZ file
data_file = np.load('transaction_data.npz')
data = data_file['data']

# TODO: Initialize the EpanechnikovKDE class
kde = EpanechnikovKDE()

# TODO: Fit the data
kde.fit(data)

no = 30

# TODO: Plot the estimated density in a 3D plot
x = np.linspace(-6, 6, no)
y = np.linspace(-6, 6, no)
X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)

# X = np.linspace(0, 100, 100)
# Y = np.linspace(0, 100, 100)
# X, Y = np.meshgrid(X, Y)
# Z = np.zeros(X.shape)

# X = np.linspace(0, 100, 100)
# Y = np.linspace(0, 100, 100)
# X, Y = np.meshgrid(X, Y)
# Z = np.zeros(X.shape)

from tqdm import tqdm

def row_wise(row, i, Y):
    value = np.zeros(row.shape)
    for j in range(Y.shape[0]):
        value[j] = kde.evaluate(np.array([row[i], Y[j][i]]))
    print('Done with row', i)
    return value
# def second_half():
#     for i in range(X.shape[0]//3, X.shape[0]//3 + 1):
#         for j in tqdm(range(X.shape[1])):
#             Z[i, j] = kde.evaluate([X[i, j], Y[i, j]])
#         # print('Done with row', i)
# def third_half():
#     for i in range(2*X.shape[0]//3, 2*X.shape[0]//3 + 1):
#         for j in tqdm(range(X.shape[1])):
#             Z[i, j] = kde.evaluate([X[i, j], Y[i, j]])
        # print('Done with row', i)

# t1 = threading.Thread(target=first_half)
# t2 = threading.Thread(target=second_half)
# t3 = threading.Thread(target=third_half)

# t1.start()
# t2.start()
# t3.start()

# t1.join()
# t2.join()
# t3.join()


import multiprocessing as mp

# print(mp.cpu_count())
pool = mp.Pool(mp.cpu_count())

for i in range(X.shape[0]):
    Z[i] = pool.apply(row_wise, args=(X[i], i, Y))

pool.close()

# TODO: Save the plot
np.save('density.npy', Z)