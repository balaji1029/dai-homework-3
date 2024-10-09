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

num = 40

# TODO: Plot the estimated density in a 3D plot
x = np.linspace(-6, 6, num)
y = np.linspace(-6, 6, num)
X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)

from tqdm import tqdm

def row_wise(row, i, Y):
    value = np.zeros(row.shape)
    for j in range(Y.shape[0]):
        value[j] = kde.evaluate(np.array([row[i], Y[j][i]]))
    print('Done with row', i)
    return value

# import multiprocessing as mp

# # print(mp.cpu_count())
# pool = mp.Pool(mp.cpu_count())

# for i in range(X.shape[0]):
#     Z[i] = pool.apply(row_wise, args=(X[i], i, Y))

# pool.close()

for i in tqdm(range(X.shape[0])):
    for j in range(X.shape[1]):
        Z[i][j] = kde.evaluate(np.array([X[i][j], Y[i][j]]))

# TODO: Save the plot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.savefig('3d_plot.png')
plt.show()