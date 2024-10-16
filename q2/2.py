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
            return 3 * (1 - np.dot(diff, diff)) / np.pi
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
# Normalize the data by scaling it between min and max
data_min = np.min(data)
data_max = np.max(data)
data_normalized = (data - data_min) / (data_max - data_min)
# TODO: Initialize the EpanechnikovKDE class
kde = EpanechnikovKDE()

# TODO: Fit the data
kde.fit(data)

num = 60

# TODO: Plot the estimated density in a 3D plot
x = np.linspace(-6, 6, num)
y = np.linspace(-6, 6, num)
X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)

mesh = np.array([X.flatten(), Y.flatten()]).T

mesh = mesh[:, np.newaxis, :]
data = kde.data[np.newaxis, :, :]

diff = (mesh - data)/kde.bandwidth

norm = np.sum(diff**2, axis=2)

norm = np.where(norm > 1, 1, norm)

kernel_values = 3 * (1 - norm) / np.pi

kernel_values_sum = np.sum(kernel_values, axis=1) / (len(kde.data) * kde.bandwidth)

Z = kernel_values_sum.reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('Estimated Density')
plt.savefig('images/transaction_distribution.png')
plt.show()