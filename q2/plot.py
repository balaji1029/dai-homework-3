import numpy as np
import matplotlib.pyplot as plt

data = np.load('density.npy')

no = 40

x = np.linspace(-6, 6, no)
y = np.linspace(-6, 6, no)
X, Y = np.meshgrid(x, y)

print(list(data))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, data, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Estimated Density')
# Save the plot to a file
plt.savefig('density.png')