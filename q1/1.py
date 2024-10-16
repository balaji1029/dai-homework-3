import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv', header=12)
filter_1 = data['D (Mpc)'][:1500]
final_data = filter_1[filter_1 < 4]
n = len(final_data)

# plt.hist(final_data, bins=10)
# title = r'\hat{p}(x)'
# plt.title('$%s$'%title)
# plt.show()

counts, bins = np.histogram(final_data, bins=10, range=(0, 4))
h = bins[1] - bins[0]
p = counts / (h * n)
print(p)

estimator = []
bin_widths = []
for bin_count in range(1, 1001):
    counts, bins = np.histogram(final_data, bins=bin_count, range=(0, 4))
    bin_width = bins[1] - bins[0]
    estimate = (1/(n**2 * bin_width)) * np.sum(counts**2)
    estimate -= (2/(n*(n-1)*bin_width)) * np.sum(counts**2 - counts)
    estimator.append(estimate)
    bin_widths.append(bin_width)

# plt.figure(dpi=1000)
# plt.plot(bin_widths, estimator)
# title = r'\hat{J}(h)'
# plt.title('$%s$'%title + ' vs h')
# plt.xlabel('h')
# plt.ylabel('$%s$'%title)
# plt.show()
# np.argmin(estimator), bin_widths[np.argmin(estimator)]

optimal_bandwidth = np.argmin(estimator)
h_star = bin_widths[np.argmin(estimator)]

# plt.hist(final_data, bins=np.argmin(estimator)+1, range=(0, 4))
# title = r'\hat{p}(x)'
# plt.title('$%s$'%title)
# plt.show()

print(h_star)