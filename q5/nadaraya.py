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
        diff = x-xi
        norm = np.linalg.norm(diff)
        argument = norm / self.bandwidth
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * argument**2)

    def evaluate(self, x):
        sum1 = 0
        sum2 = 0
        for i, xi in enumerate(self.data):
            kernel = self.kernel(x, xi)
            sum1 += kernel * self.yi[i]
            sum2 += kernel
        return 0 if sum2 == 0 else sum1 / sum2

    def predict(self, test):
        test = np.array(test)
        test_expanded = test[:, np.newaxis, :]
        data_expanded = self.data[np.newaxis, :, :]
        diff = (test_expanded - data_expanded)/self.bandwidth
        nor = np.sum(diff**2, axis=2)
        # kernel_values = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (nor / self.bandwidth)**2)
        # Epachnikov kernel
        nor = np.where(nor > 1, 1, nor)
        kernel_values = 0.75 * (1 - nor)
        kernel_values_sum = np.sum(kernel_values, axis=1)
        kernel_values_sum = np.where(kernel_values_sum == 0, 1, kernel_values_sum)
        return np.sum(kernel_values * self.yi, axis=1) / kernel_values_sum