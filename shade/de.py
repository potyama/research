import numpy as np


class searchAlgorithm:
    def randDouble(self) -> float:
        return np.random.rand()

    def cauchy_g(self, mu: float, gamma: float) -> float:
        return mu + gamma * np.tan(np.pi * (self.randDouble() - 0.5))

    def gauss(self, mu: float, sigma: float) -> float:
        return mu + sigma * np.sqrt(-2.0 * np.log(self.randDouble())) * np.sin(2.0 * np.pi * self.randDouble())

    def sortIndexWithQuickSort(self, array: np.ndarray, first: int, last: int, index: np.ndarray):
        x = array[(first + last) // 2]
        i = first
        j = last

        while True:
            while array[i] < x:
                i += 1
            while x < array[j]:
                j -= 1
            if i >= j:
                break

            array[i], array[j] = array[j], array[i]
            index[i], index[j] = index[j], index[i]

            i += 1
            j -= 1

        if first < (i - 1):
            self.sortIndexWithQuickSort(array, first, i - 1, index)
        if (j + 1) < last:
            self.sortIndexWithQuickSort(array, j + 1, last, index)