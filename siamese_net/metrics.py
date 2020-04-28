# Project: https://github.com/adambielski/siamese-triplet
# Author: Adam Bielski https://github.com/adambielski
# License: BSD 3-Clause


import numpy as np


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AverageNonzeroTripletsMetric(Metric):
    """
    Counts average number of nonzero triplets found in mini-batches
    """

    def __init__(self):
        super().__init__()
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss.item())
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'
