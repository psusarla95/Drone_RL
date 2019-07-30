import numpy as np


def H(mat):
    return mat.conj().T


def awgn(n):
    noise = 0.5 * (np.random.rand(n) + 1j * np.random.rand(n))
    return noise
