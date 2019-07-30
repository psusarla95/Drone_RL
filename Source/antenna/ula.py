import numpy as np


def steervec(ntx, theta, phi):
    P = np.zeros([3, ntx])
    P[0, :] = np.arange(0, ntx)
    P[1, :] = np.zeros(ntx)
    P[2, :] = np.zeros(ntx)

    kv = np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)])
    kv.reshape(3, 1)
    v = 1 / np.sqrt(ntx) * np.exp(1j * 2 * np.pi * 0.5 * (P.transpose()).dot(kv))
    return v
