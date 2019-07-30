import numpy as np
import math
import cmath
from Source.Misc import *

'''
class Codebook:
    def __init__(self):
        self.D
'''

def DFT_Codebook(n_xyz):
    D_el = np.fft.fft(np.eye(n_xyz[2]))

    D_az = np.fft.fft(np.eye(n_xyz[1]))

    return np.kron(D_az, D_el)

'''
def Uniform_Codebook(n_xyz, omega_vec):
    N = n_xyz[1]
    D_el = np.eye(n_xyz[2])
    D_az = np.zeros((N, len(omega_vec)), dtype=np.complex)
    for k in range(len(omega_vec)):
        D_az[:, k] = root_beam(omega_vec[k], N).ravel()
    return np.kron(D_az, D_el)
'''

#Didn't test this yet!!
def Hierarchical_Codebook(n_xyz, phi_m, level):
    if level == 0:
        phi_mv = [0]
    else:
        l = [-1, 0, 1]
        omega = [sind(phi_m) + x / math.pow(3, level) for x in l]
        phi_mv = [asind(x) for x in omega]

    #n_xyz = [1, n, 1]
    D_el = np.eye(n_xyz[2])

    Na = np.min([n_xyz[2], int(math.pow(3, level))]) #no. of active antennas

    D_az = np.zeros((Na, len(phi_mv)), dtype=np.complex)
    # print(D_az.shape)
    for k in range(len(phi_mv)):
        D_az[:, k] = root_beam(phi_mv[k], level, n_xyz[1]).ravel()
        # print("Vector values: {0}".format(D_az[:][k]))
    D = np.kron(D_az, D_el)

    return D


'''
    y = root_beam(phi_m, level, N)

    - Computes the root beam of the antenna unit
    - Useful in unit normal vector estimations

    Parameters:
    phi_m - root beam angle in degree
    level - beam width level [0,1,2,3]
    N - no. of antenna elements in the unit

    Output:
    y - root beam vector 
    '''


def root_beam(phi_m, level, N):
    phi_m = deg2rad(phi_m)  # converting deg to radians
    Na = np.min([N, int(math.pow(3, level))])

    x = np.arange(0, Na)
    y = np.zeros((N, 1), dtype=np.complex)
    # print(y.shape)
    for k in x:
        # print("phi_m: {0}".format(phi_m))
        y[k] = np.exp(1j * 2 * math.pi * 0.5 * cmath.sin(phi_m) * k)
    # print(y.shape)
    return y