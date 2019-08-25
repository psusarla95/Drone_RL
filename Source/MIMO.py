import numpy as np
import cmath
import math
from Source.Misc import *
from Source.Codebook import *
from Source.channel_mmW import *
from Source.miscfun.geometry import *
from Source.antenna import ula
import matplotlib.pyplot as plt

'''
MIMO class

- Defines the MIMO model of the system
- Uses a LOS and NLOS communication model and a free space/3GPP path loss
- Considers Ptx, RBS, TBS, NRx, NTx as inputs to the system
- Computes Transmit Energy, Antenna Response vectors, Channel coefficients, SNR estimations etc.
'''

class MIMO:
    '''
    __init__(init_ptx, init_RBS, init_TBS, tr_antenna, rx_antenna)
    ptx - power transmitter level in dB
    RBS - Receiver Beam steering angle in degree
    TBS - Transmitter Beam steering angle in degree
    tx_antennas - No. of antenna elements at TX unit
    rx_antenns - No. of antenna elements at RX unit

    - Consider a fixed location with X_range=108, X_angle = 0
    - Consider a mmwave frequency for the system, freq=28GHz
    - Consider a fixed relative antenna element space, d=0.5

    '''

    def __init__(self, ue, gnB, sc_xyz, ch_model, init_ptx, tr_antennas, rx_antennas):
        self.freq = 30e9  # 28 GHz
        self.d = 0.5  # relative element space

        self.c = 3e8  # speed of light - 299792458 m s-1
        self.lmda = self.c / self.freq  # c - speed of light, scipy constant
        self.P_tx = init_ptx # dBm
        self.N_tx = tr_antennas  # no. of transmitting antennas
        self.N_rx = rx_antennas  # no. of receiving antennas

        # transmitter and receiver location Info
        self.alpha = 0  # relative rotation between transmit and receiver arrays

        self.df = 60 * 1e3  # 75e3  # carrier spacing frequency
        self.nFFT = 1200  # 2048  # no. of subspace carriers

        self.T_sym = 1 / self.df
        self.B = self.nFFT * self.df

        #print(ue, gnB)
        self.tx = np.array([ue])
        self.rx = np.array([gnB])
        self.sc_xyz = sc_xyz



        self.ch_model = ch_model
        self.channel = Channel(self.freq, self.tx, self.rx, self.sc_xyz, 'model',self.ch_model, 'nrx', self.N_rx, 'ntx', self.N_tx, 'nFFT', self.nFFT, 'df', self.df )
        #print("[MIMO] channel: ", self.channel.__dict__)
        self.channel.generate_paths()

        self.az_aoa = np.array(self.channel.az_aoa)
        self.el_aoa = np.array(self.channel.el_aoa)
        self.az_aod = np.array(self.channel.az_aod)
        self.el_aod = np.array(self.channel.el_aod)
        self.npath = self.channel.nlos_path +1

        '''
        self.X_i, self.X_j, self.X_k = np.array(ue)-np.array(gnB)
        self.Dist = np.sqrt(self.X_i ** 2 + self.X_j ** 2 + self.X_k**2)
        self.tau_k = self.Dist / self.c

        #if self.X_j > 0:
        #    self.theta_tx = math.acos(self.X_i / self.Dist)
        #else:
        #    self.theta_tx = 2*math.pi - math.acos(self.X_i / self.Dist)

        #self.phi_rx = self.theta_tx - math.pi + self.alpha
        if self.X_j > 0:
            self.phi_rx = math.acos(self.X_i / self.Dist)
        else:
            self.phi_rx = 2*math.pi - math.acos(self.X_i / self.Dist)

        self.theta_tx = self.phi_rx + math.pi - self.alpha
        '''

        #(self.az_aoa[0], self.el_aoa[0], dist) = cart2sph(self.tx[0, 0] - self.rx[0, 0], self.tx[0, 1] -
        #                                                  self.rx[0, 1], self.tx[0, 2] - self.rx[0, 2])
        #(self.az_aod[0], self.el_aod[0], dist) = cart2sph(self.rx[0, 0] - self.tx[0, 0], self.rx[0, 1] -
        #                                                 self.tx[0, 1], self.rx[0, 2] - self.tx[0, 2])
    '''
    Es = Transmit_Energy(ptx)
    ptx - power transmitter level in dB
    Es  - Transmit Energy in W

    - Computes the transmit energy of the MIMO model based on the fixed Ptx

    Assumptions:
        - Carrier Spacing frequency considered is 75KHz
        - No. of subspace carriers - 2048

    '''

    def Transmit_Energy(self):

        #self.P_tx = ptx
        Es = db2lin(self.P_tx) * (1e-3 / self.B)
        return Es


    '''
    N0 = Noise()
    Models the noise present in the channel

    Output:
    N0 - Noise model (AWGN for now)

    '''

    def Noise(self):
        N0dBm = -174  # mW/Hz
        N0 = db2lin(N0dBm) * (10 ** -3)  # in WHz-1
        return N0



    def Calc_Rate(self, Tf, RB_ang):  # best_RSSI_val):
        self.steps = 0
        Tf = Tf * 1e-3  # for msec
        ktf = np.ceil(Tf / self.T_sym)
        Tf_time = ktf * self.T_sym

        # calc_SNR
        Es = self.Transmit_Energy()
        h = self.channel.get_h()#self.Channel()

        #print("[MIMO] h: ", h.shape)

        # Noise for freq domain
        N0 = self.Noise()
        gau = np.zeros((self.N_rx, 1), dtype=np.complex)
        for i in range(gau.shape[0]):
            gau[i] = complex(np.random.randn(), np.random.randn())
        noise = np.sqrt(N0 / 2) * gau

        #print("[MIMO]theta_tx: {0}, phi_rx: {1}, RB_ang: {2}, TB_ang{3}".format(self.theta_tx, self.phi_rx, RB_ang, TB_ang))
        wRF = ula.steervec(self.N_rx, RB_ang[0], RB_ang[1]) #RB_ang-> (az_RB, el_RB)
        #print("[MIMO] wRF:{0}".format(wRF))
        #print("[MIMO] RB_ang[0]: {0}, RB_ang[1]: {1}".format(RB_ang[0], RB_ang[1]))
        # transmit beamforming vector

        #rssi_val = np.zeros(self.npath)
        #for j in range(rssi_val.shape[0]):
        fRF = ula.steervec(self.N_tx, self.az_aod[0], self.el_aod[0])
        #print("[MIMO] az_aod:{0}, el_aod:{1}, az_aoa:{2}, el_aoa:{3}".format(self.az_aod[0], self.el_aod[0], self.az_aoa[0], self.el_aoa[0]))
        #print("[MIMO] fRF:{0}".format(fRF))
        #print("[MIMO] channel: {0}".format(h.shape))

        #rssi_val = np.zeros(h.shape[2])
        energy_val = 0
        SNR = 0
        rate = 0.0
        for i in range(h.shape[2]):
            #energy_val = np.sqrt(self.N_tx*self.N_rx) * h * np.matmul(np.matmul(np.matmul(wRF.conj().T, a_rx), a_tx.conj().T),fRF)  # + np.matmul(wRF.conj().T, noise)
            #val = np.sqrt(self.N_tx * self.N_rx) * np.matmul(np.matmul(wRF.conj().T, h[:,:,i]), fRF)  # + np.matmul(wRF.conj().T, noise)
            rssi_val = np.abs(np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(h[:, :, i])).dot(fRF))**2#+ np.conj(wRF.T).dot(noise))
            #energy_val += val
            SNR = Es * rssi_val / N0
            rate += self.B/self.nFFT* np.log2(1 + SNR) * 1e-9  # in Gbit/s
        #print("[MIMO] energy_val: {0}", energy_val)
        #rssi_val[j] = energy_val

        #best_RSSI_val = energy_val#((energy_val.real) ** 2 + (energy_val.imag) ** 2)
        #self.steps = self.steps + 1

        #best_SNR = Es * best_RSSI_val / N0
        #print("[MIMO] learnt RSSI_val: ", best_RSSI_val)
        #print("[MIMO] SNR : {0}".format(20*np.log10(best_SNR)))
        #rate =  np.log2(1 + best_SNR) * 1e-9  # in Gbit/s
        rate = 5e1*rate
        return SNR, rate

    def Los_Rate(self):

        # calc_SNR
        Es = self.Transmit_Energy()
        h = self.channel.get_h()#self.Channel()

        #LoS Channel

        #channel = Channel(self.freq, self.tx, self.rx, sc_xyz, 'model',ch_model, 'nrx', self.N_rx, 'ntx', self.N_tx, 'nFFT', self.nFFT, 'df', self.df )
        #print("[MIMO] channel: ", self.channel.__dict__)
        #channel.generate_paths()
        #h = channel.get_h()


        N0 = self.Noise()
        gau = np.zeros((self.N_rx, 1), dtype=np.complex)
        for i in range(gau.shape[0]):
            gau[i] = complex(np.random.randn(), np.random.randn())
        noise = np.sqrt(N0 / 2) * gau

        a_tx = ula.steervec(self.N_tx, self.az_aod[0], self.el_aod[0])
        a_rx = ula.steervec(self.N_rx, self.az_aoa[0], self.el_aoa[0])#array_factor(self.d, self.phi_rx, self.N_rx)
        #plotbeam(self.az_aod[0], self.N_tx)
        #plotbeam(self.az_aoa[0], self.N_rx)
        #rssi_los = np.sqrt(self.N_rx*self.N_tx)*h * np.matmul(np.matmul(np.matmul(a_rx.conj().T, a_rx), a_tx.conj().T), a_tx)
        #rssi_los = complex(0, 0)
        #for i in range(h.shape[2]):
        #    val = np.sqrt(self.N_tx * self.N_rx) * np.dot#np.matmul(np.matmul(a_rx.conj().T, h[:, :, i]), a_tx)
        #    rssi_los+= val
        #rssi_los = rssi_los

        #best_RSSI_val = (rssi_los.real) ** 2 + (rssi_los.imag) ** 2
        #print("[MIMO]LoS best_RSSI_val: {0}".format(best_RSSI_val))
        rssi_los = 0
        SNR = 0
        rate = 0.0
        for i in range(h.shape[2]):
            rssi_val = np.abs(np.sqrt(self.N_rx*self.N_tx)* np.array(np.conj(a_rx.T).dot(h[:,:,i])).dot(a_tx))**2#+ np.conj(a_rx.T).dot(noise))
            #rssi_los += val
            SNR = Es * rssi_val / N0
            rate += self.B / self.nFFT * np.log2(1 + SNR) * 1e-9  # in Gbit/s
            #print("[MIMO]val", val)
            #break
        #best_RSSI_val = rssi_los
        rate= 5e1*rate
        return SNR, rate

def plotbeam(ang, n):
    w = ula.steervec(n, ang, 0)#np.array(array_factor(d,ang, n))
    #print(w.shape)
    wh = w.T.conj()
    r = np.arange(0, 1, 0.001)
    theta = 2* math.pi * r
    #wh= wh.reshape(,)
    #print(wh, w)
    gr = np.abs(np.array([wh.dot(ula.steervec(n, x, 0)) for x in theta]))#ula.steervec(n, x, 0)
    #print("gr:{0}".format(gr))
    #ax = plt.subplot(111, projection='polar')
    ##print(theta.shape, gr.shape)
    #ax.plot(theta, gr)
    #plt.show()
    return theta, gr

'''
y = array_factor(ang, N)
Computes response vectors of the antenna unit 

Parameters:
ang - Angle in degree
N - No. of antenna elements in the unit

Output:
y - Array Response vector

'''

'''
def array_factor(d, ang, N):
    x = np.arange(0, N)
    y = np.zeros((N,1), dtype=np.complex)
    for k in x:
        y[k] = 1 / np.sqrt(N) * np.exp(1j * 2 * math.pi * (d) * cmath.cos(ang) * k)
        y[k] = complex(np.around(y[k].real, decimals=4), np.around(y[k].imag, decimals=4))
    return y
'''

'''
    h = Channel()
    h - channel
    - Computes the channel coefficient and then multiplies to receive array response vector
    - 3GPP path loss channel models are followed here between the TX and RX
'''
'''
    def Channel(self):

        #We assume a 'free space path loss' for now
        FSL = 20 * np.log10(self.Dist) + 20 * np.log10(self.freq) - 147.55  # db, free space path loss
        channel_loss = db2lin(-FSL)
        g_c = np.sqrt(channel_loss)
        channel_coeff = g_c * cmath.exp(-1j * (math.pi / 4))  # LOS channel coefficient

        #Here a_tx is fixed in my channel, only 1 beam is transmitted
        #a_rx = self.array_factor(self.phi_rx, self.N_rx)
        h = channel_coeff#*a_rx

        return h
'''
'''
    y = Communication_vector(ang, n, level)

    - function to define tranmsit or receive unit norm vector

    Parameters:
    ang - beam steering angle in degree
    n - no. of antenna elements
    level - beam width level [0,1,2,3]

    Output:
    y - matrix of Unit normal vectors along columns
'''
'''
    #Beamforming Vector
    #assumption: Uniform Code book for now. No hierarchical levels are included
    def Communication_Vector(self, omega_vec, n):
        #n_xyz = [1, n, 1]
        n_xyz = [1, n, 1]
        #D = Uniform_Codebook(n_xyz, omega_vec)
        #return D
        #D = DFT_Codebook(n_xyz)
        #return D

        D_el = np.fft.fft(np.eye(n_xyz[2]))
        D_az = np.array([np.array(x) for x in omega_vec]) #np.zeros((n, len(omega_vec)), dtype=np.complex)
        #for k in range(len(omega_vec)):
        #    D_az[:, k] = self.root_beam(omega_vec[k], n).ravel()
        return np.kron(D_az, D_el)
'''

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
'''
    def root_beam(self, phi_m, N):
        phi_m = deg2rad(phi_m)  # converting deg to radians
        Na = N
        #Na = np.min([N, int(math.pow(3, level))])

        x = np.arange(0, Na)
        y = np.zeros((N, 1), dtype=np.complex)
        for k in x:
            y[k] = np.exp(1j * 2 * math.pi * (self.d/self.lmda) * cmath.sin(phi_m) * k)
        return y
'''

'''
    rssi_vec = Compute_RSSI(self, omega_vec, TB_r):

    - Computes RSSI based on the given TX and RX parameters

    Parameters:
    omega_vec - k-different beams from Beam set from RX side
    TB_r - Random transmitted beam direction on TX side

    Output:
    RSSI_vec - Vector of RSSI values for k different beams bw TX and RX
'''
'''
    def Compute_RSSI(self, omega_vec, TB_r):

        h = self.Channel()
        Es = self.Transmit_Energy()

        # Noise for freq domain
        N0 = self.Noise()
        gau = np.zeros((self.N_rx, 1), dtype=np.complex)
        for i in range(gau.shape[0]):
            gau[i] = complex(np.random.randn(), np.random.randn())
        noise = np.sqrt(N0 / 2) * gau

        a_tx = self.array_factor(TB_r, self.N_tx)
        w_mat = self.Communication_Vector(omega_vec, self.N_rx)  # no. of active antennas = no. of antennas

        # tx_level is assumed to be '1'. Hence rssi_val = (1,w_mat.shape[1])
        rssi_vec = np.zeros(1 * len(omega_vec))  # (tx levels*rx_levels)

        for i in range(rssi_vec.shape[0]):

            wRF = np.zeros((self.N_rx, 1), dtype=np.complex)
            wRF[:, 0] = w_mat[:, i].ravel()

            # Communication model for computing RSSI Energy
            # r_f = np.sqrt(self.P_tx) * np.matmul(np.matmul(wRF.conj().T, h), a_tx.conj().T)

            r_f = np.sqrt(Es) * (np.matmul(np.matmul(wRF.conj().T, h), a_tx.conj().T) + np.matmul(wRF.conj().T, noise))
            for j in range(self.N_rx):
                rssi_vec[i] += ((r_f[0, j].real) ** 2 + (r_f[0, j].imag) ** 2)

        self.best_RSSI_val = np.max(rssi_vec)
        # print(self.best_RSSI_val)
        return rssi_vec
'''

'''
        def Calc_Rate(self, Tf, RB_vec, TB_vec): #best_RSSI_val):
            self.steps = 0
            Tf = Tf * 1e-3  # for msec
            ktf = np.ceil(Tf / self.T_sym)
            Tf_time = ktf * self.T_sym

            # calc_SNR
            Es = self.Transmit_Energy()
            h = self.Channel()

            # Noise for freq domain
            N0 = self.Noise()
            gau = np.zeros((self.N_rx, 1), dtype=np.complex)
            for i in range(gau.shape[0]):
                gau[i] = complex(np.random.randn(), np.random.randn())
            noise = np.sqrt(N0 / 2) * gau

            a_tx = self.array_factor(self.theta_tx, self.N_tx)
            a_rx = self.array_factor(self.phi_rx, self.N_rx)

            w_mat = self.Communication_Vector(RB_vec, self.N_rx)  # no. of active antennas = no. of antennas
            print("[MIMO] w_mat:{0}".format(w_mat.shape))

            # transmit beamforming vector
            f_mat = self.Communication_Vector(TB_vec, self.N_tx)
            print("[MIMO] f_mat:{0}".format(f_mat.shape))

            # tx_level is assumed to be '1'. Hence rssi_val = (1,w_mat.shape[1])
            rssi_vec = np.zeros((len(TB_vec), len(RB_vec)))  # (tx levels, rx_levels)

            for i in range(rssi_vec.shape[1]):
                for j in range(rssi_vec.shape[0]):
                    wRF = np.zeros((self.N_rx, 1), dtype=np.complex)
                    wRF[:, 0] = w_mat[i, :].ravel()

                    fRF = np.zeros((self.N_tx, 1), dtype=np.complex)
                    fRF[:, 0] = f_mat[j, :].ravel()

                    #print("[MIMO] wRF: {0}, fRF: {1}, a_rx: {2}, a_tx: {3}".format(wRF.shape, fRF.shape, a_rx.shape, a_tx.shape))
                    print("[MIMO]1: ", np.matmul(wRF.conj().T, a_rx), a_tx.conj().T, fRF )
                    # Note!! this equation is different from that in 'ComputeRSSI' function
                    energy_val = np.sqrt(Es)*h * np.matmul(np.matmul(np.matmul(wRF.conj().T, a_rx), a_tx.conj().T), fRF) #+ np.matmul(wRF.conj().T, noise)
                    rssi_vec[j, i] += ((energy_val.real) ** 2 + (energy_val.imag) ** 2)
                    self.steps = self.steps + 1

            print("[MIMO] 2: ", rssi_vec)
            # np.argmax can also be used if rssi_vec.shape == (1,m)
            best_tx_ndx, best_rx_ndx = np.unravel_index(np.argmax(rssi_vec, axis=None), rssi_vec.shape)

            best_fRF = f_mat[best_tx_ndx, :].ravel()
            best_wRF = w_mat[best_rx_ndx, :].ravel()

            rssi_max = h * np.matmul(np.matmul(np.matmul(best_wRF.conj().T, a_rx), a_tx.conj().T), best_fRF)

            #print("[MIMO]: ", rssi_max)#, best_fRF.shape, best_wRF.shape)
            best_RSSI_val = (rssi_max.real) ** 2 + (rssi_max.imag) ** 2

            best_SNR = Es * best_RSSI_val / N0
            #print("[MIMO] SNR : {0}".format(20*np.log10(best_SNR)))
            rate = self.B * (1 - (self.steps) * self.T_sym / Tf_time) * np.log2(1 + best_SNR) * 1e-9  # in Gbit/s
            return best_SNR, rate
'''

'''
    def Exhaustive_Rate(self, Tf, steps, omega_vec, TB_r):
        #steps = 0

        Tf = Tf * 1e-3  # for msec
        ktf = np.ceil(Tf / self.T_sym)
        Tf_time = ktf * self.T_sym

        # calc_SNR
        Es = self.Transmit_Energy()
        h = self.Channel()

        #Noise for freq domain
        N0 = self.Noise()
        gau = np.zeros((self.N_rx,1), dtype=np.complex)
        for i in range(gau.shape[0]):
            gau[i] = complex(np.random.randn(), np.random.randn())
        noise = np.sqrt(N0 / 2) *gau

        a_tx = self.array_factor(self.theta_tx, self.N_tx)
        a_rx = self.array_factor(self.phi_rx, self.N_rx)

        w_mat = self.Communication_Vector(omega_vec, self.N_rx)  # no. of active antennas = no. of antennas
        #print("[MIMO] w_mat: {0}".format(w_mat.shape))

        # transmit beamforming vector
        f_mat = self.Communication_Vector([TB_r], self.N_tx)
        #print("[MIMO] f_mat: {0}".format(f_mat.shape))
        # tx_level is assumed to be '1'. Hence rssi_val = (1,w_mat.shape[1])
        rssi_vec = np.zeros((1, len(omega_vec)))  # (tx levels, rx_levels)

        for i in range(rssi_vec.shape[1]):
            for j in range(rssi_vec.shape[0]):
                wRF = np.zeros((self.N_rx, 1), dtype=np.complex)
                wRF[:, 0] = w_mat[i, :].ravel()

                fRF = np.zeros((self.N_tx, 1), dtype=np.complex)
                fRF[:, 0] = f_mat[j, :].ravel()
                # Communication model for computing RSSI Energy
                # r_f = np.sqrt(self.P_tx) * np.matmul(np.matmul(wRF.conj().T, h), a_tx.conj().T)
                #print("[MIMO] wRF: {0}, fRF: {1}, a_rx: {2}, a_tx: {3}".format(wRF.shape, fRF.shape, a_rx.shape, a_tx.shape))
                # Note!! this equation is different from that in 'ComputeRSSI' function
                energy_val = h*np.matmul(np.matmul(np.matmul(wRF.conj().T, a_rx), a_tx.conj().T), fRF) + np.matmul(wRF.conj().T, noise)
                #print(energy_val.shape)
                # for j in range(self.N_rx):
                rssi_vec[j, i] += ((energy_val.real) ** 2 + (energy_val.imag) ** 2)

                #steps = steps + 1

        print(rssi_vec)
        #np.argmax can also be used if rssi_vec.shape == (1,m)
        best_tx_ndx, best_rx_ndx = np.unravel_index(np.argmax(rssi_vec, axis=None), rssi_vec.shape)

        best_fRF = f_mat[best_tx_ndx, :].ravel()
        best_wRF = w_mat[best_rx_ndx, :].ravel()

        rssi_max = h*np.matmul(np.matmul(np.matmul(best_wRF.conj().T, a_rx), a_tx.conj().T), best_fRF)
        #print(rssi_max, best_fRF.shape, best_wRF.shape)
        best_RSSI_val = (rssi_max.real)**2 + (rssi_max.imag)**2

        #rssi_vec = rssi_vec.ravel()

        #print("[MIMO] TB_r: {0}".format(TB_r*180/math.pi))
        #print("[MIMO] rssi_vec: {0}".format(rssi_vec))
        #print("[MIMO] max_rssi_ndx: {0}, {1}".format(np.argmax(rssi_vec), np.max(rssi_vec)))
        #best_RSSI_val = np.max(rssi_vec)

        best_SNR = Es * best_RSSI_val / N0
        print("[MIMO] SNR : {0}".format(20*np.log10(best_SNR)))
        # print(Es, N0)
        # print("[MIMO]: rssi_opt_val: {0}, SNR: {1}".format(best_RSSI_val, self.SNR))
        # self.SNR = Es * best_RSSI_val / N0
        # print("[MIMO] SNR: {0}".format(self.SNR))
        rate = self.B * (1 - (steps) * self.T_sym / Tf_time) * np.log2(1 + best_SNR) * 1e-9  # in Gbit/s
        return best_SNR, rate
'''
'''
    def Calc_RateOpt(self, stepcount, Tf, ptx):
        Es = self.Transmit_Energy(ptx)
        h = self.Channel()
        N0 = self.Noise()

        # transmit array response vector
        a_tx = self.array_factor(self.theta_tx, self.N_tx)

        # receiver array response vector
        a_rx = self.array_factor(self.phi_rx, self.N_rx)

        SNROpt = h * np.sqrt(self.N_rx) * np.matmul(np.matmul(np.matmul(a_rx.conj().T, a_rx), a_tx.conj().T),
                                                    a_tx) * np.sqrt(self.N_tx)
        SNROpt = Es * ((SNROpt.real) ** 2 + (SNROpt.imag) ** 2) / N0
        SNROpt = SNROpt[0][0]

        Tf = Tf * 1e-3  # for msec
        ktf = np.ceil(Tf / self.T_sym)
        Tf_time = ktf * self.T_sym
        RateOpt = self.B * (1 - stepcount * self.T_sym / Tf_time) * np.log2(1 + SNROpt) * 1e-9  # in Gbit/s

        # print("SNROpt: {0}, RateOpt: {1}".format(SNROpt, RateOpt))

        return RateOpt
'''
