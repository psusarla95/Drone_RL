import numpy as np
from Source.MIMO import MIMO
from Source.Codebook import *
from Source.Misc import *
import random


'''
*************
    NOTE
*************
This is an RF_Env which doesn't depend upon UE position of drone as the state but a (TBD, RBD) pair. 
TBD direction is assumed to be pointing to the gNB at every instant.

The Env has fixed number of UE hops from distance D1 to DK with k being the number of hops.
The Env also assumes a particular channel model LOS/ NLOS/ UMA-LOS/ SC_LOS etc.

'''

''''
####################################
    MAIN CLASS ENVIRONMENT
####################################
RFBeamEnv - RF Beam Environment

Model Characteristics:
- Considers a MIMO model with mmwave frequency
- Considers a fixed Ptx, Beam Width Level (level), Ntx, Nrx
- Transmission Beam Direction (TBD), Receiver Beam Direction (RBD) - Parameters of the model 

'''

class RFBeamEnv:
    #metadata = {'render.modes': ['human']}

    def __init__(self, Env_Info):

        self.N_tx = Env_Info['N_tx'] # Num of transmitter antenna elements
        self.N_rx = Env_Info['N_rx']  # Num of receiver antenna elements
        self.ptx = Env_Info['ptx']  #dB
        self.SF_time = Env_Info['SF_time'] #msec - for 60KHz carrier frequency in 5G
        self.alpha = Env_Info['alpha']
        self.count = 0

        # (x1,y1,z1) of UE_source location
        self.ue_s = Env_Info['ue_s']#[10,15,0]
        self.ue_v = Env_Info['ue_v']#10
        self.goal = Env_Info['ue_d']
        #gNB locations One Serving gNB_1 node, 2 visible gnB_2,gNB_3 nodes
        self.gNB = Env_Info['gnB']

        self.sc_xyz= Env_Info['sc_xyz']
        self.ch_model= Env_Info['ch_model']

        self.N = self.N_rx #Overall beam directions
        self.BeamSet = Generate_BeamDir(self.N) #Set of all beam directions

        #Observation - RSSI information of states
        self.obs = None
        self.rate = None
        self.rate_threshold = 0.7 # good enough QoS val (Rate)

        #self.seed()
        #self.viewer = None

    def GenAction_Sample(self):
        rk_beams = Gen_RandomBeams(1, self.N)
        return rk_beams


    '''
    state, reward, done, {} - step(action)
    - A basic function prototype of an Env class under gym
    - This function is called every time, the env needs to be updated into a new state based on the applied action

    Parameters:
    action - the action tuple applied by the RL agent on Env current state

    Output:
    state - The new/update state of the environment
    reward - Env reward to RL agent, for the given action
    done - bool to check if the environment goal state is reached
    {} - empty set

    '''

    def step(self, action):
        # check the legal move first and then return its reward
        self.ue_s[0] += self.ue_v #+ delta_p
        self.obs[1] = np.array(action)
        self.obs[0] = np.array(self.TB_r)#self.ue_s)


        self.mimo_model = MIMO(self.ue_s, self.gNB[0], self.sc_xyz, self.ch_model, self.ptx, self.N_tx, self.N_rx)

        self.SNR, self.rate = self.mimo_model.Calc_Rate(self.SF_time, np.array([self.obs[1][0], 0]))#rkbeam_vec, tbeam_vec )
        #print("[RF_Env] SNR: {0}, rate: {1}".format(20*np.log10(self.SNR), self.rate))

        self.cum_rate += self.rate

        self.count += 1
        rwd = self.Reward()
        done = self.Game_Over()
        return self.obs, rwd, done

    '''
    game_state = game_over(s)
    - Function to check if the agent has reached its goal in the environment

    Parameters:
    s - current state of the environment

    Output:
    game_state {    False       if s < goal_state
                    True        if s = goal_state
    '''

    #Define the reward function here
    def Reward(self):
        return self.rate

    def Game_Over(self):
        #print("gameover: obs {0}, {1}".format(self.obs[0], self.ue_s[0] + self.ue_tdist))
        #print("[Env] ue_s: {0}, goal: {1}".format(self.ue_s, self.goal))
        #return (self.ue_s == self.goal)#(self.obs[0] == (self.ue_s[0]+self.ue_tdist))
        return np.array_equal(self.ue_s, self.goal)
        #self.cum_rate += self.rate
        #self.cum_Los_rate += self.Los_rate
        #return #np.around(self.Los_rate-self.rate, decimals=4) <= self.goal_diff)#np.round(self.cum_Los_rate/self.cum_rate) <= self.rate_ratio)
    '''
      reset()
      - Resets the environment to its default values
      - Prototype of the gym environment class  
      '''

    def reset(self, ue, vel):
        # Note: should be a uniform random value between starting 4-5 SNR states
        self.TB_r = get_TBD(ue, self.alpha)#Gen_RandomBeams(1, self.N)[0]  # one random TX beam
        self.RB_r = Gen_RandomBeams(1, self.N)  # one random RX beam
        # print(self.TB_r)

        self.obs = [self.TB_r, self.RB_r]
        self.ue_s = np.array(ue)
        self.ue_v = vel
        self.count = 0
        self.rate = 0
        self.cum_rate = 0
        self.cum_Los_rate = 0
        return np.array(self.obs)


    def set_goal(self, ue_d):
        self.goal = np.array(ue_d)
        return

    def set_velocity(self, vel):
        self.ue_v = vel
        return

    def set_rate_threshold(self, rate_th):
        self.rate_threshold = rate_th
        return

    def get_Rate(self):
        return self.rate

    def get_LoS_Rate(self, ue_s):
        sc_xyz = np.array([])
        ch_model = 'fsp'
        mimo_model = MIMO(ue_s, self.gNB[0], sc_xyz, ch_model, self.ptx, self.N_tx, self.N_rx)
        #print("[Env]: LoS h: {0}".format(mimo_model.channel.pathloss))
        Los_SNR, Los_rate = mimo_model.Los_Rate()
        return Los_rate

    def get_Exh_Rate(self, ue_s):
        #print("[RF_Env] obs: {0}".format(self.obs))
        #print("[RF_Env] TB_r: {0}".format(self.TB_r))
        self.mimo_exh_model = MIMO(ue_s, self.gNB[0], self.sc_xyz, self.ch_model, self.ptx, self.N_tx, self.N_rx)
        #print("[Env]: exh h: {0}".format(self.mimo_exh_model.channel.pathloss))
        rbeam_vec = Generate_BeamDir(self.N)
        tbeam_vec = Generate_BeamDir(self.N)
        exh_SNR = []
        exh_rates=[]

        for rbeam in rbeam_vec:
            SNR, rate = self.mimo_exh_model.Calc_Rate(self.SF_time, np.array([rbeam[0], 0]))
            exh_SNR.append(SNR)
            exh_rates.append(rate)

        best_rbeam_ndx = np.argmax(exh_rates)
        return rbeam_vec[best_rbeam_ndx], np.max(exh_rates)

def Generate_BeamDir(N):
    min_ang = 0#-math.pi/2
    max_ang = math.pi#math.pi/2
    step_size = (max_ang-min_ang)/N

    BeamSet = []#np.zeros(N)#np.fft.fft(np.eye(N))

    #B_i = (i)pi/(N-1), forall 0 <= i <= N-1; 0< beta < pi/(N-1)
    val = min_ang
    for i in range(N):
        BeamSet.append(np.array([val + (i+1)*step_size]))#(i+1)*(max_ang-min_ang)/(N)

    return np.array(BeamSet) #eval(strBeamSet_list)#np.ndarray.tolist(BeamSet)

def Gen_RandomBeams(k, N):

    BeamSet = Generate_BeamDir(N)
    ndx_lst = np.random.randint(0,N, k)
    #print('B', BeamSet[4])
    k_beams = BeamSet[ndx_lst[0]]#[BeamSet[x] for x in ndx_lst]
    return np.array(k_beams)

def get_TBD(ue, alpha):
    dist = np.sqrt(ue[0] ** 2 + ue[1] ** 2)
    if ue[1] > 0:
        phi_rx = math.acos(ue[0] / dist)
    else:
        phi_rx = 2*math.pi - math.acos(ue[1] / dist)

    theta_tx = phi_rx + math.pi - alpha
    theta_tx = np.around(theta_tx, decimals=8)
    return theta_tx

'''
def Generate_BeamDir(N):
    min_ang = 0
    max_ang = math.pi

    BeamSet = np.fft.fft(np.eye(N))
    #print("before: {0}".format(BeamSet))
    rBeamSet_list = []
    for bvec in BeamSet:
        #for i in range(len(bvec)):
            #bvec[i] = complex(np.around(bvec[i].real, decimals=4), np.around(bvec[i].imag, decimals=4))
        bvec = np.around(bvec, decimals=4)
        #bvec_list = []
        #for num in bvec:
        #    bvec_list.append(num)
        rBeamSet_list.append(bvec.tolist())
    #rBeamSet = np.array(rBeamSet_list)
    #B_i = (i)pi/(N-1), forall 0 <= i <= N-1; 0< beta < pi/(N-1)
    #for i in range(BeamSet.shape[0]):
    #    BeamSet[i] = #i*(max_ang-min_ang)/(N-1)
    strBeamSet_list = str(rBeamSet_list)
    return eval(strBeamSet_list)#np.ndarray.tolist(BeamSet)

def Gen_RandomBeams(k, N):

    BeamSet = Generate_BeamDir(N)
    ndx_lst = np.random.randint(0,N, k)
    #print('B', BeamSet[4])
    k_beams = [BeamSet[x] for x in ndx_lst]
    return k_beams
'''

'''
    def GenObs_Sample(self):
        #ue_r = np.random.randint(self.ue_s[0], self.ue_s[0]+self.ue_tdist)
        rnd_ndx = np.random.randint(0, len(self.obs_space))
        ue_r = self.obs_space[rnd_ndx]
        #ue_r = (ue_ri, self.ue_s[1],self.ue_s[2])

        #omega_rvec = Gen_RandomBeams(self.K, self.N)
        #print("beams: {0}".format(omega_rvec))
        #obs_RIM = self.Compute_RIM(ue_r, omega_rvec)

        return ue_r#, obs_RIM
'''
'''

    #Compute RSSI Information Matrix

    def Compute_RIM(self, ue, omega_vec):
        #compute rssi information of each ue_x and gnB location, forming a 3kx1 nd.array
        self.mimo_models = []
        RIM = np.zeros(3*self.K) #(3k, 1), k set of receive beams each instant

        for i in range(len(self.gNB)):
            self.mimo_models.append(MIMO(ue, self.gNB[i], self.ptx, self.N_tx, self.N_rx))
            RIM[i*self.K: (i+1)*self.K] = self.mimo_models[i].Compute_RSSI(omega_vec, self.TB_r)

        return RIM
    '''