import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from Source.MIMO import MIMO
#from Source.Misc import *
from Source.miscfun.geometry import *


''''
####################################
    MAIN CLASS ENVIRONMENT
####################################
UAV_Env - UAV Environment

Model Characteristics:
- Considers a MIMO model with mmwave frequency
- Considers a fixed Ptx, Ntx, NRx
- Chooses
  Beam steering vectors- Receiver Beam Steering (RBS), Transmitter Beam Steering Vectors (TBS) 
  Beam Directions - Transmitter Beam Direction (TBD), Receiver Beam Direction (RBD)
  Distance of UAV from gnB (D)
  as the main parameters for this RF Beam model 

- Observation space - [0,1,2,.....,179] -> [-120, -119, -118,......,60]
- Action space - [0,1,2,.......5624] -> [(-60,-60,1,1), ......(RBS,TBS,RBeamWidth,TBeamWidth).......(60,60,3,3)]

- Transmit Power= 30 dB, N_tx= 1, N_rx=8
'''


class UAV_Env_v2(gym.Env):
    """
    Description:
    A UAV moves in a region around the base station. The problem is to provide the UAV with best possible QoS over N steps


    Observation:
        Type: Box(3,)
        Num Observation     Min     Max
        0   distance (D)    -100.0  100.0
        1   TBD               0.0   3.14159
        2   RBD               0.0   3.14159

    Action:
        Type:Discrete(Nrx)
        Num     Action
        0       Bdir 0
        1       Bdir 1
        ...     ....
        Nrx-1   Bdir {Nrx-1}

    Reward:
        Reward is rate value computed for every step taken, including the termination step. Rate value measured is [0.0, 4.0]

    Starting State:
        All observations are assigned a uniform random value in their respective Min Max range

    Episode Termination:
        When UAV makes N hops or N steps from the starting state
    """

    def __init__(self):

        self.N_tx = 1 # Num of transmitter antenna elements
        self.N_rx = 8  # Num of receiver antenna elements
        self.count = 0
        self.ptx = 30  #dB
        self.SF_time = 20 #msec - for 60KHz carrier frequency in 5G
        self.alpha = 0

        # (x1,y1,z1) of UE_source location
        self.ue_s = None#[10,15,0]
        #self.ue_v = None


        self.gNB = np.array([[0,0,0]])#, [20,30,0], [40,60,0]]
        self.sc_xyz= np.array([])
        self.ch_model= 'fsp'
        self.N = self.N_rx #Overall beam directions
        self.BeamSet = Generate_BeamDir(self.N) #Set of all beam directions

        #Observation - RSSI information of states
        self.state = None
        self.rate = None
        self.rate_threshold = None  # good enough QoS val (Rate)
        self.Nhops = 5

        #UE information
        self.ue_xloc = np.arange(-500, 500, 50)  #10 locs
        self.ue_yloc = np.arange(50,550, 50)     #5 locs
        self.ue_vx = np.array([-50, 100,50]) #3 vel parameters
        self.ue_vy = np.array([-50, 100, 50]) #3 vel parameters
        self.ue_xdest = np.array([450]) # 1 x-dest loc
        self.ue_ydest = np.array([450]) # 1 y-dest loc


        self.seed()
        #low_obs = np.array([-500, 0, 0.0, 10.0, 10.0])
        self.high_obs = np.array([np.max(self.ue_xloc), np.max(self.ue_yloc), np.max(self.ue_vx), np.max(self.ue_vy), np.pi, np.max(self.ue_xdest), np.max(self.ue_ydest)])
        self.obs_space = spaces.MultiDiscrete([len(self.ue_xloc), #ue_xloc
                                               len(self.ue_yloc), #ue_yloc
                                               len(self.ue_vx), #ue_vx
                                               len(self.ue_vy), #ue_vy
                                               self.N, #N beam directions
                                               len(self.ue_xdest), #ue_xdest
                                               len(self.ue_ydest)]) #ue_ydest

        self.act_space = spaces.Discrete(self.N)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.act_space.contains(action), "%r (%s) invalid" % (action, type(action))

        state = np.rint(self.state[0] * self.high_obs)
        ue_xloc, ue_yloc, ue_vx, ue_vy, rbd, ue_xdest, ue_ydest = state
        rbs = self.BeamSet[action]

        new_ue_xloc = ue_xloc + ue_vx
        new_ue_yloc = ue_yloc + ue_vy
        new_ue_pos = np.array([new_ue_xloc, new_ue_yloc, 0])

        self.state = np.array([new_ue_xloc, new_ue_yloc, ue_vx, ue_vy, rbs, ue_xdest, ue_ydest]) / self.high_obs
        self.state = self.state.reshape((1, len(self.state)))

        self.mimo_model = MIMO(new_ue_pos, self.gNB[0], self.sc_xyz, self.ch_model, self.ptx, self.N_tx, self.N_rx)
        prev_rate = self.rate
        prev_dist = np.sqrt((ue_xloc-ue_xdest)**2 + (ue_yloc-ue_ydest)**2) #x**2 + y**2
        self.SNR, self.rate = self.mimo_model.Calc_Rate(self.SF_time, np.array([rbs, 0]))#rkbeam_vec, tbeam_vec )

        self.steps_done += 1

        #rwd = self._reward(prev_dist)
        #print("[uav_env] rwd: {}".format(rwd))
        rwd, done = self._gameover(prev_dist)

        #if status == 1: #game over
        #    done = True
        #elif (status == -1) or (status == -2): #uav crossing boundaries
        #    rwd = -2.0
        #    done = True
        #else:
        #    done = False

        return self.state, rwd, done, {}

    def reset(self):
        # Note: should be a uniform random value between starting 4-5 SNR states
        #self.TB_r = get_TBD(ue, self.alpha)#Gen_RandomBeams(1, self.N)[0]  # one random TX beam
        state_indices = self.obs_space.sample()
        xloc_ndx, yloc_ndx, vx_ndx, vy_ndx, rbd_ndx, xdest_ndx, ydest_ndx = state_indices

        self.state = np.array([self.ue_xloc[xloc_ndx],
                               self.ue_yloc[yloc_ndx],
                               self.ue_vx[vx_ndx],
                               self.ue_vy[vy_ndx],
                               self.BeamSet[rbd_ndx],
                               self.ue_xdest[xdest_ndx],
                               self.ue_ydest[ydest_ndx]])


        self.steps_done = 0
        self.rate = 0

        #Computing the rate threshold for the given destination
        ue_dest = np.array([self.state[-2], self.state[-1], 0])
        dest_mimo_model = MIMO(ue_dest, self.gNB[0], self.sc_xyz, self.ch_model, self.ptx, self.N_tx, self.N_rx)
        dest_SNR = []
        dest_rates = []
        for rbeam in self.BeamSet:  # rbeam_vec:
            SNR, rate = dest_mimo_model.Calc_Rate(self.SF_time, np.array([rbeam, 0]))
            dest_SNR.append(SNR)
            dest_rates.append(rate)
        self.rate_threshold = np.max(dest_rates)

        self.state = self.state / self.high_obs
        self.state = self.state.reshape((1, len(self.state)))
        return self.state

    def render(self, mode='human', close=False):
        pass

    def _reward(self, prev_dist):

        #bf_condn = False
        #if ((prev_rate >= self.rate) and (prev_dist <= cur_dist)) or ((prev_rate <= self.rate) and (prev_dist >= cur_dist)):
        #    bf_condn = True
        #if (self.rate > self.rate_threshold) and (bf_condn is True):
        #    return 10*(self.rate-self.rate_threshold)+8#10+ self.rate-self.rate_threshold-1
        #elif (self.rate > self.rate_threshold) and (bf_condn is False):
        #    return 3
        #else:
        #    return -3

        ue_dist = np.sqrt((self.state[0][0]-self.state[0][-2]) ** 2 + (self.state[0][1]--self.state[0][-1]) ** 2)
        #ue_dest_dist = np.sqrt(self.state[0][-2]**2 + self.state[0][-1]**2)

        if (self.rate >= self.rate_threshold) and (ue_dist <= prev_dist):
            return 10*self.rate + 3
        else:
            return 0.0#10*self.rate - 3

    def _gameover(self, prev_dist):
        #ue_dist = np.sqrt(self.state[0][0]**2 + self.state[0][1]**2)
        #ue_dest_dist = np.sqrt(self.state[0][-2]**2 + self.state[0][-1]**2)
        #return ue_dist >= ue_dest_dist
        state = np.rint(self.state[0] * self.high_obs)
        ue_dist = np.sqrt((state[0] - state[-2]) ** 2 + (state[1] - -state[-1]) ** 2)


        if (self.rate >= self.rate_threshold) and (ue_dist == 0):
            rwd = 15.0
            done = True
        elif (self.rate >= self.rate_threshold) and (ue_dist <= prev_dist):
            rwd = 500*self.rate + 3
            done = False
        elif (state[0] < np.min(self.ue_xloc)) or (state[0] > np.max(self.ue_xloc)):
            rwd = -2.0
            done = False
        elif (state[1] < np.min(self.ue_yloc)) or (state[1] > np.max(self.ue_yloc)):
            rwd = -2.0
            done = False
        else:
            rwd = 0.0
            done = False
        #if (state[0] == state[-2] ) and (state[1] == state[-1]):
        #    return 1#True
        #elif (state[0] < np.min(self.ue_xloc)) or (state[0] > np.max(self.ue_xloc)):
        #    return -1#True
        #elif (state[1] < np.min(self.ue_yloc)) or (state[1] > np.max(self.ue_yloc)):
        #    return -2#True
        #else:
        #    return 0#False
        return rwd, done

    def get_Los_Rate(self, state):

        state = np.rint(state[0] * self.high_obs)
        ue_xloc, ue_yloc, _, _, _, _, _ = state

        sc_xyz = np.array([])
        ch_model = 'fsp'
        ue_pos = np.array([ue_xloc, ue_yloc, 0])

        mimo_model = MIMO(ue_pos, self.gNB[0], sc_xyz, ch_model, self.ptx, self.N_tx, self.N_rx)
        SNR, rate = mimo_model.Los_Rate()  # rkbeam_vec, tbeam_vec )

        return SNR, rate

    def get_Exh_Rate(self, state):
        state = np.rint(state[0] * self.high_obs)
        ue_xloc, ue_yloc, _, _, _, _, _ = state
        ue_pos = np.array([ue_xloc, ue_yloc,0])

        mimo_exh_model = MIMO(ue_pos, self.gNB[0], self.sc_xyz, self.ch_model, self.ptx, self.N_tx, self.N_rx)
        #rbeam_vec = self.BeamSet#Generate_BeamDir(self.N)
        exh_SNR = []
        exh_rates = []

        for rbeam in self.BeamSet:#rbeam_vec:
            SNR, rate = mimo_exh_model.Calc_Rate(self.SF_time, np.array([rbeam, 0]))
            exh_SNR.append(SNR)
            exh_rates.append(rate)

        best_rbeam_ndx = np.argmax(exh_rates)
        return self.BeamSet[best_rbeam_ndx], np.max(exh_rates) #(Best RBS, Best Rate)

    def get_Rate(self):
        return self.rate


def Generate_BeamDir(N):
    min_ang = 0#-math.pi/2
    max_ang = np.pi#math.pi/2
    step_size = (max_ang-min_ang)/N

    BeamSet = []#np.zeros(N)#np.fft.fft(np.eye(N))

    #B_i = (i)pi/(N-1), forall 0 <= i <= N-1; 0< beta < pi/(N-1)
    val = min_ang
    for i in range(N):
        BeamSet.append(val + (i+1)*step_size)#(i+1)*(max_ang-min_ang)/(N)

    return np.array(BeamSet) #eval(strBeamSet_list)#np.ndarray.tolist(BeamSet)

'''
def Gen_RandomBeams(k, N):

    BeamSet = Generate_BeamDir(N)
    ndx_lst = np.random.randint(0,N, k)
    #print('B', BeamSet[4])
    k_beams = BeamSet[ndx_lst[0]]#[BeamSet[x] for x in ndx_lst]
    return np.array(k_beams)
'''