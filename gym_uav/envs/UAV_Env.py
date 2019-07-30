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


class UAV_Env(gym.Env):
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
        self.ue_v = 10

        self.gNB = np.array([[0,0,0]])#, [20,30,0], [40,60,0]]
        self.sc_xyz= np.array([])
        self.ch_model= 'fsp'
        self.N = self.N_rx #Overall beam directions
        self.BeamSet = Generate_BeamDir(self.N) #Set of all beam directions

        #Observation - RSSI information of states
        self.state = None
        self.rate = None
        self.rate_threshold = 0.07  # good enough QoS val (Rate)
        self.Nhops = 5

        self.seed()
        low_obs = np.array([30.0, 0.0, 0.0])
        self.high_obs = np.array([100.0, 3.14159, 3.14159])
        self.obs_space = spaces.Box(low=low_obs,high=self.high_obs)

        self.act_space = spaces.Discrete(self.N)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.act_space.contains(action), "%r (%s) invalid" % (action, type(action))

        state = self.state * self.high_obs
        dist, ue_ang, rbd = state
        rbs = self.BeamSet[action]

        ue_pos = np.array(sph2cart(ue_ang, 0, dist)) #ue_pos is(x,y)

        ue_pos[0] += self.ue_v

        new_dist = np.sqrt(ue_pos[0]**2 + ue_pos[1]**2) #x**2 + y**2
        new_ang = np.arctan2(ue_pos[1],ue_pos[0])
        self.state = np.array([new_dist, new_ang, rbs]) / self.high_obs

        self.mimo_model = MIMO(ue_pos, self.gNB[0], self.sc_xyz, self.ch_model, self.ptx, self.N_tx, self.N_rx)
        prev_rate = self.rate
        prev_dist = dist
        self.SNR, self.rate = self.mimo_model.Calc_Rate(self.SF_time, np.array([rbs, 0]))#rkbeam_vec, tbeam_vec )

        self.steps_done += 1

        rwd = self._reward(prev_rate, prev_dist, new_dist)
        #print("[uav_env] rwd: {}".format(rwd))
        done = self._gameover()

        return self.state, rwd, done, {}

    def reset(self):
        # Note: should be a uniform random value between starting 4-5 SNR states
        #self.TB_r = get_TBD(ue, self.alpha)#Gen_RandomBeams(1, self.N)[0]  # one random TX beam
        dist = np.random.uniform(low=30.0, high=50.0)#self.np_random.uniform(low=30.0, high=50.0)
        TBD = np.random.uniform(low=0.0, high=3.14159)#self.np_random.uniform(low=0.0, high=3.14159)
        RBD = Gen_RandomBeams(1, self.N)  # one random RX beam
        self.state = np.array([dist, TBD, RBD]) / self.high_obs

        self.steps_done = 0
        self.rate = 0

        return np.array(self.state)

    def render(self, mode='human', close=False):
        pass

    def _reward(self, prev_rate, prev_dist, cur_dist):

        #bf_condn = False
        #if ((prev_rate >= self.rate) and (prev_dist <= cur_dist)) or ((prev_rate <= self.rate) and (prev_dist >= cur_dist)):
        #    bf_condn = True
        #if (self.rate > self.rate_threshold) and (bf_condn is True):
        #    return 10*(self.rate-self.rate_threshold)+8#10+ self.rate-self.rate_threshold-1
        #elif (self.rate > self.rate_threshold) and (bf_condn is False):
        #    return 3
        #else:
        #    return -3
        #(az_aod, temp, temp) = cart2sph(rx[0] - tx[0], rx[1] - tx[1], rx[2] - tx[2])
        #if az_aod == rb_ang:
        #    return 1
        #else:
        #    return 0
        #val = np.abs(az_aod-rb_ang)
        #print("[uav_env] val: {}, az_aod: {}, rbs: {}", val, az_aod, rb_ang)
        #if (val >= (np.pi)):
        #    return 1+ np.log10((2*val/(np.pi))-1)  #1+log10(2x/pi -1)
        #else:
        #    return 0
        if(self.rate > self.rate_threshold):
            return 100*self.rate +3
        else:
            return 100*self.rate -3

    def _gameover(self):
        return (self.steps_done == self.Nhops)

    def get_Los_Rate(self, state):
        dist, ue_ang, rbd = (state * self.high_obs)
        ue_pos = np.array(sph2cart(ue_ang, 0, dist))  # ue_pos is(x,y)

        sc_xyz = np.array([])
        ch_model = 'fsp'

        mimo_model = MIMO(ue_pos, self.gNB[0], sc_xyz, ch_model, self.ptx, self.N_tx, self.N_rx)
        SNR, rate = mimo_model.Los_Rate()  # rkbeam_vec, tbeam_vec )

        return SNR, rate

    def get_Exh_Rate(self, state):
        dist, ue_ang, rbd = (state*self.high_obs)
        ue_pos = np.array(sph2cart(ue_ang, 0, dist))  # ue_pos is(x,y)

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

def Gen_RandomBeams(k, N):

    BeamSet = Generate_BeamDir(N)
    ndx_lst = np.random.randint(0,N, k)
    #print('B', BeamSet[4])
    k_beams = BeamSet[ndx_lst[0]]#[BeamSet[x] for x in ndx_lst]
    return np.array(k_beams)
