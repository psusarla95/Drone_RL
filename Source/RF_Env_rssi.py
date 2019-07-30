import numpy as np
from Source.MIMO import MIMO
from Source.Misc import *
import random
''''
####################################
    MAIN CLASS ENVIRONMENT
####################################
RFBeamEnv - RF Beam Environment

Model Characteristics:
- Considers a MIMO model with mmwave frequency
- Considers a fixed Ptx and chooses Beam steering vectors- Receiver Beam Steering (RBS),
  Transmitter Beam Steering Vectors (TBS), Beam Width Level (level) as the main parameters for this RF Beam model 

RL Method: Q-learning
- states - integer SNR values (dB); actions- (RBS,TBS,RBeamWidth,TBeamWidth);
- Observation Threshold state: 30 dB (SNR), Observation goal state: 60 dB (SNR)
                5*exp((curr_state-goal_state)/10) if curr_state>= observation_thr_state;
  Rewards = {   5                                 if curr_state = observation_goal_state;
                -1                                 otherwise      
- RBS span -[-60,60,5] (deg), TBS span-[-60,60,5](deg), RBeamWidth span- [1,3,1] (level), TBeamWidth span- [1,3,1] (level), SNR span-[-120,60,1] (dB)
- Observation space - [0,1,2,.....,179] -> [-120, -119, -118,......,60]
- Action space - [0,1,2,.......5624] -> [(-60,-60,1,1), ......(RBS,TBS,RBeamWidth,TBeamWidth).......(60,60,3,3)]

- Transmit Power= 46 dB, N_tx= 16, N_rx=16
'''


class RFBeamEnv:
    #metadata = {'render.modes': ['human']}

    def __init__(self):

        self.N_tx = 16 # Num of transmitter antenna elements
        self.N_rx = 16  # Num of receiver antenna elements
        self.count = 0
        self.ptx = 30  #dB
        #self.level = 1
        #self.state = None

        # (x1,y1,z1) of UE_source location
        self.ue_s = [10,15,0]
        self.ue_v = 10
        self.ue_tdist = 90
        self.cur_ue = None

        #gNB locations One Serving gNB_1 node, 2 visible gnB_2,gNB_3 nodes
        self.gNB = [[0,0,0], [20,30,0], [40,60,0]]

        #Action space parameters: |A| = 8C4 x |delta_p| =70 x 5 = 350
        self.Actions = {
            'K': 4,  #Beam_set length
            'N': 8,  #Overall beam directions
            'delta_p': [0,-1,+1,-2,+2] #position control space along x-direction
        }
        self.K = self.Actions['K']  #Beam_set length
        self.N = self.Actions['N']  #Overall beam directions
        self.delta_p = self.Actions['delta_p'] #position control space along x-direction
        self.BeamSet = Generate_BeamDir(self.N) #Set of all beam directions
        #self.beta =  math.pi/(2*(self.N-1)) # Beamwidth beta, 0 < beta <= (pi / (N - 1))

        #State of the system - UE_t w.r.t gnB_1
        self.possible_states = [(x,self.ue_s[1],self.ue_s[2]) for x in range(self.ue_s[0]-2,self.ue_s[0]+self.ue_tdist+3,1)]

        #Observation - RSSI information of states
        self.obs = []


        #Action-Observation Mapping (Q_table)

        #self.observation_values = Custom_Space_Mapping(self.Observations)
        #self.rev_observation_values = dict((v[0], k) for k, v in self.observation_values.items())
        #self.num_observations = len(self.observation_values.keys())
        #self.min_state = self.observation_values[0][0]  # minimum SNR state

        #self.action_values = Custom_Space_Mapping(self.Actions)
        #self.num_actions = len(self.action_values.keys())
        #self.action_space = spaces.Discrete(self.num_actions) #I have to define this
        #self.observation_space = spaces.Discrete(self.num_observations) #I can avoid this
        # self.max_state = self.observation_values[self.num_observations-1][0]#maximum SNR state
        self.rate_threshold = 23  # good enough QoS val (Rate)

        #self.seed()
        #self.viewer = None

    def GenAction_Sample(self):
        delta_p = self.Actions['delta_p']
        rp_ndx = np.random.randint(0, len(delta_p))
        rdelta_p = delta_p[rp_ndx]
        rk_beams = Gen_RandomBeams(self.K, self.N)
        rk_beams.append(rdelta_p)
        return rk_beams

    def GenObs_Sample(self):
        #ue_r = np.random.randint(self.ue_s[0], self.ue_s[0]+self.ue_tdist)
        rnd_ndx = np.random.randint(0, len(self.possible_states))
        ue_r = self.possible_states[rnd_ndx]
        #ue_r = (ue_ri, self.ue_s[1],self.ue_s[2])

        omega_rvec = Gen_RandomBeams(self.K, self.N)
        #print("beams: {0}".format(omega_rvec))
        obs_RIM = self.Compute_RIM(ue_r, omega_rvec)

        return ue_r, obs_RIM

    '''
    Compute RSSI Information Matrix
    '''
    def Compute_RIM(self, ue, omega_vec):
        #compute rssi information of each ue_x and gnB location, forming a 3kx1 nd.array
        self.mimo_models = []
        RIM = np.zeros(3*self.K) #(3k, 1), k set of receive beams each instant

        for i in range(len(self.gNB)):
            self.mimo_models.append(MIMO(ue, self.gNB[i], self.ptx, self.N_tx, self.N_rx))
            RIM[i*self.K: (i+1)*self.K] = self.mimo_models[i].Compute_RSSI(omega_vec, self.TB_r)

        return RIM

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
        # if action in self.actions[self.current_state]:
        #assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        kbeam_vec, delta_p = action[:self.K], action[self.K]

        prev_ue = self.cur_ue[:]
        self.cur_ue[0] += self.ue_v + delta_p

        RIM = self.Compute_RIM(self.cur_ue, kbeam_vec)

        rssi_gnb1 = RIM[:self.K]
        best_rssi_val = np.max(rssi_gnb1)
        #print('[RF_Env] best_rssi_val: ', best_rssi_val)
        SF_time = 20 #in msec for 60KHz carrier frequency in 5G
        rate = self.mimo_models[0].Calc_Rate(SF_time, self.K, best_RSSI_val=best_rssi_val)
        print("[RF_Env] Rate: {0}".format(rate))

        self.count += 1
        rwd = self.Reward(rate)
        self.obs = RIM
        self.cur_ue = prev_ue[:]
        return self.obs, rwd

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
    def Reward(self, rate):
        if rate >= self.rate_threshold:
            return 1
        else:
            return 0


    '''
    reset()
    - Resets the environment to its default values
    - Prototype of the gym environment class  
    '''

    def reset(self, ue):
        # Note: should be a uniform random value between starting 4-5 SNR states
        self.TB_r = Gen_RandomBeams(1, self.N)[0] # one random TX beam
        #print(self.TB_r)

        omega_rvec = Gen_RandomBeams(self.K, self.N)
        self.obs = self.Compute_RIM(ue, omega_rvec)
        self.cur_ue = ue[:]
        #Gen_Random_Obs() between ue_s and ue_d #self.observation_space.sample()
        #self.state = self.observation_values[observation][0]

        # self.mimo = MIMO(self.ptx,self.Actions['RBS'][0], self.Actions['TBS'][0], 4, 4)
        # self.mimo = MIMO(self.ptx, 0, 180, self.N_tx, self.N_rx, self.xrange, self.xangle)
        self.count = 0
        return self.obs

    '''
    test_reset(Xrange, Xangle, action_val)
    - Reset the environment to a new MIMO model during test phase of RL model

    Parameters:
    xrange - distance between transmitter and receiver radio units
    xangle - angle between transmitter and reciever radio units
    action_val - Randomly generated action tuple (RBS,TBS, RBeamWidth, TBeamWidth) to be applied on RFBeamEnv with new MIMO model 

    '''
    '''
    def test_reset(self, xrange, xangle, action_val, goal_state):

        # New mimo model
        # self.mimo = MIMO(self.ptx, 0, 180, self.N_tx, self.N_rx, xrange, xangle)
        self.set_distance(xrange, xangle, goal_state)
        # self.mimo.X_range = xrange
        # self.mimo.X_angle = xangle
        self.count = 0
        # random action
        # action = self.action_space.sample()
        SNR = self.mimo.Calc_SNR(self.ptx, action_val[0], action_val[1], action_val[2], action_val[3])

        logSNR = int(np.around(10 * np.log10(SNR), decimals=2))  # considering int values of SNR

        if logSNR > self.max_state:
            SNR_state = self.max_state
        elif logSNR < self.min_state:
            SNR_state = self.min_state
        else:
            SNR_state = logSNR

        # print("SNR state calculated: {0}".format(SNR_state))
        state = self.rev_observation_values[SNR_state]

        return state, action_val
    
    def get_Rate(self, stepcount, Tf):
        rateOpt = self.mimo.Calc_RateOpt(stepcount, Tf, self.ptx)
        rate = self.mimo.Calc_Rate(stepcount, Tf)
        return rate, rateOpt

    def render(self, mode='human', close=False):
        pass

    def action_value(self, action):
        return self.action_values[action]

    '''
def Generate_BeamDir(N):
    min_ang = 0
    max_ang = math.pi
    BeamSet = np.zeros(N)

    #B_i = (i)pi/(N-1), forall 0 <= i <= N-1; 0< beta < pi/(N-1)
    for i in range(BeamSet.shape[0]):
        BeamSet[i] = i*(max_ang-min_ang)/(N-1)
    return BeamSet

def Gen_RandomBeams(k, N):

    BeamSet = Generate_BeamDir(N)
    ndx_lst = np.random.randint(0,N, k)
    #print('B', BeamSet[4])
    k_beams = [BeamSet[x] for x in ndx_lst]
    return k_beams

