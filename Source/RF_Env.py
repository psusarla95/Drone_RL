import numpy as np
from Source.MIMO import MIMO
from Source.Codebook import *
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

    def __init__(self, sc_xyz, ch_model):

        self.N_tx = 1 # Num of transmitter antenna elements
        self.N_rx = 8  # Num of receiver antenna elements
        self.count = 0
        self.ptx = 30  #dB
        self.SF_time = 20 #msec - for 60KHz carrier frequency in 5G
        self.alpha = 0
        #self.level = 1
        #self.state = None

        # (x1,y1,z1) of UE_source location
        self.ue_s = None#[10,15,0]
        self.ue_v = None#10
        #self.ue_tdist = 90
        #self.cur_ue = None

        #gNB locations One Serving gNB_1 node, 2 visible gnB_2,gNB_3 nodes
        self.gNB = [[0,0,0], [20,30,0], [40,60,0]]
        self.sc_xyz= sc_xyz
        self.ch_model= ch_model
        #Action space parameters: |A| = 8C4 x |delta_p| =70 x 5 = 350
        #self.Actions = {
        #    'K': 4,  #Beam_set length
        #    'N': 4,  #Overall beam directions
        #    'delta_p': [0,-1,+1,-2,+2] #position control space along x-direction
        #}

        #self.K = self.Actions['K']  #Beam_set length
        #self.N = self.Actions['N']  #Overall beam directions
        self.N = self.N_rx #Overall beam directions
        #self.delta_p = self.Actions['delta_p'] #position control space along x-direction
        self.BeamSet = Generate_BeamDir(self.N) #Set of all beam directions
        #self.beta =  math.pi/(2*(self.N-1)) # Beamwidth beta, 0 < beta <= (pi / (N - 1))

        #State of the system - UE_t w.r.t gnB_1
        #self.obs_space = [[x,self.ue_s[1],self.ue_s[2]] for x in range(self.ue_s[0],self.ue_s[0]+self.ue_tdist,self.ue_v)]

        #Observation - RSSI information of states
        self.obs = None
        self.rate = None
        self.goal_diff = None#None

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
        self.rate_threshold = 0.7 # good enough QoS val (Rate)

        #self.seed()
        #self.viewer = None

    def GenAction_Sample(self):
        #delta_p = self.Actions['delta_p']
        #rp_ndx = np.random.randint(0, len(delta_p))
        #rdelta_p = delta_p[rp_ndx]
        rk_beams = Gen_RandomBeams(1, self.N)
        #rk_beams.append([rdelta_p])
        #str_rk_beams = str(rk_beams)
        #rk_beams = eval(str_rk_beams)
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
        # if action in self.actions[self.current_state]:
        #assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        #rkbeam_vec, delta_p = action[:self.K], action[self.K][0]
        #rkbeam_vec = action[:]
        #tbeam_vec = [self.TB_r]
        #changing only receiver beam based on the action

        #prev_ue = self.cur_ue
        #self.cur_ue[0] += self.ue_v + delta_p
        #prev_obs = self.obs[:]
        #prev_obs[0] += delta_p
        self.ue_s[0] += self.ue_v #+ delta_p
        self.obs[1] = np.array(action)
        self.obs[0] = np.array(self.ue_s)
        #self.TB_r = get_TBD(self.ue_s, self.alpha)
        #RIM = self.Compute_RIM(prev_obs, kbeam_vec)
        #rssi_gnb1 = RIM[:self.K]
        #best_rssi_val = np.max(rssi_gnb1)
        #print('[RF_Env] UE_S: ', self.ue_s)

        self.mimo_model = MIMO(self.ue_s, self.gNB[0], self.sc_xyz, self.ch_model, self.ptx, self.N_tx, self.N_rx)

        self.SNR, self.rate = self.mimo_model.Calc_Rate(self.SF_time, np.array([self.obs[1][0], 0]))#rkbeam_vec, tbeam_vec )
        #print("[RF_Env] SNR: {0}, rate: {1}".format(20*np.log10(self.SNR), self.rate))

        #self.mimo_los_model = MIMO(self.ue_s, self.gNB[0], self.ptx, self.N_tx, self.N_rx)

        #print("[RF_Env] Los_SNR: {0}, Los_rate: {1}".format(20 * np.log10(self.Los_SNR), self.Los_rate))
        #self.mimo_los_model = MIMO(self.obs, self.gNB[0], self.ptx, self.N_tx, self.N_rx)
        #self.Los_SNR, self.LoS_rate = self.mimo_los_model.Los_Rate(SF_time, self.K)
        #print("[RF_Env] pos_corr: {0}, Rate: {1}".format(delta_p,self.rate))


        self.cum_rate += self.rate

        #self.Los_rate = self.get_LoS_Rate(self.ue_s)
        #self.cum_Los_rate += self.Los_rate

        self.count += 1
        rwd = self.Reward()
        done = self.Game_Over()


        #self.obs[0] += self.ue_v
        #done = self.Game_Over()
        #self.cur_ue = prev_ue
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
        #if rate <= prev_rate:#>= self.rate_threshold:
        #if self.cum_rate >= (self.count*self.rate_threshold):
        #los_rate = self.get_LoS_Rate(self.ue_s)
        #done = self.Game_Over()
        #if done and (self.cum_rate >= self.count*self.rate_threshold):
        #    return 1#self.rate
        #else:
        #    return 0
        #if self.rate >= self.rate_threshold:#done and (self.cum_rate >= self.count*self.rate_threshold):
        #   return 1
        #elif self.rate < self.rate_threshold:
        #    return -1
        #else:
         #   return 0 #-0.5
        #if self.rate >= los_rate:
        #    return los_rate
        #if self.cum_rate >= self.cum_Los_rate:
        #if self.cum_rate >= (self.count*self.rate_threshold):
        return self.rate
        #else:
        #    return 0

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
        #self.TB_r = get_TBD(ue, self.alpha)#Gen_RandomBeams(1, self.N)[0]  # one random TX beam
        self.RB_r = Gen_RandomBeams(1, self.N)  # one random RX beam
        # print(self.TB_r)

        self.obs = [np.array(ue), self.RB_r]
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