import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from Source.MIMO import MIMO
#from Source.Misc import *
from Source.miscfun.geometry import *

# This is the 3D plotting toolkit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        self.ue_step = 50
        self.ue_xloc = np.arange(-500, 550, 50)  #10 locs
        #self.ue_xloc = np.delete(self.ue_xloc, np.argwhere(self.ue_xloc == 0)) #remove (0,0) from ue_xloc
        self.ue_yloc = np.arange(50,550, 50)     #5 locs
        #self.ue_yloc = np.delete(self.ue_yloc, np.argwhere(self.ue_yloc == 0))  # remove (0,0) from ue_xloc
        self.ue_vx = np.array([50,100]) #3 speed parameters
        self.ue_vy = np.array([50,100]) #3 speed parameters
        self.ue_xdest = np.array([np.min(self.ue_xloc)]) # 1 x-dest loc np.min(self.ue_xloc)
        self.ue_ydest = np.array([np.min(self.ue_yloc)]) # 1 y-dest loc
        self.ue_xsrc = np.array([np.max(self.ue_xloc)]) # 1 source x-loc
        self.ue_ysrc = np.array([np.max(self.ue_yloc)]) # 1 source y-loc
        self.ue_moves = np.array(['L', 'R', 'U', 'D'])  # moving direction of UAV

        self.seed()
        #low_obs = np.array([-500, 0, 0.0, 10.0, 10.0])
        self.high_obs = np.array([np.max(self.ue_xloc), np.max(self.ue_yloc)])
        self.obs_space = spaces.MultiDiscrete([len(self.ue_xloc), #ue_xloc
                                               len(self.ue_yloc), #ue_yloc
                                             ])

        self.act_space = spaces.Discrete(self.N*len(self.ue_moves)) #n(RBD)*n(ue_xvel)*n(ue_yvel)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.act_space.contains(action), "%r (%s) invalid" % (action, type(action))

        state = np.rint(self.state * self.high_obs)

        rbd_ndx, ue_mv_ndx = self.decode_action(action)
        ue_vx, ue_vy = self.choose_vel(ue_mv_ndx)
        rbs = self.BeamSet[rbd_ndx]
        ue_xdest = self.ue_xdest[0]
        ue_ydest = self.ue_ydest[0]

        ue_xloc, ue_yloc = state

        ue_mv = self.ue_moves[ue_mv_ndx]
        if ue_mv == 'L':
            new_ue_xloc = max(ue_xloc + ue_vx, np.min(self.ue_xloc))
            new_ue_yloc = ue_yloc + ue_vy
        if ue_mv == 'U':
            new_ue_xloc = ue_xloc + ue_vx
            new_ue_yloc = min(ue_yloc + ue_vy, np.max(self.ue_yloc))
        if ue_mv == 'R':
            new_ue_xloc = min(ue_xloc + ue_vx, np.max(self.ue_xloc))
            new_ue_yloc = ue_yloc + ue_vy
        if ue_mv == 'D':
            new_ue_xloc = ue_xloc + ue_vx
            new_ue_yloc = max(ue_yloc + ue_vy, np.min(self.ue_yloc))

        cur_ue_pos = np.array([ue_xloc, ue_yloc, 0])
        self.mimo_model = MIMO(cur_ue_pos, self.gNB[0], self.sc_xyz, self.ch_model, self.ptx, self.N_tx, self.N_rx)
        rwd, done = self._gameover(rbs, self.mimo_model.az_aod)

        self.SNR, self.rate = self.mimo_model.Calc_Rate(self.SF_time, np.array([rbs, 0]))  # rkbeam_vec, tbeam_vec )

        self.cur_rate = self.rate
        self.cur_dist = np.sqrt((ue_xloc-ue_xdest)**2 + (ue_yloc-ue_ydest)**2) #x**2 + y**2
        self.state = np.array([new_ue_xloc, new_ue_yloc]) / self.high_obs


        #self.rate = 1e3*self.rate

        new_ue_xndx = np.where(self.ue_xloc ==new_ue_xloc)[0][0]
        new_ue_yndx = np.where(self.ue_yloc == new_ue_yloc)[0][0]
        self.ue_path_rates.append(self.rate)
        #self.ue_path_rates.append(self.rate)
        self.ue_path.append(np.array([new_ue_xloc, new_ue_yloc]))

        self.steps_done += 1

        #rwd = self._reward(prev_dist)
        #print("[uav_env] rwd: {}".format(rwd))


        return self.state, rwd, done, {}

    def reset(self, rate_thr):
        # Note: should be a uniform random value between starting 4-5 SNR states
        #self.TB_r = get_TBD(ue, self.alpha)#Gen_RandomBeams(1, self.N)[0]  # one random TX beam
        #state_indices = self.obs_space.sample()
        xloc_ndx, yloc_ndx = self.obs_space.sample()

        #Start from a fixed start location
        self.state = np.array([self.ue_xloc[xloc_ndx],
                               self.ue_yloc[yloc_ndx]
                               ])
        #self.state = np.array([self.ue_xsrc[0],
        #                       self.ue_ysrc[0]
        #                       ])

        self.steps_done = 0
        self.rate = 0.0
        self.cur_dist = np.Inf
        self.cur_rate = 0.0
        self.ue_path = []
        self.ue_path.append(self.state)
        self.ue_xsrc = self.state[0]
        self.ue_ysrc = self.state[1]
        self.ue_path_rates = []
        #self.ue_path_rates = []
        #Computing the rate threshold for the given destination
        #ue_dest = np.array([self.ue_xloc[xloc_ndx], self.ue_yloc[yloc_ndx], 0])
        #dest_mimo_model = MIMO(ue_dest, self.gNB[0], self.sc_xyz, self.ch_model, self.ptx, self.N_tx, self.N_rx)
        #dest_SNR = []
        #dest_rates = []
        #for rbeam in self.BeamSet:  # rbeam_vec:
        #    SNR, rate = dest_mimo_model.Calc_Rate(self.SF_time, np.array([rbeam, 0]))
        #    dest_SNR.append(SNR)
        #    dest_rates.append(rate)


        self.rate_threshold = rate_thr #np.max(dest_rates)

        self.state = self.state / self.high_obs
        #self.state = self.state.reshape((1, len(self.state)))
        return self.state

    def render(self, mode='human', close=False):
        #fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        #x_axis = [x[0] for x in self.ue_path]
        #y_axis = [x[1] for x in self.ue_path]
        #z_axis = self.ue_path_rates
        #plt.plot(x_axis, y_axis)

        #plt.show()

        from matplotlib.path import Path
        import matplotlib.patches as patches

        verts = [(int(x[0]),int(x[1])) for x in self.ue_path]
        #print(self.ue_path)
        #print(verts)

        codes = [Path.LINETO for x in range(len(verts))]
        codes[0] = Path.MOVETO
        codes[-1] = Path.STOP

        path = Path(verts, codes)

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        patch = patches.PathPatch(path, facecolor='none', lw=2)
        ax.add_patch(patch)

        xs, ys = zip(*verts)
        ax.plot(xs, ys, 'x--', lw=2, color='black')

        #xdisplay, ydisplay = ax.transData.transform_point((self.ue_xsrc, self.ue_ysrc))

        bbox = dict(boxstyle="round", fc="0.8")
        arrowprops = dict(
            arrowstyle="->",
            connectionstyle="angle,angleA=0,angleB=90,rad=10")

        offset = 40
        ax.annotate('Src = (%d, %d)' % (self.ue_xsrc, self.ue_ysrc),
                    (self.ue_xsrc, self.ue_ysrc), xytext=(-2 * offset, offset), textcoords='offset points',
                    bbox=bbox, arrowprops=arrowprops)

        ax.annotate('Dest = (%d, %d)' % (self.ue_xdest[0], self.ue_ydest[0]),
                           (self.ue_xdest[0], self.ue_ydest[0]), xytext=(0.5 * offset, -offset),
                           textcoords='offset points',
                           bbox=bbox, arrowprops=arrowprops)


        offset= 10
        bbox =dict(boxstyle="round", facecolor='yellow', edgecolor='none')
        for i in range(0,len(self.ue_path_rates)):
            ax.annotate('%.2f' % np.around(self.ue_path_rates[i], decimals=2),
                        (verts[i][0], verts[i][1]), xytext=(-2 * offset, offset), textcoords='offset points',
                        bbox=bbox, arrowprops=arrowprops)

        ax.grid()
        ax.set_xticks(self.ue_xloc)
        ax.set_yticks(self.ue_yloc)
        ax.set_title("UAV graph w.r.t gNB [0,0,0]")
        ax.set_xlabel("X direction")
        ax.set_ylabel("Y direction")

        plt.show()

        return

    #Not using this function
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

        ue_dist = np.sqrt((self.state[0]-self.ue_xdest) ** 2 + (self.state[1]-self.ue_ydest) ** 2)
        #ue_dest_dist = np.sqrt(self.state[0][-2]**2 + self.state[0][-1]**2)

        if (self.rate >= self.rate_threshold) and (ue_dist <= prev_dist):
            return 10*self.rate + 3
        else:
            return 0.0#10*self.rate - 3

    def _gameover(self, aoa, aod): #prev_dist, curr_rate):
        #ue_dist = np.sqrt(self.state[0][0]**2 + self.state[0][1]**2)
        #ue_dest_dist = np.sqrt(self.state[0][-2]**2 + self.state[0][-1]**2)
        #return ue_dist >= ue_dest_dist
        state = np.rint(self.state * self.high_obs)
        next_dist = np.sqrt((state[0] - self.ue_xdest[0]) ** 2 + (state[1] - self.ue_ydest[0]) ** 2)
        ang_1 = 3.14 - np.around(np.pi/self.N,decimals=2)
        ang_2 = 3.14 + np.around(np.pi/self.N,decimals=2)
        ang_3 = 0#2*3.14 - np.around(np.pi/self.N,decimals=2)
        ang_4 = np.around(np.pi/self.N,decimals=2)#2*3.14

        if (next_dist < 50) and (ang_1 < np.around(aod-aoa, decimals=2) < ang_2):
        #if (next_dist < 50) and (self.rate >= self.rate_threshold):
            rwd = 2.0#3.1#2.1#self.rate + 2.0#2000.0
            done = True
        elif (next_dist < 50) and (ang_3 < np.around(aod-aoa, decimals=2) < ang_4):
            rwd = 2.0#3.1#2.1#self.rate + 2.0#2000.0
            done = True
        elif (ang_1 < np.around(aod-aoa, decimals=2) < ang_2): #(ang_1 < np.around(aod-aoa, decimals=2) < ang_2) and
        #elif (self.rate >= self.rate_threshold):
            rwd = 2.0*np.exp(-1*(self.steps_done-1)/20)#1.0#self.rate+1.0#self.rate + 2.0 #10*np.log10(val+1) + 2.0
            done = False
        elif (ang_3 < np.around(aod-aoa, decimals=2) < ang_4): #(self.rate >= self.rate_threshold) and
            rwd = 2.0 * np.exp(-1 * (self.steps_done - 1) / 10)  # 1.0#self.rate+1.0#self.rate + 2.0 #10*np.log10(val+1) + 2.0
            done = False
        else:
            rwd = -1.0#-self.rate-1.0#-self.rate -2.0#-20.0
            done = False
        self.aoa = aoa
        self.aod = aod
        return rwd, done

    def decode_action(self, action_ndx):
        #ue_vy_ndx = action_ndx % len(self.ue_vy)
        #action_ndx = action_ndx // len(self.ue_vy)
        #ue_v_ndx = action_ndx % len(self.ue_vx)
        #action_ndx = action_ndx // len(self.ue_vx)

        ue_mv_ndx = action_ndx % len(self.ue_moves)
        action_ndx = action_ndx // len(self.ue_moves)

        beam_ndx = action_ndx % self.N
        action_ndx = action_ndx // self.N

        assert 0<= action_ndx <= self.act_space.n
        return (beam_ndx, ue_mv_ndx)

    #Not using this function
    def encode_action(self, beam_ndx, ue_vx_ndx, ue_vy_ndx):
        i = beam_ndx
        i*= self.N

        i += ue_vx_ndx
        i*=len(self.ue_vx)

        i += ue_vy_ndx
        i*= len(self.ue_vy)

        return i

    def choose_vel(self, ue_mv_ndx):
        ue_mv = self.ue_moves[ue_mv_ndx]

        if ue_mv == 'L': #move left
            ue_vx = -1 * self.ue_vx[0]
            ue_vy = 0
        elif ue_mv == 'U': #move up
            ue_vx = 0
            ue_vy = self.ue_vy[0]
        elif ue_mv == 'D': #move down
            ue_vx = 0
            ue_vy = -1*self.ue_vy[0]
        else: #move right
            ue_vx = self.ue_vx[0]
            ue_vy = 0

        return ue_vx, ue_vy

    def get_Los_Rate(self, state):

        state = np.rint(state * self.high_obs)
        ue_xloc, ue_yloc = state

        sc_xyz = np.array([])
        ch_model = 'fsp'
        ue_pos = np.array([ue_xloc, ue_yloc, 0])

        mimo_model = MIMO(ue_pos, self.gNB[0], sc_xyz, ch_model, self.ptx, self.N_tx, self.N_rx)
        SNR, rate = mimo_model.Los_Rate()  # rkbeam_vec, tbeam_vec )
        #rate = 1e3 * rate
        return SNR, rate

    def get_Exh_Rate(self, state):
        state = np.rint(state * self.high_obs)
        ue_xloc, ue_yloc = state
        ue_pos = np.array([ue_xloc, ue_yloc,0])

        mimo_exh_model = MIMO(ue_pos, self.gNB[0], self.sc_xyz, self.ch_model, self.ptx, self.N_tx, self.N_rx)
        #rbeam_vec = self.BeamSet#Generate_BeamDir(self.N)
        exh_SNR = []
        exh_rates = []

        for rbeam in self.BeamSet:#rbeam_vec:
            SNR, rate = mimo_exh_model.Calc_Rate(self.SF_time, np.array([rbeam, 0]))
            #rate = 1e3 * rate
            exh_SNR.append(SNR)
            exh_rates.append(rate)

        best_rbeam_ndx = np.argmax(exh_rates)
        #print("[UAV_Env]: AOD: {}, AoA: {}, AoD-AoA: {}".format(mimo_exh_model.channel.az_aod[0], self.BeamSet[best_rbeam_ndx], -self.BeamSet[best_rbeam_ndx]+mimo_exh_model.channel.az_aod[0]))
        return self.BeamSet[best_rbeam_ndx], np.max(exh_rates) #(Best RBS, Best Rate)

    def get_Rate(self):
        return self.rate




def Generate_BeamDir(N):
    #if np.min(self.ue_xloc) < 0 and np.max(self.ue_xloc) > 0:

    min_ang = 0#-math.pi/2
    max_ang = np.pi#math.pi/2
    step_size = (max_ang-min_ang)/N
    beam_angles = np.arange(min_ang+step_size, max_ang+step_size, step_size)

    BeamSet = []#np.zeros(N)#np.fft.fft(np.eye(N))

    #B_i = (i)pi/(N-1), forall 0 <= i <= N-1; 0< beta < pi/(N-1)
    val = min_ang
    for i in range(N):
        BeamSet.append(np.arctan2(np.sin(beam_angles[i]), np.cos(beam_angles[i])))#(i+1)*(max_ang-min_ang)/(N)

    return np.array(BeamSet) #eval(strBeamSet_list)#np.ndarray.tolist(BeamSet)


'''
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
'''