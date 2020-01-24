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
This version of the UAV enivronment is for NLOS channelling conditions

Ground Base Station (gNB) Characteristics:
- Considers a MIMO model with mmwave frequency
- Considers a fixed Ptx, Ntx, NRx
- Chooses
  UAV moves - U - Up,L - Left, R- Right, D- Down 
  Beam Directions - Receiver Beam Direction (RBD)
  UAV location - UAV_xloc, UAV_yloc, UAV_zloc
  Data Rate, Signl-to-Noise-Ratio - rate, SNR
  as the main parameters for this RF Beam model 

- Observation space - [(-500,-500),(-500,-450).....(uav_xloc,uav_yloc)....(450,500),(500,500)] 
- Action space - [0,1,2,.......NRx*4] -> [(pi/NRx,'U'),(pi/NRx,'L'), (pi/NRx, 'R'), 
                 (pi/NRx, 'D')......(RBD,UAV_move).......(7pi/NRx,'D')]

- Transmit Power= 30 dB, N_tx= 1, N_rx=8


UAV characteristics:
- moves in a 2D (x,y) grid (z direction is assumed to be '0')
- Transmit Power is fixed Ptx = 30 dB
- Transmits a single Beam in LoS direction
- moves with constant speed v_x, v_y in x and y direction. v_x = 50ms-1, v_y=50ms-1


Channel characteristics:
- predefined through the env definition (self.channel)
- Free Space Path (fsp) Loss modelling with no scattering is assumed
- One gNB and one UE drone scenario 
'''


class UAV_Env_v5(gym.Env):
    """
    Description:
    A UAV moves in a region within the coverage area of the base station.
    The objective of the problem is to guide UAV (using gNB)in a rate requirement path,
    reaching the destination in an energy minimized way as early as possible

    Observation:
        Type: MultiDiscrete(2,)
        Num Observation     Min     Max     Step
        1   UAV_xloc       -500.0   500.0   50.0
        2   UAV_yloc       -500.0   500.0   50.0

    Action:
        Type:Discrete(Nrx*num(uav_moves))
        Num                   Action
        0                   Bdir0, mov0
        1                   Bdir0, mov1
        2                   Bdir0, mov2
        3                   Bdir0, mov3
        4                   Bdir 1, ...
        ...                     ....
        (Nrx-1)*uav_moves   Bdir{Nrx-1}, mov3

    Reward:
        Reward is value computed based on rate measurements and energy minimization conditions. Range [-1.0, 1.0]

    Starting State:
        Obs_space.sample() - Any random location with the Observation range

    Episode Termination:
        When UAV reaches the defined destination D
    """

    def __init__(self):

        #Antenna Modelling
        #Uniform Linear Arrays (ULA) antenna modelling is considered
        self.N_tx = 8 # Num of transmitter antenna elements
        self.N_rx = 8  # Num of receiver antenna elements
        self.count = 0
        self.ptx = 30  #dB
        self.SF_time = 20 #msec - for 60KHz carrier frequency in 5G
        self.alpha = 0


        #Base Statin Locations
        self.gNB = np.array([[0,0,0]])#, [20,30,0], [40,60,0]]

        #Channel
        self.sc_xyz= np.array([[0,150,0],[250,50,0],[-200,-150,0]])
        self.ch_model= 'uma-nlos'
        self.N = self.N_rx #Overall beam directions
        self.BeamSet = Generate_BeamDir(self.N) #Set of all beam directions

        #Observation - RSSI information of states
        self.state = None
        self.rate = None
        self.rate_threshold = None  # good enough QoS val (Rate)

        #UE information
        self.ue_step = 50
        self.ue_xloc = np.arange(-500, 550, self.ue_step)  #10 locs
        #self.ue_xloc = np.delete(self.ue_xloc, np.argwhere(self.ue_xloc == 0)) #remove (0,0) from ue_xloc
        self.ue_yloc = np.arange(-500,550, self.ue_step)     #5 locs
        #self.ue_yloc = np.delete(self.ue_yloc, np.argwhere(self.ue_yloc == 0))  # remove (0,0) from ue_xloc
        self.ue_vx = np.array([50,100]) #3 speed parameters
        self.ue_vy = np.array([50,100]) #3 speed parameters
        self.ue_xdest = np.array([np.min(self.ue_xloc)]) # 1 x-dest loc np.min(self.ue_xloc)
        self.ue_ydest = np.array([np.min(self.ue_yloc)]) # 1 y-dest loc
        self.ue_xsrc = np.array([np.max(self.ue_xloc)]) # 1 source x-loc
        self.ue_ysrc = np.array([np.max(self.ue_yloc)]) # 1 source y-loc
        self.ue_moves = np.array(['L', 'R', 'U', 'D'])  # moving direction of UAV

        self.seed()

        #Observation and Action Spaces

        #low_obs = np.array([-500, 0, 0.0, 10.0, 10.0])
        self.high_obs = np.array([np.max(self.ue_xloc), np.max(self.ue_yloc)])
        self.obs_space = spaces.MultiDiscrete([len(self.ue_xloc), #ue_xloc
                                               len(self.ue_yloc), #ue_yloc
                                             ])

        self.act_space = spaces.Discrete(self.N*len(self.ue_moves)) #n(RBD)*n(ue_moves)


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

        self.cur_dist = np.sqrt((ue_xloc - ue_xdest) ** 2 + (ue_yloc - ue_ydest) ** 2)  # x**2 + y**2
        self.cur_state = state

        if self.done: #reached terminal state
            return self.state, self.rwd, self.done, {}

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

        new_ue_pos = np.array([new_ue_xloc, new_ue_yloc, 0])

        #Approximating (0,0) to (20,20) location to prevent rate->Inf
        if(new_ue_xloc == 0) and (new_ue_yloc == 0):
            self.mimo_model = MIMO(np.array([40, 40, 0]), self.gNB[0], self.sc_xyz, self.ch_model, self.ptx, self.N_tx, self.N_rx)
            self.SNR, self.rate = self.mimo_model.Calc_Rate(self.SF_time, np.array([rbs, 0]))  # rkbeam_vec, tbeam_vec )

        else:
            self.mimo_model = MIMO(new_ue_pos, self.gNB[0], self.sc_xyz, self.ch_model, self.ptx, self.N_tx, self.N_rx)
            self.SNR, self.rate = self.mimo_model.Calc_Rate(self.SF_time, np.array([rbs, 0]))  # rkbeam_vec, tbeam_vec )

        if self.measure == 'rate_thr_path':
            self.rwd, self.done = self._gameover()
        elif self.measure == 'rate_path':
            self.rwd, self.done = self.rate_path_gameover()
        elif self.measure == 'short_path':
            self.rwd, self.done = self.short_path_gameover()
        else:
            print("Err: Incorrect measure str\n")
            self.rwd, self.done = -100.0, True

        self.ue_path_rates.append(self.rate)
        self.ue_path.append(np.array([new_ue_xloc, new_ue_yloc]))

        self.cur_rate = self.rate
        self.prev_dist = self.cur_dist
        self.state = np.array([new_ue_xloc, new_ue_yloc]) / self.high_obs


        self.steps_done += 1

        return self.state, self.rwd, self.done, {}

    def beyond_border(self,ue_xpos, ue_ypos):
        if (ue_xpos == np.min(self.ue_xloc)) or (ue_xpos == np.max(self.ue_xloc)) or (ue_ypos == np.min(self.ue_yloc)) or (ue_ypos == np.max(self.ue_yloc)):
            return True
        else:
            return False

    def reset(self, rate_thr, meas, state_indices):

        #state_indices = self.obs_space.sample()
        #xloc_ndx, yloc_ndx = self.obs_space.sample()
        xloc_ndx, yloc_ndx = state_indices

        #Start from a random start location
        self.state = np.array([self.ue_xloc[xloc_ndx],
                               self.ue_yloc[yloc_ndx]
                               ])
        #self.state = np.array([self.ue_xsrc[0],
        #                       self.ue_ysrc[0]
        #                       ])

        self.steps_done = 0
        self.rate = 0.0

        self.ue_path = []
        #self.ue_path.append(self.state)
        self.ue_xsrc = np.array([self.state[0]])
        self.ue_ysrc = np.array([self.state[1]])
        self.ue_path_rates = []
        self.measure = meas
        self.rwd = 0.0
        self.done = False
        #self.ue_path_rates = []

        self.rate_threshold = rate_thr #np.max(dest_rates)

        self.state = self.state / self.high_obs
        self.prev_dist = np.Inf

        #_, self.cur_rate = self.get_Exh_Rate(self.state)
        self.cur_rate = 0.0
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

    def dest_check(self):
        reached = False
        #state = self.cur_state#np.rint(self.cur_state * self.high_obs)
        #curr_dist = np.sqrt((state[0] - self.ue_xdest[0]) ** 2 + (state[1] - self.ue_ydest[0]) ** 2)

        if self.cur_dist < self.ue_step:
            reached = True
        return reached

    def _gameover(self):

        #print(self.cur_dist, self.ue_xdest[0], self.ue_ydest[0])
        #print(np.sqrt((self.ue_xsrc[0] - self.ue_xdest[0]) ** 2 + (self.ue_ysrc[0] - self.ue_ydest[0]) ** 2))
        norm_dist = (self.cur_dist+0.01)/(np.sqrt((self.ue_xsrc[0] - self.ue_xdest[0]) ** 2 + (self.ue_ysrc[0] - self.ue_ydest[0]) ** 2)+0.01)
        data_rate = self.cur_rate
        rwd = np.log10((data_rate+1e-4)/norm_dist)
        #print(rwd, norm_dist, self.cur_dist, data_rate)
        done = False
        if(self.dest_check()):
            done = True
        return rwd, done
    '''
    def _gameover(self):
        #state = np.rint(self.state * self.high_obs)
        #curr_dist = np.sqrt((state[0] - self.ue_xdest[0]) ** 2 + (state[1] - self.ue_ydest[0]) ** 2)

        #if (self.dest_check()) and (self.rate >= self.rate_threshold):
        #    rwd = 1.0#3.1#2.1#self.rate + 2.0#2000.0
        #    done = True

        if self.dest_check():
            rwd = 1.0*np.log10(8*self.rate + 1)#*np.log10(self.rate + 1) #*np.exp(-self.steps_done/100)
            done = True
        elif (self.cur_rate >= self.rate_threshold): #and (self.cur_dist < self.prev_dist):

            #rwd = 1.0*np.exp(-1*(self.steps_done-1)/50)*np.log10(max(21-self.rate,0)+1)#*np.exp(self.rate/10)/20#np.log10(max(21.5-self.rate, 0)+1)/3#*np.log10(self.rate+1)# np.exp(self.rate/50)#1.0#self.rate+1.0#self.rate + 2.0 #10*np.log10(val+1) + 2.0
            rwd = 0.6*np.exp(-self.cur_dist/1000)*np.exp(-2*(self.steps_done-1)/50)*np.log10(8*self.rate + 1)#*np.exp(-2*(self.steps_done-1)/50)#*np.log10(self.rate + 1)#*np.exp(self.rate/20)#*min(np.exp(self.rate/20), np.exp((self.rate_threshold-self.rate)/20.0))#0.5 * np.exp(-1 * (self.steps_done - 1) / 50) *(1-self.rate/30)
            #print(rwd)
            done = False
        elif (self.cur_dist < self.prev_dist):
            rwd = 0.8*np.exp(-self.cur_dist/1000)*np.exp(-2*(self.steps_done-1)/50)*np.log10(8*self.rate + 1)#*np.exp(-self.rate/20)#-self.rate-1.0#-self.rate -2.0#-20.0
            done = False

        elif (self.cur_dist > self.prev_dist): #self.cur_rate >= self.rate_threshold) and
            rwd = 0.2*np.exp(-self.cur_dist/1000)*np.exp(-2*(self.steps_done-1)/50)*np.log10(8*self.rate + 1)#*np.log10(self.rate + 1)#*np.exp(self.rate/20)#*min(np.exp(self.rate/20), np.exp((self.rate_threshold-self.rate)/20.0))
            done = False

        #elif (self.beyond_border(self.cur_state[0], self.cur_state[1])):
        #    rwd = 0.3*np.exp(-curr_dist/1000)*np.exp(-2*(self.steps_done-1)/50)*np.log10(8*self.rate + 1)
        #    done = False
        #elif (self.cur_dist >= self.prev_dist):
        #    rwd = 0.2*np.exp(-self.cur_dist/1000)*np.exp(-2*(self.steps_done-1)/50)*np.log10(8*self.cur_rate + 1)#*np.exp(-self.rate/20)#-self.rate-1.0#-self.rate -2.0#-20.0
        #    done = False
        else:
            rwd = 0.0#0.2*np.exp(-self.cur_dist/1000)*np.exp(-2*(self.steps_done-1)/50)*np.log10(8*self.rate + 1)
            done = False

        #done = False
        #if self.dest_check():
        #    done = True
        #rwd = np.log10(8*self.rate + 1) + np.exp(-self.cur_dist/1000)*np.exp(-2*(self.steps_done-1)/50)
        return rwd, done
    '''

    #Reward Function for max rate path
    def rate_path_gameover(self):
        state = np.rint(self.state * self.high_obs)
        next_dist = np.sqrt((state[0] - self.ue_xdest[0]) ** 2 + (state[1] - self.ue_ydest[0]) ** 2)

        if (self.dest_check()):
            rwd = 1.0  # 3.1#2.1#self.rate + 2.0#2000.0
            done = True

        elif (next_dist < self.cur_dist):

            # rwd = 1.0*np.exp(-1*(self.steps_done-1)/50)*np.log10(max(21-self.rate,0)+1)#*np.exp(self.rate/10)/20#np.log10(max(21.5-self.rate, 0)+1)/3#*np.log10(self.rate+1)# np.exp(self.rate/50)#1.0#self.rate+1.0#self.rate + 2.0 #10*np.log10(val+1) + 2.0
            rwd = 0.5 * np.exp(-1 * (self.steps_done - 1) / 50) * np.exp(self.rate / 20)  # *min(np.exp(self.rate/20), np.exp((self.rate_threshold-self.rate)/20.0))#0.5 * np.exp(-1 * (self.steps_done - 1) / 50) *(1-self.rate/30)
            # print(rwd)
            done = False

        elif (next_dist > self.cur_dist):
            rwd = 0.2 * np.exp(-1 * (self.steps_done - 1) / 50) * np.exp(self.rate / 20)  # *min(np.exp(self.rate/20), np.exp((self.rate_threshold-self.rate)/20.0))
            done = False
        else:
            rwd = -1.0 * np.exp(-1 * (self.steps_done - 1) / 50) * np.exp(-self.rate / 20)  # -self.rate-1.0#-self.rate -2.0#-20.0
            done = False

        return rwd, done

    #Reward function for shortest path
    def short_path_gameover(self):
        state = np.rint(self.state * self.high_obs)
        next_dist = np.sqrt((state[0] - self.ue_xdest[0]) ** 2 + (state[1] - self.ue_ydest[0]) ** 2)

        if (self.dest_check()):
            rwd = 1.0  # 3.1#2.1#self.rate + 2.0#2000.0
            done = True
        elif (next_dist < self.cur_dist):
            rwd = 0.2
            done = False
        else:
            rwd = -1.0
            done = False

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

        if (ue_xloc == 0) and (ue_yloc) == 0:
            ue_pos = np.array([ue_xloc+40, ue_yloc+40, 0])

        mimo_model = MIMO(ue_pos, self.gNB[0], sc_xyz, ch_model, self.ptx, self.N_tx, self.N_rx)
        SNR, rate = mimo_model.Los_Rate()  # rkbeam_vec, tbeam_vec )

        return SNR, rate

    def get_Exh_Rate(self, state):
        state = np.rint(state * self.high_obs)
        ue_xloc, ue_yloc = state
        ue_pos = np.array([ue_xloc, ue_yloc, 0])

        if (ue_xloc == 0) and (ue_yloc) == 0:
            #return -1.0,-1.0
            ue_pos = np.array([ue_xloc + 40, ue_yloc + 40, 0])


        mimo_exh_model = MIMO(ue_pos, self.gNB[0], self.sc_xyz, self.ch_model, self.ptx, self.N_tx, self.N_rx)
        #rbeam_vec = self.BeamSet#Generate_BeamDir(self.N)
        exh_SNR = []
        exh_rates = []

        for rbeam in self.BeamSet:#rbeam_vec:
            SNR, rate = mimo_exh_model.Calc_ExhRate(self.SF_time, np.array([rbeam, 0]))
            #rate = 1e3 * rate
            exh_SNR.append(SNR)
            exh_rates.append(rate)

        best_rbeam_ndx = np.argmax(exh_rates)
        best_beam = self.BeamSet[best_rbeam_ndx]
        SNRmax, rate_max = mimo_exh_model.Calc_ExhRate(self.SF_time, np.array([best_beam, 0]), noise_flag=False)
        #print("[UAV_Env]: AOD: {}, AoA: {}, AoD-AoA: {}".format(mimo_exh_model.channel.az_aod[0], self.BeamSet[best_rbeam_ndx], -self.BeamSet[best_rbeam_ndx]+mimo_exh_model.channel.az_aod[0]))
        return best_beam, rate_max #(Best RBS, Best Rate)

    def get_Rate(self):
        return self.cur_rate




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
def Gen_RandomBeams(k, N):

    BeamSet = Generate_BeamDir(N)
    ndx_lst = np.random.randint(0,N, k)
    #print('B', BeamSet[4])
    k_beams = BeamSet[ndx_lst[0]]#[BeamSet[x] for x in ndx_lst]
    return np.array(k_beams)
'''