import numpy as np
from Source.RF_Env_nopos import RFBeamEnv, Generate_BeamDir, Gen_RandomBeams
import itertools


'''
####################################
    Agent CLASS
####################################

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

class RL_Agent:
    def __init__(self, ag_info):
        self.env = ag_info['env']  # Environment
        self.alpha = ag_info['alpha']  # learning rate
        self.gamma = ag_info['gamma'] #discount factor
        self.N_tx = self.env.N_tx
        self.N_rx = self.env.N_rx
        self.obs = {
            'UE':  ag_info['ue_loc'], #list(np.around(np.pi/180*np.arange(180,360,1), decimals=8)), #Generate_BeamDir(self.N),
            'RBD': Generate_BeamDir(self.N_rx)
        }
        self.actions = {
            'RBS': Generate_BeamDir(self.N_rx)
        }
        self.obs_dtypes, self.obs_map = Custom_Space_Mapping(self.obs)
        self.act_dtypes, self.act_map = Custom_Space_Mapping(self.actions)

        self.inv_obs_map = dict((str(v), k) for k, v in self.obs_map.items())
        self.inv_act_map = dict((str(v), k) for k, v in self.act_map.items())
        #print(self.obs_map)
        #print(self.inv_obs_map)
        #print("obs_map: {0}".format(self.obs_map))
        #print("act_map: {0}".format(self.act_map))
        #print("Inv obs_map: {0}".format(self.inv_obs_map))
        #print("Inv act_map: {0}".format(self.inv_act_map))
        self.obs_space = self.obs_map.values()
        self.act_space = self.act_map.values()

        self.Q = np.zeros((len(self.obs_space), len(self.act_space)))  # Q_table

    def sample_action(self, obs, eps):

        if (np.random.random() < eps):
            #print("came here")
            rnd_ndx = np.random.randint(0, self.N)
            action = self.act_map[rnd_ndx]
            #print("enc action1:{0}".format(action))
            action = get_Values(self.act_dtypes, action)
        else:
            #print(obs)
            #print(self.obs_map)
            #print(self.inv_obs_map)
            obs_ndx = self.inv_obs_map[ConvObs_ToString(obs)]
            action_ndx = np.argmax(self.Q[obs_ndx, :])
            action = self.act_map[action_ndx]
            #print("enc action2:{0}".format(action))
            action = get_Values(self.act_dtypes, action)
        return action


    def Update_Q(self, prev_obs, action, obs, rwd):
        #print(self.obs_map)
        #print(obs)
        prev_obs_ndx = self.inv_obs_map[ConvObs_ToString(prev_obs)]
        obs_ndx = self.inv_obs_map[ConvObs_ToString(obs)]
        action_ndx = self.inv_act_map[ConvObs_ToString(action)]
        self.Q[prev_obs_ndx, action_ndx] += self.alpha*(rwd + self.gamma*np.max(self.Q[obs_ndx, :])- self.Q[prev_obs_ndx, action_ndx])
        return True

    def Best_Action(self, obs):
        obs_ndx = self.inv_obs_map[ConvObs_ToString(obs)]
        act_ndx = np.argmax(self.Q[obs_ndx, :])
        action_val = self.Q[obs_ndx, act_ndx]
        action = self.act_map[act_ndx]
        action = get_Values(self.act_dtypes, action)
        return action, action_val

def ConvObs_ToString(obs):
    #str_obs = " "
    #obs_list = [str(x) for x in obs]
    #str_obs = str_obs.join(obs_list)
    str_obs = str(tuple(x.tostring() for x in obs))
    #print(str_obs)
    return str_obs

def Custom_Space_Mapping(actions):

    parameter_list = []
    dtype_list =[]
    for key in actions.keys():
        val_list = actions[key]#[actions.keys[i]]
        dtype_list.append(val_list[0].dtype.name)
        #print(val_list)
        parameter_list.append([val.tostring() for val in val_list])#list(range(par_range[0],par_range[1]+1,par_range[2])))

    #print(parameter_list)
    #creates a list of all possible tuples from given lists of action values
    #action_val_tuples = [list(x) for x in np.array(np.meshgrid(*parameter_list)).T.reshape(-1,len(parameter_list))]
    action_val_tuples = list(itertools.product(*parameter_list))
    #print(c)
    #print(y)

    action_key_list = list(np.arange(len(action_val_tuples)))

    action_values = dict(zip(action_key_list,action_val_tuples))
    #print("action_values: {0}".format(action_values))

    return dtype_list, action_values

def get_Values(dtypes, input):
    output = []
    for i in range(len(dtypes)):
        #print(np.frombuffer(input[i], dtype=dtypes[i]))
        output.append(np.frombuffer(input[i], dtype=dtypes[i]))[0]

    return np.array(output)