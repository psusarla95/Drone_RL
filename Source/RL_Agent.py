import numpy as np
from Source.RF_Env import RFBeamEnv, Generate_BeamDir, Gen_RandomBeams
import itertools


class RL_Agent:
    def __init__(self, env, alpha, gamma, ue_loc):
        self.env = env  # Environment
        self.alpha = alpha  # learning rate
        self.gamma = gamma #discount factor
        self.N = env.N
        self.obs = {
            'UE':  ue_loc, #list(np.around(np.pi/180*np.arange(180,360,1), decimals=8)), #Generate_BeamDir(self.N),
            'RBD': Generate_BeamDir(self.N)
        }
        self.actions = {
            'RBS': Generate_BeamDir(self.N)
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
            action = get_Values(self.act_dtypes, action)[0]
        else:
            #print(obs)
            #print(self.obs_map)
            #print(self.inv_obs_map)
            obs_ndx = self.inv_obs_map[ConvObs_ToString(obs)]
            action_ndx = np.argmax(self.Q[obs_ndx, :])
            action = self.act_map[action_ndx]
            #print("enc action2:{0}".format(action))
            action = get_Values(self.act_dtypes, action)[0]
        return action


    def Update_Q(self, prev_obs, action, obs, rwd):
        #print(self.obs_map)
        #print(obs)
        prev_obs_ndx = self.inv_obs_map[ConvObs_ToString(prev_obs)]
        obs_ndx = self.inv_obs_map[ConvObs_ToString(obs)]
        action_ndx = self.inv_act_map[ConvObs_ToString(action)]
        self.Q[prev_obs_ndx, action_ndx] += self.alpha*(rwd + self.gamma*np.max(self.Q[obs_ndx, :])- (1-self.alpha)*self.Q[prev_obs_ndx, action_ndx])
        return True

    def Best_Action(self, obs):
        obs_ndx = self.inv_obs_map[ConvObs_ToString(obs)]
        act_ndx = np.argmax(self.Q[obs_ndx, :])
        action_val = self.Q[obs_ndx, act_ndx]
        action = self.act_map[act_ndx]
        action = get_Values(self.act_dtypes, action)[0]
        return action, action_val

def ConvObs_ToString(obs):
    #str_obs = " "
    #obs_list = [str(x) for x in obs]
    #str_obs = str_obs.join(obs_list)
    str_obs = str(tuple(x.tostring() for x in obs))
    #print(str_obs)
    return str_obs
'''
def ConvAct_ToString(obs):
    #str_obs = " "
    #obs_list = [str(x) for x in obs]
    #str_obs = str_obs.join(obs_list)
    str_obs = str(tuple(x.tostring() for x in obs))
    #print(str_obs)
    return str_obs
'''
'''
    def Best_Action(self, obs):
        str_obs = ConvObs_ToString(obs)
        assert self.Check_Obs(obs), "%r (%s) obs is invalid" % (obs, type(obs))

        best_a = max(self.Q[str_obs], key=self.Q[str_obs].get)
        best_val = self.Q[str_obs][best_a]

        return ConStr_Action(best_a), best_val



def ConvAct_String(action):
    str_l = str(action)
    return str_l

def ConStr_Action(str_action):
    return eval(str_action)
'''
def Custom_Space_Mapping(actions):

    parameter_count = len(actions.keys())
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
    '''
    obs = action_val_tuples[0]
    print(obs)

    for i in range(len(dtype_list)):
        print(obs[i])
        print(np.frombuffer(obs[i],dtype=dtype_list[i]))
    #action_val_tuples = [[x[0],x[1]] for x in list(itertools.product())
    '''
    action_key_list = list(np.arange(len(action_val_tuples)))

    action_values = dict(zip(action_key_list,action_val_tuples))
    #print("action_values: {0}".format(action_values))

    return dtype_list, action_values

def get_Values(dtypes, input):
    output = []
    for i in range(len(dtypes)):
        #print(np.frombuffer(input[i], dtype=dtypes[i]))
        output.append(np.frombuffer(input[i], dtype=dtypes[i]))

    return np.array(output)