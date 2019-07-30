import numpy as np
from Source.RF_Env import RFBeamEnv, Generate_BeamDir, Gen_RandomBeams


class RL_Agent:
    def __init__(self, env, alpha, gamma):
        self.env = env  # Environment
        self.alpha = alpha  # learning rate
        self.gamma = gamma #discount factor
        self.Q = {}  # Q_table

    def sample_action(self, obs, eps):
        assert self.Check_Obs(obs), "%r (%s) obs is invalid" % (obs, type(obs))
        str_obs = ConvObs_ToString(obs)
        if (np.random.random() < eps) or (not self.Q):
            action = self.env.GenAction_Sample()
            self.Q[str_obs][str(action)] = 0.0
        else:

            #print("[RL_A] Q[obs]:{0}".format(self.Q[str_obs]))
            action = max(self.Q[str_obs], key= self.Q[str_obs].get)
            action = eval(action)
        return action

    def Check_Obs(self, obs):
        str_obs = ConvObs_ToString(obs)
        if str_obs in self.Q:
            pass
        else:
            self.Q[str_obs] ={}
            a = self.env.GenAction_Sample()
            self.Q[str_obs][str(a)] = 0.0
        return True

    def Check_Action(self, obs, action):
        str_obs = ConvObs_ToString(obs)
        if str(action) in self.Q[str_obs]:
            pass
        else:
            self.Q[str_obs] = {}
            self.Q[str_obs][str(action)] = 0.0
        return True

    def Action_Vals(self, obs):
        action_vals = []
        str_obs = ConvObs_ToString(obs)
        for a in self.Q[str_obs]:
            action_vals.append(self.Q[str_obs][a])
        #print("[RL_A] action_vals: {0}".format(action_vals))
        return action_vals

    def Update_Q(self, prev_obs, action, obs, rwd):

        assert self.Check_Obs(prev_obs), "%r (%s) prev_obs is invalid" % (prev_obs, type(prev_obs))
        assert self.Check_Obs(obs), "%r (%s) obs is invalid" % (obs, type(obs))
        assert self.Check_Action(prev_obs, action), "%r (%s) action is invalid" % (action, type(action))
        #self.Check_Obs(prev_obs) and self.Check_Obs(obs) and self.Check_Action(prev_obs, action):
        str_prev_obs = ConvObs_ToString(prev_obs)
        self.Q[str_prev_obs][str(action)] += self.alpha*(rwd + self.gamma*np.max(self.Action_Vals(obs))- self.Q[str_prev_obs][str(action)])
        return True

def ConvObs_ToString(obs):
    str_obs = " "
    obs_list = [str(x) for x in obs]
    str_obs = str_obs.join(obs_list)
    #print(str_obs)
    return str_obs

'''
class RL_Agent:
    def __init__(self):

        self.N_tx = 16  # Num of transmitter antenna elements
        self.N_rx = 16  # Num of receiver antenna elements
        #self.count = 0
        self.ptx = 30  #dB

        # gNB locations One Serving gNB_1 node, 2 visible gnB_2,gNB_3 nodes
        self.gNB = [(0, 0, 0), (20, 30, 0), (40, 60, 0)]

        # Action space parameters: |A| = 8C4 x |delta_p| =70 x 5 = 350
        self.Actions = {
            'K': 4,  # Beam_set length
            'N': 8,  # Overall beam directions
            'delta_p': [0, -1, +1, -2, +2]  # position control space along x-direction
        }
        self.K = self.Actions['K']  # Beam_set length
        self.N = self.Actions['N']  # Overall beam directions
        self.delta_p = self.Actions['delta_p']  # position control space along x-direction
        self.BeamSet = Generate_BeamDir(self.N)  # Set of all beam directions

        #observations-action mapping
        self.Q = {}

    def GenAction_Sample(self):
        delta_p = self.Actions['delta_p']
        rp_ndx = np.random.randint(0, len(delta_p))
        rdelta_p = delta_p[rp_ndx]
        rk_beams = Gen_RandomBeams(self.K, self.N)
        rk_beams.append(rdelta_p)
        return rk_beams
'''