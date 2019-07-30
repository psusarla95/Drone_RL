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
            #print("came here")
            action = self.env.GenAction_Sample()
            #print("new action: {0}".format(action))
            str_action = ConvAct_String(action)
            #print("new str action: {0}".format(str_action))
            self.Q[str_obs][str_action] = 0.0
        else:
            action = max(self.Q[str_obs], key= self.Q[str_obs].get)
            #print("exist str action: {0}".format(action))
            eval_action = ConStr_Action(action)
            #print("exist action: {0}".format(eval_action))
            action = eval_action

        return action

    def Check_Obs(self, obs):
        str_obs = ConvObs_ToString(obs)
        if str_obs in self.Q:
            pass
        else:
            self.Q[str_obs] ={}
            a = self.env.GenAction_Sample()
            #print("check obs action: {0}".format(a))
            str_action = ConvAct_String(a)
            self.Q[str_obs][str_action] = 0.0
            #print("[RL_A] obs:{0}, a: {1}".format(str_obs, self.Q[str_obs]))
        return True

    def Check_Action(self, obs, action):
        str_obs = ConvObs_ToString(obs)
        str_action = ConvAct_String(action)
        if str_action in self.Q[str_obs]:
            #print("ac: {0}".format(str(action)))
            #print(self.Q)
            pass
        else:

            self.Q[str_obs][str_action] = 0.0
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
        str_action = ConvAct_String(action)
        self.Q[str_prev_obs][str_action] += self.alpha*(rwd + self.gamma*np.max(self.Action_Vals(obs))- self.Q[str_prev_obs][str_action])
        return True


    def Best_Action(self, obs):
        str_obs = ConvObs_ToString(obs)
        assert self.Check_Obs(obs), "%r (%s) obs is invalid" % (obs, type(obs))

        best_a = max(self.Q[str_obs], key=self.Q[str_obs].get)
        best_val = self.Q[str_obs][best_a]

        return ConStr_Action(best_a), best_val

def ConvObs_ToString(obs):
    #str_obs = " "
    #obs_list = [str(x) for x in obs]
    #str_obs = str_obs.join(obs_list)
    str_obs = str(obs)
    #print(str_obs)
    return str_obs

def ConvAct_String(action):
    str_l = str(action)
    return str_l

def ConStr_Action(str_action):
    return eval(str_action)
