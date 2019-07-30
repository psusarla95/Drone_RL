from Source.MIMO import *
from Source.RF_Env import RFBeamEnv, Generate_BeamDir
from Source.RL_Agent import RL_Agent, Custom_Space_Mapping, ConvObs_ToString, get_Values #, Eval_StrAction
import matplotlib.pyplot as plt

def play_one_episode(agent, eps, ue, vel):
    obs = agent.env.reset(ue, vel)
    ue_t = ue[:]
    iters = 0
    total_rwd = 0
    done = False
    while not done:
        action = agent.sample_action(obs, eps)

        prev_obs = obs
        ue_t[0] += vel
        obs, rwd, done = agent.env.step(action)

        agent.Update_Q(prev_obs, action, obs, rwd)
        print("Rwd: {0}, Learnt rate: {1}, Exh rate:{2}, LoS_rate:{3}".format(rwd, agent.env.get_Rate(), agent.env.get_LoS_Rate(ue_t), agent.env.get_Exh_Rate(ue_t)))

        total_rwd += rwd
        iters+=1

    return total_rwd

'''
def ConvObs_ToString(obs):
    str_obs = " "
    obs_list = [str(x) for x in obs]
    str_obs = str_obs.join(obs_list)
    #print(str_obs)
    return str_obs
'''

if __name__ == "__main__":
    ue_s = np.array([0,100,0])
    ue_v = 10
    ue_d = np.array([100,100,0])
    gnB = [0,0,0]
    Ntx = 8
    Nrx = 8
    K = 4
    N = 8
    Ptx = 30
    rate_thr = 0.7
    omega_vec = [30, 45, 60, 90]
    sc_xyz = np.array([[0, 50, 0], [20, 50, 0]])  # [10,50,0], [20,50,0]])
    ch_model = 'uma-los'
    ue_loc = np.array([np.array([x, ue_s[1], ue_s[2]]) for x in range(ue_s[0], ue_d[0] + 1, ue_v)])

    #obs = {
    #    'UE': ue_loc,  # list(np.around(np.pi/180*np.arange(180,360,1), decimals=8)), #Generate_BeamDir(self.N),
    #    'RBD': Generate_BeamDir(N)
    #}

    #print(obs)
    #obs_map = Custom_Space_Mapping(obs)
    #print(obs_map)
    #inv_obs_map = dict((str(v), k) for k, v in obs_map.items())

    #obs_ndx = 1
    #obs1 = obs_map[obs_ndx]
    #print(obs1)
    #inv_obs_ndx =  inv_obs_map[str(obs1)]
    #print(eval(str(obs1)))

    '''
    beamset = Generate_BeamDir(N)
    print(beamset)
    for beam in beamset:
        plotbeam(beam[0], N)
    '''
    '''
    env = RFBeamEnv(sc_xyz, ch_model)
    env.set_goal(ue_d)
    obs = env.reset(ue_s, ue_v)
    print(obs)

    done = False
    while not done:
        action = beamset[0]#np.arctan2(100,0)
        print(action)
        #x, y = plotbeam(action,  N)
        #plt.plot(x,y)
        #plt.show()

        new_obs, rwd, done = env.step(action)

        print('rate: ', env.get_Rate())
        print('LoS rate: ', env.get_LoS_Rate(ue_s))

        print('New_obs: ', new_obs, rwd, done)
    '''


    env = RFBeamEnv(sc_xyz, ch_model)
    alpha = 0.5
    gamma = 0.7
    eps = 0.3
    agent = RL_Agent(env, alpha, gamma, ue_loc)
    agent.env.set_rate_threshold(rate_thr)
    agent.env.set_goal(ue_d)

    obs = agent.env.reset(ue_s, ue_v)
    print("obs: {0}".format(obs))

    str_obs = ConvObs_ToString(obs)
    obs_ndx = agent.inv_obs_map[str_obs]
    ret_str_obs = agent.obs_map[obs_ndx]
    ret_obs = get_Values(agent.obs_dtypes, ret_str_obs)
    print("ret obs: {0}".format(ret_obs))
    '''
    obs = agent.env.reset(ue_s, ue_v)
    print("obs: {0}".format(obs))
    done = False

    action = agent.sample_action(obs, eps)
    print("action:{0}".format(action))

    next_obs, rwd, done = agent.env.step(action)
    print("new_obs: {0}, rwd: {1}, done: {2}".format(next_obs, rwd, done))

    print("learnt rate: {0}, los rate: {1}".format(agent.env.get_Rate(), agent.env.get_LoS_Rate(ue_s)))
    up_status = agent.Update_Q(obs, action, next_obs, rwd)
    print("up_status: {0}".format(up_status))

    action, action_val = agent.Best_Action(obs)
    print("Best action: {0}, Best action val: {1}".format(action, action_val))
    '''
    '''
    ep_rwd = play_one_episode(agent, eps, ue_s, ue_v)
    print("ep_rwd: {0}".format(ep_rwd))
    
    obs = agent.env.reset(ue_s, ue_v)
    print("obs: {0}".format(obs))
    done = False

    #while not done:
    action = agent.sample_action(obs, eps)
    

    
   
    '''

    '''

    #print(Generate_BeamDir(N))
    #print(Gen_RandomBeams(K, N))


    #possible_ue_loc = [[x, ue_s[1], ue_s[2]] for x in range(ue_s[0], ue_s[0] + ue_tdist + 1, ue_v)]


    #beamset = Generate_BeamDir(4)
    #print("beamset {0}".format(beamset))

    #print(e)

    for i in range(len(possible_ue_loc)):
        ue = possible_ue_loc[i][:]
        obs = agent.env.reset(ue)

        print("Estimated UE loc: {0}".format(obs))
        action = agent.sample_action(obs, eps)

        prev_obs = obs[:]

        obs, rwd = agent.env.step(action)
        agent.Update_Q(prev_obs, action, obs, rwd)

        print("Rate observed: {0}, Exh Rate: {1}".format(agent.env.get_Rate(), agent.env.get_Exh_Rate()))
        #los_rate = env.get_LoS_Rate()
        #print("ue: {0}, exh_rate: {1} \n".format(ue,exh_rate))
    print(agent.Q)
    '''

    '''
    for i in range(len(possible_ue_loc)):
        ue = possible_ue_loc[i][:]
        obs = env.reset(ue)
        exh_rate = env.get_Exh_Rate()
        print("ue: {0}, exh_rate: {1} \n".format(ue,exh_rate))
    '''
    #THIS WORKS!
    '''
    ue = possible_ue_loc[0][:]
    obs = env.reset(ue)
    print("Reset: random obs - {0}".format(obs))
    rand_action = env.GenAction_Sample()

    str_obs = ConvObs_ToString(obs)
    Q = {}
    Q[str_obs] = {}
    Q[str_obs][str(rand_action)] = 0.1
    if str(rand_action) in Q[str_obs]:
            print("yes", Q[str_obs][str(rand_action)])
    '''
    #print("Random Env action: {0}".format(rand_action))
    #print(str(rand_action))
    #print(eval(str(rand_action)))

    #new_obs, rwd = env.step(rand_action)
    #print("new obs: {0}, prev_obs reward: {1}".format(new_obs, rwd))


    '''
    agent = RL_Agent(env, alpha, gamma)
    ue = possible_ue_loc[0][:]
    obs = agent.env.reset(ue)
    print("obs: {0}".format(obs))
    #print("obs: {0}".format(eval(str(obs))))
    action = agent.sample_action(obs, eps)
    print("action: {0}".format(action))

    prev_obs = obs[:]

    obs, rwd = agent.env.step(action)
    agent.Update_Q(prev_obs, action, obs, rwd)
    '''

    '''
    agent = RL_Agent(env, alpha, gamma)
    print("Before episode: {0}".format(agent.Q))

    play_one_episode(agent, eps, ue)

    print("After episode: {0}".format(agent.Q))

    file = open('Q_data.txt', 'w')
    for k1, act in agent.Q.items():
        for k2,v2 in act.items():
            file.write('RIM: ' + k1 + '\t' + 'Action: '+ k2  +'\t'+ 'Value: ' + str(v2) + '\n')
    file.close()
    '''

    '''
    env = RFBeamEnv()
    agent = RL_Agent(env, alpha, gamma)
    print(agent.env.GenAction_Sample())
    print(agent.env.GenAction_Sample())
    print(agent.env.GenAction_Sample())
    '''