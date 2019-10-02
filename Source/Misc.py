import numpy as np
import math
import cmath
import torch
import matplotlib
import matplotlib.pyplot as plt
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display


# conversion from dB to linear
def db2lin(val_db):
    val_lin = 10 ** (val_db / 10)
    return val_lin


# cosine over degree
def cosd(val):
    return cmath.cos(val * math.pi / 180)


# sine over degree
def sind(val):
    return cmath.sin(val * math.pi / 180)

# asin in degree
def asind(val):
    #return 180/pi*math.asin(val)
    #return np.degrees(np.sinh(val))
    c1 = cmath.asin(val)
    c2 = complex(math.degrees(c1.real), math.degrees(c1.imag))
    return c2

#deg2rad for complex number
def deg2rad(val):
    l = [val.real*cmath.pi/180, val.imag*cmath.pi/180]
    c1 = complex(np.around(l[0], decimals=4), np.around(l[1], decimals=4))
    return c1

# acosine in degree
def acosd(val):
    return np.degrees(np.sinh(val))



'''

###################
Custom Space Mapping
####################

Example:
actions = {
    ['RBS'] :[0,2,1],
    ['TBS'] :[0,2,1]
    }
Custom_Space_Mapping(actions) =
    { 0:[0,0],1:[0,1],2:[0,2],3:[1,0],4:[1,1], 5:[1,2], 6:[2,0], 7:[2,1], 8: [2,2]}

'''

def Custom_Space_Mapping(actions):

    parameter_count = len(actions.keys())
    parameter_list = []
    for key in actions.keys():
        par_range = actions[key]#[actions.keys[i]]
        parameter_list.append(list(range(par_range[0],par_range[1]+1,par_range[2])))


    #creates a list of all possible tuples from given lists of action values
    action_val_tuples = [list(x) for x in np.array(np.meshgrid(*parameter_list)).T.reshape(-1,len(parameter_list))]
    action_key_list = list(np.arange(len(action_val_tuples)))

    action_values = dict(zip(action_key_list,action_val_tuples))

    return action_values


def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episode", len(values), "\t", moving_avg_period, "episode moving avg:", moving_avg[-1], end="\r")
    #if is_ipython: display.clear_output(wait=True)


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()
