import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from Source.miscfun.geometry import *

low_obs = np.array([30.0,0.0])
high_obs = np.array([100.0,np.pi])

obs_space = spaces.Box(low=low_obs,high=high_obs)
rx = np.array([0,0,0])

for i in range(100):
    dist, ue_ang = obs_space.sample()

    tx = np.array(sph2cart(ue_ang, 0, dist))  # ue_pos is(x,y)

    (az_aoa, el_aoa, temp) = cart2sph(tx[0] - rx[0], tx[1] - rx[1], tx[2] - rx[2])
    (az_aod, el_aod, temp) = cart2sph(rx[0] - tx[0], rx[1] - tx[1], rx[2] - tx[2])

    print("obs: {0}, cart co-od: {1}".format((dist, ue_ang), tx))
    print("obs: {0}, aod: {1}, aoa: {2}\n\n".format((dist,ue_ang), (az_aod,el_aod), (az_aoa, el_aoa)))
