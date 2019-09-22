from gym.envs.registration import register

register(
    id='uav-v0',
    entry_point='gym_uav.envs:UAV_Env',
)

register(
    id='uav-v2',
    entry_point='gym_uav.envs:UAV_Env_v2',
)

register(
    id='uav-v3',
    entry_point='gym_uav.envs:UAV_Env_v3',
)