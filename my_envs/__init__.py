from gym.envs.registration import register

register(
    id='fourrooms-v0',
    entry_point='my_envs.envs:FourRoomsEnv',
)

register(
    id='manipulator2d-v0',
    entry_point='my_envs.envs:Manipulator2DEnv',
)