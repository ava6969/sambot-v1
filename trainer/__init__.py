from sam_env.sam_single import *
import gym

gym.envs.register(
     id='SamSingle-v0',
     entry_point='sam_env.sam_single:SamSingle',
     max_episode_steps=1000,
)

gym.envs.register(
     id='SamSingle2D-v0',
     entry_point='sam_env.sam_single:SamSingle2D',
     max_episode_steps=1000,
)

gym.envs.register(
     id='SamSingle-v1',
     entry_point='sam_env.sam_single:SamSingleIntraDay',
     max_episode_steps=1000,
)

gym.envs.register(
     id='DMWrapper-v0',
     entry_point='dm_env:PyProcessDmLab',
     max_episode_steps=400000,
)