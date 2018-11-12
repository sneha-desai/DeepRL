import numpy as np
import gym
import mujoco_py
from gym.envs import mujoco
AntEnv = mujoco.AntEnv()

env = gym.make('Ant-v2')
for i in np.arange(10):
    env.reset()
    for i in np.arange(1000):
        a = (np.random.rand(*env.action_space.shape) - 0.5) * 1.1
        # returns observation, reward, done, info
        o, r, d, i = env.step(a)  
        env.render()
env.close()

