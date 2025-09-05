from use_cases.moral_envs.gym_wrapper import VecEnv, GymWrapper
from use_cases.moral_envs.randomized_v3 import MAX_STEPS
from gymnasium.envs.registration import register
import gymnasium as gym
from gymnasium import spaces
import numpy as np
class MoralVecEnvProxy(VecEnv, gym.Env):
    def __init__(self, env_id=None, n_envs=3, **kwargs):
        super().__init__('randomized_v3', n_envs)
        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=float)
    def reset(self, seed=None, options=None):
        ret = VecEnv.reset(self)
        return ret, {}

    def step(self, actions):
        obs, reward, terminated, info = VecEnv.step(self, actions)
        truncated = terminated
        
        return obs, reward, terminated, truncated, info
class MoralGymWrapperProxy(GymWrapper, gym.Env):
    def __init__(self, env_id=None, **kwargs):
        GymWrapper.__init__(self, 'randomized_v3')
        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=float)
    def reset(self, seed=None, options=None):
        ret = GymWrapper.reset(self)
        return ret, {}
    def close(self):
        return GymWrapper.close(self)
    def step(self, actions):
        obs, reward, terminated, info = GymWrapper.step(self, actions)
        truncated = terminated
        info.update({'avoid': reward[3]})
        return obs, reward[0:3], terminated, truncated, info

