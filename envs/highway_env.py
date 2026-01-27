
from copy import deepcopy
import enum
from functools import partial
import os
from typing import Any, Dict, Optional, SupportsFloat, Tuple, TypeVar
import mo_gymnasium
import numpy as np
from gymnasium import wrappers, spaces
import torch
from use_cases.firefighters_use_case.constants import STATE_MEDICAL
from use_cases.firefighters_use_case.env import HighRiseFireEnv


from gymnasium.wrappers import FlattenObservation
DiscreteSpaceInt = np.int64
from seals import util
from gymnasium import spaces

from envs.tabularVAenv import encrypt_state, grounding_func_from_matrix
import gymnasium as gym


ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")



class HighwayEnvMO(gym.Env):
    """
    A simplified two-objective MDP environment for an urban high-rise fire scenario.
    Objectives: Professionalism and Proximity
    """
    metadata = {'render.modes': ['human']}
    

    def render(self):
        return self.real_env.render()

    def __init__(self,  horizon=50, initial_state_distribution='random', **kwargs):
        self.real_env = FlattenObservation(mo_gymnasium.make("mo-highway-fast-v0"))
        self.horizon = horizon
        
        self.n_values = 3  # Three objectives: Achievement, Conformity, Safety
        """ 0: high speed reward
            1: right lane reward
            2: collision reward"""

        self.reward_space = spaces.Box(low=np.array([0.0,0.0,-1.0]), high=np.array([1.0,1.0,0.0]), shape=(self.n_values,), dtype=np.float32)
        self.reward_dim = self.n_values# No, states as indexes, observations in one-hot encoding.

        self.observation_space = self.real_env.observation_space
        print(self.real_env.unwrapped.reward_space)
        print(self.reward_space)
        print(self.real_env.observation_space)

        self.action_space = self.real_env.action_space
        
        # self.action_dim = self.real_env.action_space.n
        #         self.state_dim = self.real_env.n_states


    

    def calculate_assumed_grounding(self, variants=None, variants_save_files=None, save_folder=None, recalculate=False, **kwargs):

        pass
        
    def obtain_grounding(self, variant=None, file_save=None, recalculate=True):
        pass
    
    def reset(self, *, seed: int = None, options: dict[str, Any] = None) -> tuple[Any, dict[str, Any]]:
        #super().reset(seed=seed, options=options)
        return self.real_env.reset(seed=seed, options=options)
    

    
    def step(self, action):
        ns,r,d,t,i = self.real_env.step(action)
        i = {}
        if d or t and r[2] == -1.0:
            return ns, r, False, False, i
        else:
            return ns, r, d, t, i