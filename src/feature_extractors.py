from typing import Any, Dict, Union
import gymnasium
import numpy as np
import torch
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from envs.tabularVAenv import ContextualEnv

def one_hot_encoding(a, n, dtype=torch.float32):
    v = torch.zeros(shape=(n,), dtype=dtype)
    v[a] = 1.0
    return v

def one_hot_encoding_space(observation_space, one_hot):
        modified_space = observation_space
        one_hot = one_hot and (isinstance(observation_space, spaces.Discrete) or isinstance(observation_space, spaces.MultiDiscrete))
        if one_hot:
            if isinstance(observation_space, spaces.Discrete):
                one_hot = 'discrete'
                modified_space = spaces.MultiBinary(int(observation_space.n))
            elif isinstance(observation_space, spaces.MultiDiscrete):
                one_hot = 'multi_discrete'
                modified_space = spaces.MultiBinary(int(np.prod(observation_space.nvec)))
        return modified_space, one_hot

import gymnasium.spaces as spaces
class BaseRewardFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space,
                 action_space: spaces.Space,
                 use_state: bool, use_action: bool, use_next_state: bool, use_done: bool, 
                 one_hot_actions: bool = True,
                 one_hot_observations: bool = True,
                 one_hot_dones: bool = True,
                 done_space: spaces.Space = spaces.Discrete(2),
                 device: Union[str, torch.device] = 'cpu', dtype=torch.float32):
        self.use_state = use_state
        self.use_action = use_action
        self.use_next_state = use_next_state
        self.use_done = use_done
        self.device = device
        self.dtype = dtype
        self._action_space = action_space
        self._done_space = done_space
        self._observation_space = observation_space

        self.modified_observation_space, self.one_hot_observations = one_hot_encoding_space(observation_space, one_hot_observations)
        self.modified_action_space, self.one_hot_actions = one_hot_encoding_space(action_space, one_hot_actions)
        self.modified_done_space, self.one_hot_dones = one_hot_encoding_space(done_space, one_hot_dones)

        features_dim = 0

        self.modify_observation_space()
        self.modify_action_space()
        self.modify_done_space()
        features_dim = self.features_dim
        super().__init__(observation_space, features_dim)
    def modify_observation_space(self):
        pass
    def modify_action_space(self):
        pass
    def modify_done_space(self):        
        pass
    @property
    def features_dim(self) -> int:
        #TODO Danger here. get_action_dim returns 1 with discrete space.
        return (get_flattened_obs_dim(self.modified_observation_space) if self.use_state else 0) + (get_action_dim(self.modified_action_space) if self.use_action else 0) + (get_flattened_obs_dim(self.modified_observation_space) if self.use_next_state else 0) + (get_flattened_obs_dim(self.modified_done_space) if self.use_done else 0)

    def modify_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if self.one_hot_observations == 'discrete':
            if observations[0].shape[0] != self._observation_space.n:
                observations = torch.nn.functional.one_hot(observations.long(), self._observation_space.n).to(dtype=self.dtype, device=self.device)
        elif self.one_hot_observations == 'multi_discrete':
            observations = torch.tensor([one_hot_encoding(o, n, dtype=self.dtype) for o, n in zip(observations.long(), self._observation_space.nvec)], device=self.device)
        return observations
    def modify_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.one_hot_actions == 'discrete':
            actions = torch.nn.functional.one_hot(actions.long(), self._action_space.n).to(device=self.device)
        elif self.one_hot_actions == 'multi_discrete':
                actions = torch.stack([torch.nn.functional.one_hot(a.long(), n, dtype=self.dtype) for a, n in zip(actions.long(), self._action_space.nvec)], dim=-1, device=self.device)
        #print("NEW ACTS", actions)
        return actions
    def modify_dones(self, dones: torch.Tensor) -> torch.Tensor:
        dones = torch.nn.functional.one_hot(dones.long(), 2).to(device=self.device)
        return dones
    def adapt_info(self, info: Dict) -> None:
        return
    
    def _forward(self, observations: torch.Tensor = None, actions: torch.Tensor = None, next_observations: torch.Tensor = None, dones: torch.Tensor = None, info: Dict=None) -> torch.Tensor:
        with torch.no_grad():
            features = []
            self.adapt_info(info)
            if self.use_state and observations is not None:

                new_obs = self.modify_observations(observations)
                features.append(torch.flatten(new_obs, 1))
                assert self.modified_observation_space.contains(new_obs[0]), f"Observation {new_obs[0]} ({new_obs[0].shape}) is not in the observation space {self.modified_observation_space}"
            if self.use_action and actions is not None:
                new_acts = self.modify_actions(actions)
                acts_insert = torch.flatten(new_acts, 1)
                features.append(acts_insert)
                assert self.modified_action_space.contains(new_acts[0].detach().numpy()), f"Action {new_acts[0]} is not in the action space {self.modified_action_space}"
            if self.use_next_state and next_observations is not None:
                new_obs_next = self.modify_observations(next_observations)
                features.append(torch.flatten(new_obs_next, 1))
                assert self.modified_observation_space.contains(new_obs_next[0]), f"Next observation {new_obs_next[0]} is not in the observation space {self.modified_observation_space}"
            if self.use_done and dones is not None:
                new_dones = self.modify_dones(dones.flatten())
                dones_insert = torch.flatten(new_dones, 1)
                features.append(dones_insert)
                assert self.modified_done_space.contains(new_dones[0].detach().numpy()), f"Done {new_dones[0]} is not in the done space {self.modified_done_space}"
            ret = torch.cat(features, dim=1).to(dtype=self.dtype, device=self.device) # TODO unsure if here or before...
            assert ret.dtype == self.dtype, f"Expected dtype {self.dtype}, but got {ret.dtype}"
            assert ret.shape[1] == self.features_dim, f"Expected features_dim {self.features_dim}, but got {ret.shape[1]}"
            ret.requires_grad_(False)
            return ret
    
    def forward(self, state=None, action=None, next_state=None, done=None, info=None) -> torch.Tensor:
        # convert inputs to tensors if they are not already
        assert state is not None or action is not None or next_state is not None or done is not None, "At least one input must be provided"
        
        if state is not None and not isinstance(state, torch.Tensor):
            assert isinstance(state, (torch.Tensor, np.ndarray)), f"Expected state to be Tensor or ndarray, but got {type(state)}"
            state = torch.tensor(state, dtype=self.dtype, device=self.device)
            #assert self.modified_observation_space.contains(state[0])

        if action is not None and not isinstance(action, torch.Tensor):
            assert isinstance(action, (torch.Tensor, np.ndarray, np.int64, int)), f"Expected action to be Tensor or ndarray, but got {type(action)}"
            #assert self._action_space.contains(action[0]), f"Action {action[0]} is not in the action space {self._action_space}"
            action = torch.tensor(action, dtype=self.dtype, device=self.device)
            #assert self.modified_action_space.contains(action[0])
        if next_state is not None and not isinstance(next_state, torch.Tensor):
            assert isinstance(next_state, (torch.Tensor, np.ndarray)), f"Expected next_state to be Tensor or ndarray, but got {type(next_state)}"
            #assert self._observation_space.contains(next_state[0]), f"Next state {next_state} is not in the observation space {self._observation_space}"
            next_state = torch.tensor(next_state, dtype=self.dtype, device=self.device)
            #assert self.modified_observation_space.contains(next_state) 
        if done is not None and not isinstance(done, torch.Tensor):
            assert isinstance(done, (torch.Tensor, np.ndarray, bool, np.bool_)), f"Expected done to be Tensor or ndarray, but got {type(done)}"
            #assert self._done_space.contains(done), f"Done {done} is not in the done space {self.modified_done_space}"
            done = torch.tensor(done, dtype=self.dtype, device=self.device)
            #assert self.modified_done_space.contains(done)

        if state is not None and len(state.shape) == 1:
            state = state.unsqueeze(0)
            assert self._observation_space.shape == state[0].shape, f"State {state[0]} is not in the observation space {self._observation_space}"
            
        if action is not None and len(action.shape) < 1:
            action = action.unsqueeze(0)
            assert self._action_space.shape == action[0].shape, f"Action {action[0]} is not in the action space {self._action_space}"
            
        if next_state is not None and len(next_state.shape) == 1:
            next_state = next_state.unsqueeze(0)
            assert self._observation_space.shape == next_state[0].shape, f"State {next_state[0]} is not in the observation space {self._observation_space}"

        if done is not None and len(done.shape) < 1:
            done = done.unsqueeze(0)
            assert self._done_space.shape == done[0].shape, f"Done {done[0]} is not in the done space {self._done_space}"

        return self._forward(state, action, next_state, done, info)
    
    
    
class ObservationMatrixFeatureExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor that performs one-hot encoding on integer observations.
    """

    def __init__(self, observation_space, observation_matrix, dtype=np.float32):
        # The output of the extractor is the size of one-hot encoded vectors
        super(ObservationMatrixFeatureExtractor, self).__init__(
            observation_space, features_dim=observation_matrix.shape[1])
        self.observation_matrix = torch.tensor(
            observation_matrix, dtype=dtype, requires_grad=False)

    def forward(self, observations):
        # Convert observations to integers (if needed) and perform one-hot encoding
        """batch_size = observations.shape[0]
        one_hot = torch.zeros((batch_size, self.n_categories), device=observations.device)

        one_hot.scatter_(1, observations.long(), 1)"""
        with torch.no_grad():
            idx = observations
            """if idx.shape[-1] > 1:
                ret =  torch.vstack([self.observation_matrix[id] for id in idx.bool()])
            else:
                ret =  torch.vstack([self.observation_matrix[id] for id in idx.long()])"""
            if idx.shape[-1] > 1:
                # Convert idx to a boolean mask and use it to index the observation_matrix
                mask = idx.bool()
                # Get indices where mask is True
                selected_indices = mask.nonzero(as_tuple=True)[-1]
                assert len(selected_indices) == observations.shape[0]
                # Select the first 32 True indices to maintain the output shape
                ret = self.observation_matrix[selected_indices]
            else:
                # Directly index the observation_matrix using long indices
                ret = self.observation_matrix[idx.view(-1).long()]
        return ret

    
class ObservationMatrixRewardFeatureExtractor(BaseRewardFeatureExtractor):
    def __init__(self, observation_matrix, **kwargs) -> None:
        assert kwargs['observation_space'] == spaces.Discrete(observation_matrix.shape[0])
        kwargs['one_hot_observations'] = False
        super().__init__(**kwargs)
        #self.env = kwargs['env']
        self.torch_obs_mat = torch.tensor(observation_matrix, dtype=self.dtype, device=self.device )
    def modify_observations(self, observations: torch.Tensor) -> torch.Tensor:
        return self.torch_obs_mat[observations.long()]
    def modify_observation_space(self):
        self.modified_observation_space = spaces.Box(
            low=self.torch_obs_mat.min().item(),
            high=self.torch_obs_mat.max().item(),
            shape=self.torch_obs_mat.shape[1:],
            dtype=self.dtype
        )
        return self.modified_observation_space
    
from imitation.util import util
class ObservationWrapperFeatureExtractor(BaseRewardFeatureExtractor):


    def __init__(self, method, observation_space, action_space, use_state, use_action, use_next_state, use_done, one_hot_actions = True, one_hot_observations = True, one_hot_dones = True, done_space = spaces.Discrete(2), device = 'cpu', dtype=torch.float32):
        self.method = method
        super().__init__(observation_space, action_space, use_state, use_action, use_next_state, use_done, one_hot_actions, one_hot_observations, one_hot_dones, done_space, device, dtype)
    
    def modify_observation_space(self):
        obs = self.method(self._observation_space.sample())
        self.modified_observation_space = spaces.Box(
            low=-100000.0,
            high=100000.0,
            shape=obs.shape,
            dtype=obs.dtype
        )
        return self.modified_observation_space
    def modify_observations(self, observations):
        return util.safe_to_tensor(self.method(observations))
