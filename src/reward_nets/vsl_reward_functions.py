from abc import ABC, abstractmethod
from copy import deepcopy
import enum
import os
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Self, Tuple, Type, Union, override
import dill
import gymnasium as gym
from imitation.rewards import reward_nets
from matplotlib import pyplot as plt
import numpy as np
import torch as th
from imitation.util import util
from typing import cast
from imitation.data.rollout import discounted_sum

from stable_baselines3.common import preprocessing
from itertools import chain

from envs.tabularVAenv import ValueAlignedEnvironment
from src.dataset_processing.data import TrajectoryWithValueSystemRews
from src.feature_extractors import BaseRewardFeatureExtractor
from defines import CHECKPOINTS, transform_weights_to_tuple

from baraacuda.envs import living_room, deep_sea_treasure
from envs import firefighters_env_mo

def __copy_args__(self, new):
        for k, v in vars(self).items():
            
            try:
                if isinstance(v, dict) and 'env' in v.keys():
                    nv = dict()
                    for a,av in v.items():
                        if a != 'env' and a != 'features_extractor':
                            nv[a] = deepcopy(av)
                        else:
                            
                            nv[a] = av
                  
                
                else:
                    if k != 'env' and k != 'features_extractor':
                            nv = deepcopy(v)
                    else:
                        
                        nv = v
                """if isinstance(v, th.nn.Module):
                    nv.load_state_dict(v.state_dict())
                if isinstance(v, GroundingEnsemble):
                    for i in range(v.networks):
                        #nv.networks[i] = deepcopy(v.networks[i])
                        nv.networks[i].load_state_dict(v.networks[i].state_dict())"""
                setattr(new, k, nv)
            except Exception as e:
                print(e)
                pass   # non-pickelable stuff wasn't needed
        
        return new

def create_alignment_layer(new_align_func, basic_layer_class, kwargs, bias=False, dtype=th.float32):
    kwargs['dtype'] = dtype
    kwargs['bias'] = bias
    kwargs['out_features'] = 1
    new_alignment_net: LinearAlignmentLayer = basic_layer_class(**kwargs)
    if new_align_func is not None:
            kwargs['in_features'] == len(new_align_func)
            with th.no_grad():
                assert isinstance(new_align_func, tuple)
                state_dict = new_alignment_net.state_dict()
                state_dict['weight'] = th.log(
                    th.as_tensor([list(new_align_func)], dtype=dtype).clone()+1e-8)

                new_alignment_net.load_state_dict(state_dict)
    return new_alignment_net
class TrainingModes(enum.Enum):
    VALUE_SYSTEM_IDENTIFICATION = 'profile_learning'
    VALUE_GROUNDING_LEARNING = 'value_learning'
    SIMULTANEOUS = 'sim_learning'
    EVAL = 'eval'
class ConvexLinearModule(th.nn.Linear):

    def __init__(self, in_features, out_features, bias = False, device=None, dtype=None, seq=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        state_dict = self.state_dict()
        state_dict['weight'] = th.rand(*(state_dict['weight'].shape), requires_grad=True)*10
        state_dict['weight'] = state_dict['weight'] / (th.sum(state_dict['weight']) + 1e-8)

        #state_dict['weight'] = th.zeros_like(state_dict['weight'])
        #print(state_dict['weight'])
        #state_dict['weight'][0][seq] = 10000.0

        
        self.load_state_dict(state_dict)

        assert th.all(self.state_dict()['weight'] >= 0), self.state_dict()['weight']


    def forward(self, input: th.Tensor) -> th.Tensor:
        #self.weight.requires_grad_(False)
        #print("INPUT", input[0])
        w_normalized = th.nn.functional.softmax(self.weight, dim=1)
        output = th.nn.functional.linear(input, w_normalized)
        #assert th.all(output >= 0)
        #print("Weight normalized", w_normalized)
        #print("WEIGHT", self.weight)
        # assert th.all(input > 0.0)
        #print("OUTPUT", output[0])
        return output


class ConvexTensorModule(th.nn.Module):
    def __init__(self, size, init_tuple=None, dtype=th.float32, bias=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size = size
        self.dtype= dtype
        self.reset(init_tuple)

    def get_tensor_as_tuple(self):
        return tuple(self.forward().detach().cpu().numpy())

    def reset(self, init_tuple=None):
        new_profile = init_tuple
        if init_tuple is None:
            new_profile = np.random.rand(self.size)
            new_profile = new_profile/np.sum(new_profile)
            new_profile = transform_weights_to_tuple(new_profile)
        self.weights = th.tensor(
            np.array(new_profile, dtype=np.float32), dtype=self.dtype, requires_grad=True)

    def forward(self, x=None):
        return th.nn.functional.softmax(self.weights, dtype=self.dtype, dim=0)

    def parameters(self, recurse: bool = True) -> Iterator[th.nn.Parameter]:
        return iter([self.weights])


class LinearAlignmentLayer(th.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=None, data=None, n_values=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.linear_bias = bias
        self.n_values = in_features if n_values is None else n_values # TODO: This might not be the case in future works...

        with th.no_grad():
            state_dict = self.state_dict()
            random_vector = th.rand_like(state_dict['weight'])
            state_dict['weight'] = th.nn.functional.sigmoid(
                state_dict['weight']) * random_vector

            self.load_state_dict(state_dict)

    

    def forward(self, input: th.Tensor) -> th.Tensor:
        w_bounded, b_bounded = self.get_alignment_layer()

        output = th.nn.functional.linear(input, w_bounded)
        

        return output

    def get_alignment_layer(self):
        w_bounded = self.weight
        # assert th.allclose(w_bounded, th.nn.functional.softmax(self.weight))
        b_bounded = 0.0
        if self.linear_bias:
            b_bounded = self.bias
        return w_bounded, b_bounded

    def copy(self):
        with th.no_grad():
            new = self.__class__(in_features=self.in_features, out_features=self.out_features, bias=self.linear_bias, device=self.weight.device, dtype=self.weight.dtype)
            new.load_state_dict(deepcopy(self.state_dict()))
        return new

class PositiveLinearAlignmentLayer(LinearAlignmentLayer):
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None, data=None):
        super().__init__(in_features, out_features, False, device, dtype, data)

    def get_alignment_layer(self):
        w_bounded = th.nn.functional.softplus(self.weight)
        # assert th.allclose(w_bounded, th.nn.functional.softmax(self.weight))
        b_bounded = 0.0
        if self.linear_bias:
            b_bounded = th.nn.functional.sigmoid(self.bias)
        return w_bounded, b_bounded


class ConvexAlignmentLayer(LinearAlignmentLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=th.float32, data=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype, data)

    def set_weights(self, weights: tuple):
        # Convert to tensor with same dtype and device as self.weight
        pure_w = th.tensor(weights, dtype=self.weight.dtype, device=self.weight.device)
        new_weights = th.log(pure_w+1e-8)
        # Reshape to match weight shape
        new_weights = new_weights.view_as(self.weight)
        # Ensure requires_grad matches previous setting
        new_weights.requires_grad = self.weight.requires_grad
        # Update state dict in place
        with th.no_grad():
            self.weight.copy_(new_weights)
        assert th.allclose(pure_w, th.nn.functional.softmax(self.weight, dim=1, dtype=self.weight.dtype)), f"{new_weights} vs {th.nn.functional.softmax(self.weight, dim=1, dtype=self.weight.dtype)}"

    def get_alignment_layer(self):
        w_bounded = th.nn.functional.softmax(self.weight, dim=1, dtype=self.weight.dtype)
        # assert th.allclose(w_bounded, th.nn.functional.softmax(self.weight))
        b_bounded = 0.0
        
        return w_bounded, b_bounded


class PositiveBoundedLinearModule(ConvexAlignmentLayer):

    def forward(self, input: th.Tensor) -> th.Tensor:
        output = super().forward(input)
        assert th.all(output < 0.0)
        assert th.all(input < 0.0)
        return output

from imitation.rewards.reward_nets import RewardNet
class VectorModule(th.nn.Module, ABC):
    def __init__(self, *args, num_outputs, basic_classes, input_size, hid_sizes, activations, use_bias, debug=False, dtype=th.float32, **kwargs):
        super(VectorModule, self).__init__()
        
        self.hid_sizes = hid_sizes
        self.num_outputs = num_outputs
        self.input_size = input_size
        self.basic_layer_classes = basic_classes
        self.desired_dtype=dtype
        self.use_bias = use_bias
        if isinstance(self.use_bias, bool):
            self.use_bias = [self.use_bias]*len(self.hid_sizes)
        self.activations = activations
        self.debug = debug
        # Initialize multiple copies of base network with 1 output each
        self.networks = th.nn.ModuleList([
            self._create_single_output_net(seq) for seq in range(self.num_outputs)
        ])
        
    def reset(self):
        new_networks = th.nn.ModuleList([
            self._create_single_output_net(seq) for seq in range(self.num_outputs)
        ])
        for i, net in enumerate(new_networks):
            self.networks[i].load_state_dict(deepcopy(net.state_dict()))
    def copy_new(self):
        """Creates a new copy of the ensemble with the same structure but different weights."""
        return VectorModule(num_outputs=self.num_outputs, basic_classes=self.basic_layer_classes, input_size=self.input_size, hid_sizes=self.hid_sizes, activations=self.activations, use_bias=self.use_bias, debug=self.debug, dtype=self.desired_dtype)

    def get_network_for_value(self, value_id: int) -> th.nn.Module:
        """Returns the network corresponding to the given value ID."""
        if value_id < 0 or value_id >= len(self.networks):
            raise ValueError(f"Value ID {value_id} is out of bounds for the ensemble with {len(self.networks)} networks.")
        return self.networks[value_id]
    def _create_single_output_net(self, seq=0):
        """Creates a single-output version of the original network structure."""
        modules = []

        for i in range(len(self.basic_layer_classes)-1):
            if i == len(self.hid_sizes) - 1:
                input_s = self.hid_sizes[i-1]
                output_s = 1
            elif i == 0:
                input_s = self.input_size
                output_s = self.hid_sizes[i]  
            else:
                input_s = self.hid_sizes[i-1]
                output_s = self.hid_sizes[i]  
            modules.append(self.basic_layer_classes[i](input_s, output_s, bias=self.use_bias[i]))
            modules.append(self.activations[i]())
            
        return th.nn.Sequential(*modules)
    def requires_grad_(self, requires_grad = True):
        r = super().requires_grad_(requires_grad)
        for n in self.networks:
            n.requires_grad_(requires_grad)
        self.networks.requires_grad_(requires_grad)
        return r
    def __str__(self):
        
        ret = ""
        for ic in range(self.num_outputs):
            ret += (f"{ic}:" + str(self.networks[ic].state_dict()))
        
        
        return (ret + super().__str__())
    
    def forward(self, *input, **kwargs):
        return th.cat([net(*input, **kwargs) for net in self.networks], dim=1)
    
    def requires_grad_(self, requires_grad=True):
        r = super().requires_grad_(requires_grad)
        for n in self.networks:
            n.requires_grad_(requires_grad)
        self.networks.requires_grad_(requires_grad)
        return r
from imitation.data.types import Trajectory
class RewardVectorModule(VectorModule):
    @property
    def device(self):
        return self.feature_extractor.device
    def __init__(self, *args, 
                 basic_classes, num_outputs, hid_sizes, activations, use_bias,
                 feature_extractor: BaseRewardFeatureExtractor,
                 normalize_output=False,
                 normalize_output_layer=None,
                 update_stats=None,
                 clamp_rewards=None, # or use, e.g. [-100, 100]
                     debug=False):
        
        
        input_size = feature_extractor.features_dim
        super().__init__(*args, num_outputs=num_outputs, basic_classes=basic_classes, input_size=input_size, hid_sizes=hid_sizes, activations=activations, use_bias=use_bias, debug=debug, 
                         dtype=feature_extractor.dtype)
        self.feature_extractor = feature_extractor
        self.observation_space = feature_extractor._observation_space
        self.action_space = feature_extractor._action_space
        self.normalize_images = False # TODO: this is useful from the RewardNet official implementation
        self.debug = debug
        self.clamp_rewards = clamp_rewards
        self.normalize_output = normalize_output
        self.update_stats = update_stats
        if normalize_output and normalize_output_layer is not None:
            self.normalize_output_layer = normalize_output_layer(self.num_outputs)
            if self.normalize_output_layer is not None:
                self.normalize_output_layer.requires_grad_(False)
        else:
            self.normalize_output_layer = None

        self.requires_grad_(True)

        """print(self)
        a = input()
        if a == "R":
            raise KeyboardInterrupt("Stopped by user")"""
    def save(self, path: str, file_name: str = 'reward_vector'):
        # Save ALL  with dill.dump without the feature extractor
        if not os.path.exists(path):
            os.makedirs(path)
        self.feature_extractor = None  # Remove feature extractor to avoid saving it
        with open(os.path.join(path, f'{file_name}.pkl'), 'wb') as f:
            dill.dump(self, f)

    def set_mode(self, mode: str):
        self.mode = mode
        """print(self)
        print(set(self.parameters()))
        input()"""
        if self.mode == 'eval':
            self.requires_grad_(False)
            self.normalize_output = True if self.normalize_output_layer is not None else False
        elif self.mode == 'train':
            self.requires_grad_(True)
            self.normalize_output = True if self.normalize_output_layer is not None else False
        elif self.mode == 'test':
            self.requires_grad_(False)
            self.normalize_output = True if self.normalize_output_layer is not None else False

    def requires_grad_(self, requires_grad=True):
        self.update_stats = requires_grad
        if self.normalize_output_layer is not None:
            self.normalize_output_layer.requires_grad_(False)
        return super().requires_grad_(requires_grad)
    @staticmethod
    def load(path: str, file_name: str = 'reward_vector', feature_extractor=None) -> Self:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")
        with open(os.path.join(path, f'{file_name}.pkl'), 'rb') as f:
            obj: RewardVectorModule = dill.load(f)
        if feature_extractor is not None:
            obj.feature_extractor = feature_extractor

    def _forward(self, *args, **kwargs):
        rew_th = super().forward(*args, **kwargs)
        if self.clamp_rewards:
            rew_th = th.clamp(rew_th, min=self.clamp_rewards[0], max=self.clamp_rewards[1])


        if self.normalize_output:
            ret = self.normalize_output_layer(rew_th)
            if self.update_stats:
                with th.no_grad():
                    self.normalize_output_layer.update_stats(rew_th)
        else:
            ret = rew_th
        assert ret.shape[-1] == self.num_outputs, f"Output shape {ret.shape} does not match expected {self.num_outputs}"
        return ret
    def forward(self, state_or_trajectories: Union[th.Tensor, List[TrajectoryWithValueSystemRews]], action: th.Tensor=None, next_state: th.Tensor=None, done: bool=None, info=None, discount_factor=None, squeeze_output=True) -> th.Tensor:

        if discount_factor is None or discount_factor == 1.0:
            pass
        else:
            raise NotImplementedError("Discounted rewards are not implemented yet.")
        if isinstance(state_or_trajectories[0], TrajectoryWithValueSystemRews):
            rews_array = []
            for traj in state_or_trajectories:
                state_ = traj.obs[:-1] 
                action_ = traj.acts
                next_state_ = traj.obs[1:]
                done_ = traj.dones
                rews_ = self._forward(self.feature_extractor(state_, action_, next_state_, done_, info=info))
                total_rew = rews_.sum(dim=0)
                assert total_rew.shape[0] == self.num_outputs, f"Output shape {total_rew.shape} does not match expected {self.num_outputs}"
                rews_array.append(total_rew)
            rews_array = th.stack(rews_array, dim=0)
            assert rews_array.shape == (len(state_or_trajectories), self.num_outputs), f"Output shape {rews_array.shape} does not match expected {self.num_outputs}"
        elif isinstance(state_or_trajectories, (tuple, list)):
            idxs = [0]*(len(state_or_trajectories[1])+1)
            for i, ts in enumerate(state_or_trajectories[1]):
                idxs[i+1] = len(ts) + idxs[i]
            states = th.stack([si  for s in state_or_trajectories[0] for si in s], dim=0)
            actions = th.stack([ai for a in state_or_trajectories[1] for ai in a], dim=0)
            next_states = th.stack([ni  for n in state_or_trajectories[2] for ni in n], dim=0)
            dones = th.stack([di for d in state_or_trajectories[3] for di in d], dim=0)
            rews_array = self._forward(self.feature_extractor(states, actions, next_states, dones, None))
            rews_array = th.stack([th.sum(rews_array[pidx:idx], dim=0) for pidx, idx in zip(idxs[0:-1], idxs[1:])])
            assert rews_array.shape == (len(state_or_trajectories[0]), self.num_outputs), f"Output shape {rews_array.shape} does not match expected {(len(state_or_trajectories[0]), self.num_outputs)}"
        else:
            rews_array = self._forward(self.feature_extractor(state_or_trajectories, action, next_state, done, info=info))
            
        assert rews_array.device == self.device, f"Expected device {self.device}, but got {rews_array.device}"
        if squeeze_output and rews_array.shape == (1, self.num_outputs):
                rews_array = rews_array.squeeze(0)
        
        return rews_array
    def copy_new(self):
        """Creates a new copy of the ensemble with the same structure but different weights."""
        return RewardVectorModule(basic_classes=self.basic_layer_classes, 
                                  num_outputs=self.num_outputs,
                                  hid_sizes=self.hid_sizes, activations=self.activations, 
                                  use_bias=self.use_bias, debug=self.debug,clamp_rewards=self.clamp_rewards, 
                                  feature_extractor=self.feature_extractor, normalize_output=self.normalize_output, 
                                  update_stats=self.update_stats, 
                                  normalize_output_layer=self.normalize_output_layer.__class__ if self.normalize_output_layer is not None else None)
    
    def parameters(self, recurse = True):
        return chain(*list(n.parameters(recurse=recurse) for n in self.networks))
    def copy(self):
        with th.no_grad():
            new = self.copy_new()
            new: RewardVectorModule = __copy_args__(self, new)
            self.requires_grad_(True)
            new.load_state_dict(deepcopy(self.state_dict()))
            assert th.allclose(list(new.parameters())[0], list(self.parameters())[0]), f"Parameters do not match: have {list(new.parameters())[0]}\n, should be {list(self.parameters())[0]}"
        #assert list(new.parameters())[0].requires_grad
        return new
from imitation.util import networks, util
from imitation.rewards.reward_nets import PredictProcessedWrapper

class EnsembleRewardVectorModule(th.nn.Module):


    def parameters(self, recurse=True):
        return chain(*[n.parameters(recurse=recurse) for n in self.rewards])


    def set_mode(self, mode):
        self.mode = mode
        self.rewards.requires_grad_(mode == 'train')

        for i, r in enumerate(self.rewards):
            if i not in self.use_rewards_at_index:
                r.set_mode('eval')
            else:
                r.set_mode(mode)
        return self

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, index):
        return self.rewards[index]


    @property
    def device(self): 
        return self.rewards[0].device if len(self.rewards) > 0 else None
    @property
    def desired_dtype(self):
        return self.rewards[0].desired_dtype if len(self.rewards) > 0 else None
    @property
    def num_outputs(self): 
        return self.rewards[0].num_outputs if len(self.rewards) > 0 else 0

    def save(self, path, file_name = 'reward_vector'):
        for i, r in enumerate(self.rewards):
            r.save(path, file_name + f'_ensemble{i}')
    @staticmethod
    def load(path: str, file_name: str = 'reward_vector', feature_extractor=None) -> Self:
        i = 0
        rewards = []
        while True:
            try:
                r = RewardVectorModule.load(path, file_name + f'_ensemble{i}', feature_extractor=feature_extractor)
            except FileNotFoundError:
                print(f"File {file_name + f'_ensemble{i}'} not found. Stopping loading.")
                break
            if r is not None:
                break
            i += 1
            rewards.append(r)
        return EnsembleRewardVectorModule(rewards)

    
    @property
    def n_models(self):
        return len(self.rewards)
    
    @property
    def feature_extractor(self):
        return self.rewards[0].feature_extractor if len(self.rewards) > 0 else None

    def use_models(self, *indices):
        """Set the indices of the reward models to use.

        Args:
            indices (List[int]): List of indices of the reward models to use.
        """
        if indices[0] == 'all':
            self.use_rewards_at_index = np.arange(len(self.rewards))
        else:
            self.use_rewards_at_index = np.array(list(indices))
        self.set_mode(self.mode)
        assert np.all(self.use_rewards_at_index < len(self.rewards)), f"Indices {self.use_rewards_at_index} are out of bounds for the ensemble with {len(self.rewards)} models."

    def __init__(self, rewards):
        """Initialize an ensemble of reward networks.

        Args:
            rewards (List[RewardVectorModule]): List of reward vector modules.
        """
        super().__init__()
        self.rewards = th.nn.ModuleList(rewards)
        self.use_rewards_at_index = np.arange(len(self.rewards))
        self.mode = 'eval'  # Default mode is evaluation
    def copy_new(self):
        with th.no_grad():
            new = EnsembleRewardVectorModule([r.copy_new() for r in self.rewards])
            new.use_rewards_at_index = deepcopy(self.use_rewards_at_index)
            return new
    def copy(self):
        new = EnsembleRewardVectorModule([r.copy() for r in self.rewards])
        new.use_rewards_at_index = self.use_rewards_at_index.copy()
        return new
    
    def _forward(self, *args, **kwargs):
        raise NotImplementedError("Should not call this. Use forward")
    
    def forward(self, state_or_trajectories, action = None, next_state = None, done = None, info=None, discount_factor=None):
        f = None
        with networks.training(self.rewards):
            for it, index in enumerate(self.use_rewards_at_index):
                r = self.rewards[index].forward(state_or_trajectories, action, next_state, done, info, discount_factor)
                if it == 0:
                    f = r
                else:
                    f += r
        #assert f.grad is not None, f"Gradient is None for the ensemble with {len(self.rewards)} models."
        return f/len(self.use_rewards_at_index)
def squeeze_r(r_output: th.Tensor) -> th.Tensor:
    """Squeeze a reward output tensor down to one dimension, if necessary.

    Args:
         r_output (th.Tensor): output of reward model. Can be either 1D
            ([n_states]) or 2D ([n_states, 1]).

    Returns:
         squeezed reward of shape [n_states].
    """
    if r_output.ndim == 2:
        return th.squeeze(r_output, 1)
    assert r_output.ndim == 1
    return r_output



def parse_layer_name(layer_name):
        if layer_name == 'nn.Linear':
            return th.nn.Linear
        if layer_name == 'ConvexAlignmentLayer':
            return ConvexAlignmentLayer
        if layer_name == 'nn.LeakyReLU':
            return th.nn.LeakyReLU
        if layer_name == 'nn.Tanh':
            return th.nn.Tanh
        if layer_name == 'nn.Softplus':
            return th.nn.Softplus
        if layer_name == 'nn.Sigmoid':
            return th.nn.Sigmoid
        if layer_name == 'nn.Identity':
            return th.nn.Identity
        if layer_name == 'ConvexLinearModule':
            return ConvexLinearModule
        
        raise ValueError(f'Unknown layer name: {layer_name}')