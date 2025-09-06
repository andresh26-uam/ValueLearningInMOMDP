from abc import abstractmethod
from copy import deepcopy
import math
import time
from typing import override

from morl_baselines.multi_policy.pcn.pcn import PCN, Transition
from morl_baselines.multi_policy.envelope.envelope import Envelope
import numpy as np

from baraacuda.utils.wrappers import RewardVectorFunctionWrapper
import torch

from defines import transform_weights_to_tuple
from src.algorithms.preference_based_vsl_lib import probability_BT
from src.algorithms.preference_based_vsl_lib import probs_to_label
from src.dataset_processing.data import FixedLengthVSLPreferenceDataset, TrajectoryWithValueSystemRews, VSLPreferenceDataset
from src.reward_nets.vsl_reward_functions import RewardVectorModule
from imitation.util import util
from morl_baselines.common.networks import get_grad_norm
from abc import abstractmethod
from copy import deepcopy
import math
import time

from morl_baselines.multi_policy.pcn.pcn import PCN, Transition
from morl_baselines.multi_policy.envelope.envelope import Envelope
import numpy as np


from morl_baselines.common.utils import linearly_decaying_value
from morl_baselines.common.networks import polyak_update
from morl_baselines.common.weights import random_weights
from colorama import Fore, Style

from baraacuda.utils.wrappers import RewardVectorFunctionWrapper
import torch

from defines import transform_weights_to_tuple
from src.algorithms.preference_based_vsl_lib import probability_BT
from src.algorithms.preference_based_vsl_lib import probs_to_label
from src.dataset_processing.data import FixedLengthVSLPreferenceDataset, TrajectoryWithValueSystemRews, VSLPreferenceDataset
from src.reward_nets.vsl_reward_functions import RewardVectorModule
from imitation.util import util

from src.utils import most_recent_indices_to_ptr

class MOCustomRewardVector:
    def __init__(self):
        self.reward_vector = None

    @abstractmethod
    def set_reward_vector_function(self, reward_vector):
        self.reward_vector = reward_vector

        if self.env.has_wrapper_attr("set_reward_vector_function"):
            self.env.get_wrapper_attr("set_reward_vector_function")(reward_vector)
        else:
            self.env = RewardVectorFunctionWrapper(self.env, reward_vector)

    
    
from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.prioritized_buffer import PrioritizedReplayBuffer
import imitation.data.rollout as rollout
class CustomRewardReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, obs_shape, action_dim, rew_dim=1, estimated_horizon=20, buffer_with_weights=True, max_size=100000, obs_dtype=np.float32, action_dtype=np.float32, min_priority=0.00001, relabel_buffer=True, reward_vector_function = None, maintain_original_reward=False, prioritized=True):
        
        super().__init__(obs_shape, action_dim, rew_dim, max_size, obs_dtype, action_dtype, min_priority=min_priority)
        self.relabel_buffer = relabel_buffer
        self.prioritized = prioritized

        self.buffer_with_weights= buffer_with_weights
        self.reward_vector_function = reward_vector_function
        self.maintain_original_reward = maintain_original_reward
        self.estimated_horizon = estimated_horizon
        self.weights = np.zeros((max_size, rew_dim), dtype=np.float32)
        if self.maintain_original_reward:
            self._original_rewards = np.array(self.rewards).copy()

    def set_reward_vector_function(self, reward_vector_function):
        self.reward_vector_function = reward_vector_function
        if self.relabel_buffer:
            self.rewards = self.reward_vector_function(self.obs, self.actions.flatten(), self.next_obs, self.dones).detach().cpu().numpy()
            if self.maintain_original_reward:
                assert self.rewards.shape == self._original_rewards.shape, f"Rewards shape mismatch: {self.rewards.shape} vs {self._original_rewards.shape}"
                


    def add(self, obs, action, reward, next_obs, done, original_rew=None, weights=None):
        r = reward.detach().numpy()
        ptr_old = self.ptr
        super().add(obs, action, r, next_obs, done)
        if weights is not None:
            self.weights[self.ptr - 1] = weights
        if self.maintain_original_reward:
            assert original_rew is not None, "original_rew must be provided if maintain_original_reward is True"
            self._original_rewards[self.ptr - 1] = np.array(original_rew).copy()
            assert self.ptr - 1 == ptr_old or self.ptr - 1 == - 1, f"Pointer mismatch: {self.ptr - 1} vs {ptr_old} vs {len(self._original_rewards) - 1}"
            """for i in range(len(original_rew)):
                assert any(np.isclose(original_rew[i], v, atol=1e-6) for v in [-1.0, -0.1, -0.5, 0.0, 0.4, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), f"Invalid original_rew: {original_rew}"
            """#print(action, obs, next_obs, done, reward)
            assert isinstance(action, (int, np.uint8, np.int64)), type(action)
            #r2 = self.reward_vector_function(obs, action, next_obs, done).detach().numpy()
            #assert np.allclose(r, r2), f"Reward mismatch: {r} vs {r2}"
            #assert r.shape == self.rewards.shape[-1] or r.shape == (self.rewards.shape[-1],), f"Reward shape mismatch: {r.shape} vs {self.rewards.shape[-1]}"
            #self.rewards[self.ptr - 1] = r # TODO..
            assert self.rewards.shape == self._original_rewards.shape, f"Rewards shape mismatch: {self.rewards.shape} vs {self._original_rewards.shape}"
    def sample(self, batch_size, to_tensor=False, device=None, replace=True, use_cer=True):
        if self.prioritized:
            idxes_random = self.tree.sample(batch_size//2)
            idxes_random = np.remainder(idxes_random, self.size)
        else:
            idxes_random = np.random.choice(self.size, batch_size//2, replace=replace)

        if use_cer:
            idxes_cer = np.random.randint(self.ptr-self.estimated_horizon*batch_size//2, self.ptr, size=batch_size//2)

        idxes = np.concatenate([idxes_random, idxes_cer])
        experience_tuples = [
            self.obs[idxes],
            self.actions[idxes],
            self.next_obs[idxes],
            self.dones[idxes],
        ]
        if to_tensor:
            experience_tuples = list(map(lambda x: torch.tensor(x).to(device), experience_tuples))  # , weights)
                
            if not isinstance(self.rewards, torch.Tensor):
                experience_tuples.insert(2, util.safe_to_tensor(self.rewards[idxes], dtype=th.float32).to(device))
            else: 
                experience_tuples.insert(2, self.rewards[idxes].float().to(device))
            
            experience_tuples = tuple(experience_tuples)
        else:
            if isinstance(self.rewards, torch.Tensor):
                experience_tuples.insert(2, util.safe_to_numpy(self.rewards[idxes]))
            else:
                experience_tuples.insert(2, self.rewards[idxes])
        if self.buffer_with_weights:
            experience_tuples = tuple(experience_tuples) + (self.weights[idxes],)
        if self.prioritized:
            return tuple(experience_tuples) + (idxes,)
        else:
            return tuple(experience_tuples)

    def sample_trajs(self, ns, to_tensor=False, device=None, get_rewards_orig=True, gamma=None, H=None):
        if get_rewards_orig:
            assert self.maintain_original_reward, "To sample original rewards, maintain_original_reward must be True."
            assert self._original_rewards is not None, "Original rewards must be stored."
            assert self._original_rewards.shape == self.rewards.shape, f"Original rewards shape {self._original_rewards.shape} does not match rewards shape {self.rewards.shape}."
            assert gamma is not None, "Gamma must be provided to compute discounted rewards."
            assert gamma > 0, "Gamma must be positive."
            assert gamma <= 1.0, "Gamma must be less than or equal to 1."
        """Sample ns trajectories from the buffer."""
        # First find dones. And go backwards until finding another done or the beginning of the replay buffer.
        idxes_dones = np.where(self.dones == 1)[0]
        if len(idxes_dones) < 3:
            raise GeneratorExit("Not enough done states in the replay buffer to sample trajectories.")
        
        states1, states2 = [], []
        acts1, acts2 = [], []
        next_states1, next_states2 = [], []
        dones1, dones2 = [], []
        rewards_original1, rewards_original2 = [], []
        rewards1, rewards2 = [], []

        dsrewards_original1, dsrewards_original2 = [], []
        dsrewards1, dsrewards2 = [], []
        idxs_valid = idxes_dones[0:-1]
        assert len(idxs_valid) == len(idxes_dones)-1, f"Valid idxes length mismatch: {len(idxs_valid)} vs {len(idxes_dones)}"
        assert len(idxs_valid) >= 2, f"Not enough valid done states to sample trajectories: {len(idxs_valid)}"
        counter = 0
        while True:
                ptr = self.ptr
                using_H = H is not None
                if not using_H:

                    #SHOULD BE 
                    closest_idxs, _ = most_recent_indices_to_ptr(ns, ptr, self.size, self.max_size, idxs_valid=idxs_valid)
    
                    
                    #idxs_sorted = np.argsort(ptr-idxs_valid)
                    #closest_idxs = idxs_sorted[:ns] if len(idxs_sorted) >= ns else idxs_sorted
                    # Now sample two different indices from these closest
                    
                    d1, d2 = np.random.choice(closest_idxs, size=2, replace=False)
                    start_idx1 = idxes_dones[d1]+1
                    end_idx1 = idxes_dones[d1 + 1]+1

                    start_idx2 = idxes_dones[d2]+1
                    end_idx2 = idxes_dones[d2 + 1]+1

                    assert self.dones[end_idx1-1] == 1, f"End idx 1 {end_idx1} is not done."
                    assert self.dones[end_idx2-1] == 1, f"End idx 2 {end_idx2} is not done."
                    
                    max_starting_point_1 = min(end_idx1 - start_idx1, H) if using_H else end_idx1 - start_idx1
                    max_starting_point_2 = min(end_idx2 - start_idx2, H) if using_H else end_idx2 - start_idx2
                    if using_H:
                        assert ValueError("not possible, this was old implementation")
                        h1_start = np.random.randint(0, max_starting_point_1)
                        h2_start = np.random.randint(0, max_starting_point_2)
                        max_starting_point_1 = H + h1_start
                        max_starting_point_2 = H + h2_start
                        
                    else:
                        h1_start = 0
                        h2_start = 0
                else:
                    # Only consider valid indices within the buffer size and estimated horizon window
                    window_start = max(self.size - self.estimated_horizon * ns, 0)
                    window_end = self.size  # Only up to valid data
                    possible_idxs = np.arange(window_start, window_end, step=H)
                    if len(possible_idxs) < 2:
                        raise ValueError("Not enough valid indices to sample two distinct trajectories.")
                    d1, d2 = np.random.choice(possible_idxs, size=2, replace=False)
                    start_idx1 = d1
                    
                    start_idx2 = d2
                    max_starting_point_1 = H
                    max_starting_point_2 = H
                    end_idx1 = start_idx1 + H
                    end_idx2 = start_idx2 + H 
                    h1_start = 0
                    h2_start = 0
                #d1, d2 = np.random.choice(len(idxs_valid), size=2, replace=False, p=)
                assert d1 != d2
                #print("sampled?", d1, d2, "from", len(idxes_dones[0:-1]))
                

                traj_obs1, traj_actions1, traj_next_obs1, trajs_rewards_orig1, traj_real_rewards1, traj_dones1 = self._get_traj_data(to_tensor, device, start_idx1+h1_start, start_idx1+max_starting_point_1, self.dones, using_H)

                traj_obs2, traj_actions2, traj_next_obs2, trajs_rewards_orig2, traj_real_rewards2, traj_dones2 = self._get_traj_data(to_tensor, device, start_idx2+h2_start, start_idx2+max_starting_point_2, self.dones, using_H)
                assert start_idx1 + h1_start != start_idx2 + h2_start, f"Sampled identical starting points: {start_idx1 + h1_start}"
                assert end_idx1 != end_idx2, f"Sampled identical ending points: {end_idx1}"
                assert end_idx1 - (start_idx1 + h1_start) > 0, f"Trajectory 1 has non-positive length: {end_idx1} - ({start_idx1} + {h1_start})"
                assert len(traj_obs1) > 0, f"Trajectory 1 has non-positive length: {len(traj_obs1)}"
                """if len(traj_obs1) == len(traj_obs2):
                    if np.allclose(traj_obs1, traj_obs2) and np.allclose(traj_actions1, traj_actions2):
                        print(f"Sampled identical trajectories.{traj_actions1, traj_actions2}")
                        continue"""
                if H is None:
                    assert traj_dones1[-1] == 1, f"Last done state in trajectory 1 is not 1: {traj_dones1[-1]}"
                    assert all(traj_dones1[0:-1] == 0), f"Non-done states in trajectory 1 are not 0: {traj_dones1[0:-1]}"
                #else:
                    #assert len(traj_obs1) == H, f"Trajectory 1 length {len(traj_dones1)} exceeds H {H}."
                assert len(traj_dones1) == len(traj_obs1) == len(traj_actions1) == len(traj_next_obs1) == len(trajs_rewards_orig1) == len(traj_real_rewards1), \
                    f"Length mismatch in trajectory 1: {len(traj_dones1)}, {len(traj_obs1)}, {len(traj_actions1)}, {len(traj_next_obs1)}, {len(trajs_rewards_orig1)}, {len(traj_real_rewards1)}"
                states1.append(traj_obs1)
                acts1.append(traj_actions1)
                next_states1.append(traj_next_obs1)
                dones1.append(traj_dones1)
                states2.append(traj_obs2)
                acts2.append(traj_actions2)
                next_states2.append(traj_next_obs2)
                dones2.append(traj_dones2)
                rewards1.append(traj_real_rewards1)
                rewards2.append(traj_real_rewards2)
                assert len(traj_real_rewards1) > 0
                assert traj_next_obs1.shape[-1] == traj_obs2.shape[-1] == self.obs.shape[-1], f"Observation shape mismatch: {traj_next_obs1.shape} vs {traj_obs2.shape} vs {self.obs.shape}"
                assert len(traj_real_rewards2) == len(traj_next_obs2) == len(traj_obs2) == len(traj_actions2) == len(traj_dones2)
                assert len(traj_real_rewards1) == len(traj_next_obs1) == len(traj_obs1) == len(traj_actions1) == len(traj_dones1)

                assert len(traj_real_rewards1.shape) == 2
                rewards_original1.append(trajs_rewards_orig1)
                rewards_original2.append(trajs_rewards_orig2)
                #print("RO?", rewards_original1)
                #input()
                if H is not None:
                    assert len(trajs_rewards_orig1) == len(trajs_rewards_orig1) <= H, f"Sampled {len(trajs_rewards_orig1)} rewards, expected {H}."
                else:
                    assert len(trajs_rewards_orig1) >= 1
                
                assert len(trajs_rewards_orig1.shape) == 2
                dsrewards_original1.append(rollout.discounted_sum(trajs_rewards_orig1, gamma=gamma))
                dsrewards_original2.append(rollout.discounted_sum(trajs_rewards_orig2, gamma=gamma))
                dsrewards1.append(rollout.discounted_sum(traj_real_rewards1, gamma=gamma))
                dsrewards2.append(rollout.discounted_sum(traj_real_rewards2, gamma=gamma))
                counter+=1
                if counter >= ns:
                    break
        obs_shapes = [len(frag) for frag in rewards1]
        
        if len(set(obs_shapes)) == 1:
            states1 = np.asarray(states1)
            #assert states1.shape == (len(states2), obs_shapes[0], 19)
            acts1 = np.asarray(acts1)
            next_states1 = np.asarray(next_states1)
            assert len(states1) == len(acts1) == len(next_states1) == len(dones1), f"Length mismatch: {len(states1)}, {len(acts1)}, {len(next_states1)}, {len(dones1)}"
            #np.testing.assert_allclose(states1[1], next_states1[0])
            dones1 = np.asarray(dones1)
            states2 = np.asarray(states2)
            acts2 = np.asarray(acts2)
            next_states2 = np.asarray(next_states2)
            dones2 = np.asarray(dones2)

        else:

            states1 = np.array(states1, dtype=np.ndarray)
            acts1 = np.array(acts1, dtype=np.ndarray)
            next_states1 = np.array(next_states1, dtype=np.ndarray)
            dones1 = np.array(dones1, dtype=np.ndarray)
            states2 = np.array(states2, dtype=np.ndarray)
            acts2 = np.array(acts2, dtype=np.ndarray)
            next_states2 = np.array(next_states2, dtype=np.ndarray)
            dones2 = np.array(dones2, dtype=np.ndarray)
        assert len(states1) == len(states2) == ns, f"Sampled {len(states1)} trajectories, expected {ns}."
        assert len(acts1) == len(acts2) == ns, f"Sampled {len(acts1)} actions, expected {ns}."
        assert len(next_states1) == len(next_states2) == ns, f"Sampled {len(next_states1)} next states, expected {ns}."
        assert len(dones1) == len(dones2) == ns, f"Sampled {len(dones1)} dones, expected {ns}."
        assert len(rewards_original1) == len(rewards_original2) == ns, f"Sampled {len(rewards_original1)} original rewards, expected {ns}."
        assert len(rewards1) == len(rewards2) == ns, f"Sampled {len(rewards1)} rewards, expected {ns}."
        assert rewards1[0].shape == (rewards1[0].shape[0], self.reward_vector_function.num_outputs), f"Rewards shape mismatch: {rewards1.shape} vs {rewards2.shape} vs {(ns, self.reward_vector_function.num_outputs)}"
        obs_shapes = [frag.shape[0] for frag in rewards1]
        if len(set(obs_shapes)) > 1:
            rewards1_th = np.array([util.safe_to_tensor(r).to(device) for r in rewards1], dtype=th.types.Tensor)
            rewards2_th = np.array([util.safe_to_tensor(r).to(device)  for r in rewards2], dtype=th.types.Tensor)

            rewards_original1_th = np.array([util.safe_to_tensor(r).to(device) for r in rewards_original1], dtype=th.types.Tensor)
            rewards_original2_th = np.array([util.safe_to_tensor(r).to(device) for r in rewards_original2], dtype=th.types.Tensor)
        else:
            rewards1_th = util.safe_to_tensor(np.array(rewards1), device=device)
            rewards2_th = util.safe_to_tensor(np.array(rewards2), device=device)
            rewards_original1_th = util.safe_to_tensor(np.array(rewards_original1), device=device)
            rewards_original2_th = util.safe_to_tensor(np.array(rewards_original2), device=device)
        assert len(rewards1) == len(rewards2) == ns, f"Sampled {len(rewards1)} rewards, expected {ns}."
        assert rewards1_th[0].shape == (rewards1_th[0].shape[0], self.reward_vector_function.num_outputs), f"Rewards shape mismatch: {rewards1.shape} vs {rewards2.shape} vs {(ns, self.reward_vector_function.num_outputs)}"
        
        assert len(states1) == len(states2) == ns, f"Sampled {len(states1)} trajectories, expected {ns}."
        #print(rewards1_th.shape)
        assert rewards1_th.shape[0] == rewards2_th.shape[0] == ns, f"Rewards shape mismatch: {rewards1_th.shape} vs {rewards2_th.shape} vs {(ns,)}"
        if H is not None and len(set(obs_shapes)) == 1:
            assert rewards1_th.shape == rewards2_th.shape == (ns, H, rewards1_th.shape[-1]), f"Rewards shape mismatch: {rewards1_th.shape} vs {rewards2_th.shape} vs {(ns, H)}"
        dsrewards1_th = util.safe_to_tensor(np.array(dsrewards1), device=device)
        dsrewards2_th = util.safe_to_tensor(np.array(dsrewards2), device=device)
        dsrewards_original1_th = util.safe_to_tensor(np.array(dsrewards_original1), device=device)
        dsrewards_original2_th = util.safe_to_tensor(np.array(dsrewards_original2), device=device)

        """valid_values = np.array([-1.0, -0.1, -0.5, 0.0, 0.4, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # Check all values in _original_rewards are close to one of the valid values
        if not np.all(np.isclose(self._original_rewards[..., None], valid_values, atol=1e-6).any(axis=-1)):
            raise AssertionError(f"Invalid original_rew: {self._original_rewards}")
        if not np.all(np.isclose(rewards_original1_th[..., None], valid_values, atol=1e-6).any(axis=-1)):
            raise AssertionError(f"Invalid original_rew: {rewards_original1_th}")
        if not np.all(np.isclose(rewards_original2_th[..., None], valid_values, atol=1e-6).any(axis=-1)):
            raise AssertionError(f"Invalid original_rew: {rewards_original2_th}")"""

        return (states1, acts1, next_states1, dones1), (states2, acts2, next_states2, dones2), rewards_original1_th, rewards_original2_th, rewards1_th, rewards2_th, dsrewards_original1_th, dsrewards_original2_th, dsrewards1_th, dsrewards2_th

    def _get_traj_data(self, to_tensor, device, start_idx, end_idx, dones, using_H):
        assert start_idx < end_idx, f"Start index {start_idx} must be less than end index {end_idx}."
        if not using_H:
            assert dones[end_idx-1]
            if start_idx != 0:
                assert dones[start_idx-1]
        

        #print(start_idx, end_idx, "START END", initial_point, other_candidate)
        
        assert start_idx < end_idx
        traj_obs = self.obs[start_idx:end_idx]
        assert len(traj_obs) > 0, f"Trajectory has non-positive length: {len(traj_obs)}"
        traj_actions = self.actions[start_idx:end_idx].flatten()
        traj_next_obs = self.next_obs[start_idx:end_idx]
        #print(dones[start_idx:end_idx])
        #assert np.allclose(traj_obs[1:], traj_next_obs[0:-1]), f"{start_idx}, {end_idx}, {len(self.obs)}"
        traj_rewards_orig = self._original_rewards[start_idx:end_idx]
        
        traj_dones = self.dones[start_idx:end_idx].flatten()
        traj_rewards = self.rewards[start_idx:end_idx]
        assert len(traj_rewards) == len(traj_obs) == len(traj_actions) == len(traj_next_obs) == len(traj_dones) == len(traj_rewards_orig), f"Length mismatch in trajectory: {len(traj_rewards)}, {len(traj_obs)}, {len(traj_actions)}, {len(traj_next_obs)}, {len(traj_dones)}, {len(traj_rewards_orig)}"
        #assert not any(traj_dones[0:-1])

        if to_tensor:
            traj_rewards = util.safe_to_tensor(traj_rewards, device=device)
            traj_rewards_orig = util.safe_to_tensor(traj_rewards_orig, device=device)
        return traj_obs,traj_actions,traj_next_obs,traj_rewards_orig,traj_rewards, traj_dones
    

import pprint

import numpy as np
import torch as th
import wandb

from morl_baselines.common.evaluation import (
    log_all_multi_policy_metrics,
    log_episode_info,
)
from morl_baselines.common.prioritized_buffer import PrioritizedReplayBuffer
from morl_baselines.common.weights import equally_spaced_weights, random_weights


class EnvelopeCustomReward(Envelope, MOCustomRewardVector):
    def __init__(self, env, buffer_with_weights, maintain_original_reward, learning_rate = 0.0003, estimated_horizon=10, initial_epsilon = 0.01, final_epsilon = 0.01, epsilon_decay_steps = None, tau = 1, target_net_update_freq = 200, buffer_size = ..., net_arch = ..., batch_size = 256, learning_starts = 100, gradient_updates = 1, gamma = 0.99, max_grad_norm = 1, envelope = True, num_sample_w = 4, per = True, per_alpha = 0.6, initial_homotopy_lambda = 0, final_homotopy_lambda = 1, homotopy_decay_steps = None, project_name = "MORL-Baselines", experiment_name = "Envelope", wandb_entity = None, log = True, seed = None, device = "auto", activation = None, group = None, masked = False, 
                 relabel_buffer=True, reward_vector_function=None):
        Envelope.__init__(self, env, learning_rate, initial_epsilon, final_epsilon, epsilon_decay_steps, tau, target_net_update_freq, buffer_size, net_arch, batch_size, learning_starts, gradient_updates, gamma, max_grad_norm, envelope, num_sample_w, per, per_alpha, initial_homotopy_lambda, final_homotopy_lambda, homotopy_decay_steps, project_name, experiment_name, wandb_entity, log, seed, device, activation, group, masked)
        MOCustomRewardVector.__init__(self)
        self.replay_buffer = CustomRewardReplayBuffer(
            prioritized=per,
            obs_shape=self.observation_shape,
            estimated_horizon=estimated_horizon,
            action_dim=1,
            buffer_with_weights=buffer_with_weights,
            rew_dim=self.reward_dim,
            max_size=buffer_size,
            action_dtype=np.uint8,
            obs_dtype=env.observation_space.dtype,
            min_priority=0.01,#per_alpha if per else 0.1,
            relabel_buffer=relabel_buffer,
            reward_vector_function=reward_vector_function,
            maintain_original_reward=maintain_original_reward,
            
        )
    def set_buffer(self, buffer):
        self.replay_buffer = buffer
        
    def set_reward_vector_function(self, reward_vector_function, warn_relabel_not_possible=False):
        MOCustomRewardVector.set_reward_vector_function(self, reward_vector_function)
        self.replay_buffer.set_reward_vector_function(reward_vector_function)
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

class MORecordEpisodeStatisticsCR(MORecordEpisodeStatistics):
    def __init__(self, env, gamma = 1, buffer_length = 100, stats_key = "episode"):
        super().__init__(env, gamma, buffer_length, stats_key)
    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        # This is very close the code from the RecordEpisodeStatistics wrapper from Gymnasium.
        (
            observation,
            rewards,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)
        assert isinstance(
            info, dict
        ), f"`info` dtype is {type(info)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        
        if self.env.has_wrapper_attr("set_reward_vector_function"):
            real_reward = info.get('untransformed_reward', 0)
            
        # CHANGE: The discounted returns are also computed here
        else:
            real_reward = rewards
        if isinstance(real_reward, th.Tensor):
            real_reward_ = real_reward.detach().cpu().numpy()
        else:
            real_reward_ = real_reward
        if isinstance(rewards, th.Tensor):
            rewards_ = rewards.detach().cpu().numpy()
        else:
            rewards_ = rewards
        self.episode_returns += real_reward_
        self.disc_episode_returns += real_reward_ * np.repeat(self.gamma**self.episode_lengths, self.reward_dim).reshape(
                self.episode_returns.shape
            )
        self.disc_episode_returns_l += rewards_ * np.repeat(self.gamma**self.episode_lengths, self.reward_dim).reshape(
            self.episode_returns.shape
        )
        self.episode_returns_l += rewards_
        self.episode_lengths += 1

        if terminated or truncated:
            #assert self._stats_key not in info

            episode_time_length = round(time.perf_counter() - self.episode_start_time, 6)

            # Make a deepcopy to void subsequent mutation of the numpy array
            episode_returns = deepcopy(self.episode_returns)
            episode_returns_l = deepcopy(self.episode_returns_l)
            disc_episode_returns = deepcopy(self.disc_episode_returns)
            disc_episode_returns_l = deepcopy(self.disc_episode_returns_l)

            info["episode"] = {
                "r": episode_returns,
                "rl": episode_returns_l,
                "dr": disc_episode_returns,
                "drl": disc_episode_returns_l,
                "l": self.episode_lengths,
                "t": episode_time_length,
            }

            self.time_queue.append(episode_time_length)
            self.return_queue.append(episode_returns)
            self.length_queue.append(self.episode_lengths)

            self.episode_count += 1
            self.episode_start_time = time.perf_counter()

        return (
            observation,
            rewards,
            terminated,
            truncated,
            info,
        )

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)

        # CHANGE: Here we just override the standard implementation to extend to MO
        self.episode_returns = np.zeros(self.rewards_shape, dtype=np.float32)
        self.episode_returns_l = np.zeros(self.rewards_shape, dtype=np.float32)
        self.disc_episode_returns = np.zeros(self.rewards_shape, dtype=np.float32)
        self.disc_episode_returns_l = np.zeros(self.rewards_shape, dtype=np.float32)

        return obs, info
class EnvelopePBMORL(EnvelopeCustomReward):
    
    def __init__(self, env, buffer_with_weights, maintain_original_reward=False, learning_rate=0.0003, estimated_horizon=10, initial_epsilon=0.01, final_epsilon=0.01, epsilon_decay_steps=None, tau=1, target_net_update_freq=200, buffer_size=..., net_arch=..., batch_size=256, learning_starts=100, gradient_updates=1, gamma=0.99, max_grad_norm=1, envelope=True, num_sample_w=4, per=False, per_alpha=0.6, initial_homotopy_lambda=0, final_homotopy_lambda=1, homotopy_decay_steps=None, project_name="MORL-Baselines", experiment_name="Envelope", wandb_entity=None, log=True, seed=None, device="auto", activation=None, group=None, masked=False, relabel_buffer=True, reward_vector_function=None):
        super().__init__(env, buffer_with_weights, maintain_original_reward, learning_rate,  estimated_horizon, initial_epsilon, final_epsilon, epsilon_decay_steps, tau, target_net_update_freq, buffer_size, net_arch, batch_size, learning_starts, gradient_updates, gamma, max_grad_norm, envelope, num_sample_w, per, per_alpha, initial_homotopy_lambda, final_homotopy_lambda, homotopy_decay_steps, project_name, experiment_name, wandb_entity, log, seed, device, activation, group, masked, relabel_buffer, reward_vector_function)
        self.running_dataset = None
        self.base_dataset = None
        #assert not self.per  #!!!!

    def reward_training(self, Ns, Nw, H, gamma_preferences, reward_train_callback, max_reward_buffer_size, qualitative_preferences):
        assert isinstance(self.replay_buffer, CustomRewardReplayBuffer), "Replay buffer must be an instance of CustomRewardReplayBuffer to use reward training."
        assert self.replay_buffer.relabel_buffer, "Replay buffer must have relabel_buffer set to True to use reward training."
        if self.replay_buffer.maintain_original_reward:
            assert self.replay_buffer._original_rewards is not None, "Original rewards must be stored in the replay buffer to use reward training."

        
        t = time.time()
        self.overseer_new_preferences(Ns, Nw, H, gamma_preferences, max_reward_buffer_size, qualitative_preferences=qualitative_preferences)
        rt = time.time()   
        print(f"\033[33mOverseer new preferences took {rt-t:.4f} seconds\033[0m")
        assert th.is_grad_enabled()
        t = time.time()
        return_dict = reward_train_callback(base_dataset=self.base_dataset, running_dataset=self.running_dataset, reward_vector_function=self.replay_buffer.reward_vector_function)
        rt = time.time()
        print(f"\033[33mReward callback took {rt-t:.4f} seconds\033[0m")
        t = time.time()
        self.set_reward_vector_function(return_dict['reward_net'])
        return_dict['reward_net'].set_mode('eval')
        rt = time.time()
        print(f"\033[33mSet reward vector function took {rt-t:.4f} seconds\033[0m")
        return return_dict


    def overseer_new_preferences(self, Ns, Nw, H, gamma_preferences, max_reward_buffer_size, qualitative_preferences):
        with th.no_grad():
            fragments1, fragments2, real_v_rew1, real_v_rew2, lear_v_rew1, lear_v_rew2, dsreal_v_rew1, dsreal_v_rew2, dslear_v_rew1, dslear_v_rew2  = self.replay_buffer.sample_trajs(ns=Ns, gamma=gamma_preferences,get_rewards_orig=True,device=self.device, to_tensor=True, H=H)
            assert real_v_rew1.shape[0] == real_v_rew2.shape[0], f"Original rewards shape mismatch: {real_v_rew1.shape} vs {real_v_rew2.shape}"
            assert lear_v_rew1.shape[0] == lear_v_rew2.shape[0], f"Rewards shape mismatch: {lear_v_rew1.shape} vs {lear_v_rew2.shape}"
            assert len(fragments1[0]) == len(fragments2[0]), f"Fragments shape mismatch: {len(fragments1)} vs {len(fragments2[0])}"
            assert len(real_v_rew1) == len(fragments1[0]), f"Original rewards and rewards length mismatch: {len(real_v_rew1)} vs {len(fragments1[0])}"
            assert len(real_v_rew2) == len(fragments2[0]), f"Original rewards and rewards length mismatch: {len(real_v_rew2)} vs {len(fragments2[0])}"
            assert dsreal_v_rew1.shape == (Ns, self.reward_dim), f"Rewards shape mismatch: {dsreal_v_rew1.shape} vs {(Ns, self.reward_dim)}"
            assert dsreal_v_rew2.shape == (Ns, self.reward_dim), f"Rewards shape mismatch: {dsreal_v_rew2.shape} vs {(Ns, self.reward_dim)}"
            assert dslear_v_rew1.shape == (Ns, self.reward_dim), f"Rewards shape mismatch: {dslear_v_rew1.shape} vs {(Ns, self.reward_dim)}"
            assert dslear_v_rew2.shape == (Ns, self.reward_dim), f"Rewards shape mismatch: {dslear_v_rew2.shape} vs {(Ns, self.reward_dim)}"

            sampled_w = self.sample_new_weights(Nw)

            if max_reward_buffer_size < len(self.running_dataset) + Ns * Nw:
                    print("\033[95mPOPPPED",  len(self.running_dataset) - max_reward_buffer_size + Ns * Nw, "out of", len(self.running_dataset), "max", max_reward_buffer_size, "\033[0m")
                    self.running_dataset.pop(len(self.running_dataset) - max_reward_buffer_size + Ns * Nw)


            for w in sampled_w:
                wnp = w
                w = th.tensor(w).float().to(self.device)
                
                # dsreal_v_rew1 has shape [Ns, w.shape[0]]. apply dot over the last dimension
                dsrw1 = (dsreal_v_rew1 * w).sum(dim=1)
                dsrw2 = (dsreal_v_rew2 * w).sum(dim=1)
                assert dsrw1.shape == dsrw2.shape == (Ns,), f"Rewards shape mismatch: {dsrw1.shape} vs {dsrw2.shape} vs {(Ns,)}"
                preferences_w = probability_BT(dsrw1, dsrw2).detach()
                if qualitative_preferences:
                    preferences_w = probs_to_label(preferences_w)
                preferences_with_grounding = th.zeros((len(preferences_w), self.reward_dim)).to(self.device)
                """if not th.all((dsrw1 > dsrw2) == (preferences_w > 0.5)):
                    for i in range(len(dsrw1)):
                        if not ((dsrw1[i] > dsrw2[i]) == (preferences_w[i] > 0.5)):
                            print(f"dsrw1[{i}]: {dsrw1[i]}, dsrw2[{i}]: {dsrw2[i]}, preferences_w[{i}]: {preferences_w[i]}")
                assert th.all((dsrw1 >= dsrw2 + 0.0001) == (preferences_w >= 0.5)), "preferences_w should be 1 when dsrw1 > dsrw2"
                if not th.all((dsrw1 < dsrw2) == (preferences_w < 0.5)):
                    for i in range(len(dsrw1)):
                        if not ((dsrw1[i] < dsrw2[i]) == (preferences_w[i] < 0.5)):
                            print(f"dsrw1[{i}]: {dsrw1[i]}, dsrw2[{i}]: {dsrw2[i]}, preferences_w[{i}]: {preferences_w[i]}")
                assert th.all((dsrw1 <= dsrw2-0.0001) == (preferences_w <= 0.5)), "preferences_w should be 0 when dsrw1 < dsrw2"
                """
                for vi in range(self.reward_dim):
                    assert dsreal_v_rew1.shape == dsreal_v_rew2.shape == (Ns,self.reward_dim), f"Rewards shape mismatch: {dsreal_v_rew1.shape} vs {dsreal_v_rew2.shape} vs {(Ns,)}"
                    preferences_with_grounding[:, vi] =  probability_BT(dsreal_v_rew1[:,vi], dsreal_v_rew2[:,vi]).detach()
                    if qualitative_preferences:
                        preferences_with_grounding[:, vi] = probs_to_label(preferences_with_grounding[:, vi])
                    #preferences_learned_w = probability_BT(lear_rew1, lear_rew2)
                fpairs_w = []
                aname = '_synth_' + str(transform_weights_to_tuple(w))
                for i in range(len(fragments1[0])):
                    s1, a1, n1, d1 = (fragments1[stand][i] for stand in range(len(fragments1)))
                    s2, a2, n2, d2 = (fragments2[stand][i] for stand in range(len(fragments2)))
                    #print(len(s1), len(a1), len(n1), len(d1), "FRAGMENTS", i, "OF", len(fragments1[0]), type(s1))
                    """if H is None:
                        assert np.allclose(s1[1:], n1[0:-1]), f"Fragment mismatch: {np.where(s1[1:] != n1[0:-1])} in {s1.shape} vs {n1.shape}"
                    """
                    fobs1 = np.concatenate([s1, n1[-1][None]], axis=0)
                    fobs2 = np.concatenate([s2, n2[-1][None]], axis=0)
                    assert fobs1.shape == (s1.shape[0] + 1, *s1.shape[1:]), f"Fragment obs shape mismatch: {fobs1.shape} vs {s1.shape[0] + 1}"
                    f1 = TrajectoryWithValueSystemRews(obs=fobs1, 
                                                    acts=a1,
                                                    dones=d1, 
                                                    terminal=d1[-1],
                                                    v_rews=lear_v_rew1[i].T, 
                                                    rews=th.sum(lear_v_rew1[i] * w, dim=1),v_rews_real=real_v_rew1[i].T, 
                                                    rews_real=th.sum(real_v_rew1[i] * w, dim=1),n_vals=self.reward_dim, infos=[{'_envelope_weights': wnp} for _ in range(len(a1))], 
                                                    agent=aname)
                    
                    f2 = TrajectoryWithValueSystemRews(obs=fobs2, 
                                                    acts=a2,
                                                    dones=d2, 
                                                    terminal=d2[-1],
                                                    v_rews=lear_v_rew2[i].T, 
                                                    rews=th.sum(lear_v_rew2[i] * w, dim=1),v_rews_real=real_v_rew2[i].T, 
                                                    rews_real=th.sum(real_v_rew2[i] * w, dim=1),n_vals=self.reward_dim, infos=[{'_envelope_weights': wnp} for _ in range(len(a2))], 
                                                    agent=aname)
                    """if len(fobs1) == len(fobs2):
                        assert not (np.allclose(fobs2, fobs1) and np.allclose(a2, a1)), f"Sampled identical fragments.{a1, a2}"
                    """
                    """valid_values = np.array([-1.0, -0.1, -0.5, 0.0, 0.4, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                    if not np.all(np.isclose(real_v_rew1[..., None], valid_values, atol=1e-6).any(axis=-1)):
                        raise AssertionError(f"Invalid original_rew: {real_v_rew1}")
                    valid_values = np.array([-1.0, -0.1, -0.5, 0.0, 0.4, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                    if not np.all(np.isclose(real_v_rew2[..., None], valid_values, atol=1e-6).any(axis=-1)):
                        raise AssertionError(f"Invalid original_rew: {real_v_rew2}")
                    valid_values = np.array([-1.0, -0.1, -0.5, 0.0, 0.4, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                    if not np.all(np.isclose(f1.v_rews_real[..., None], valid_values, atol=1e-6).any(axis=-1)):
                        raise AssertionError(f"Invalid original_rew: {f2.rews_real}")
                    valid_values = np.array([-1.0, -0.1, -0.5, 0.0, 0.4, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                    if np.all(np.isclose(f1.v_rews[..., None], valid_values, atol=1e-6).any(axis=-1)):
                        raise AssertionError(f"Invalid original_rew: {f1.value_rews}")"""
                    #print("DIFF", th.norm(f1.rews_real-f1.value_rews_real))
                    #input()
                    fpairs_w.append((f1, f2))
                adata = {
                            "trajectory_pairs": len(fpairs_w),
                            "rationality": self.epsilon, "random_traj_proportion": 0.0
                        }
                t = time.time()
                self.running_dataset.push(np.asarray(fpairs_w, dtype=tuple), preferences=preferences_w, preferences_with_grounding=preferences_with_grounding, agent_data={aname: {'value_system': w, 'name': aname, 'data': adata, "n_agents": 1}}, agent_ids=[aname for _ in range(len(fpairs_w))])
                rt = time.time()
                print(f"\033[35mPush to running dataset took {rt-t:.4f} seconds\033[0m")

            #self.running_dataset.transition_mode(self.device)
                
    def sample_eval_weights(self, n):
        return equally_spaced_weights(self.reward_dim, n=n)

    def train(self, total_timesteps, Ns, Nw, H, K, gamma_preferences, reward_train_callback, dataset: VSLPreferenceDataset=None, eval_env = None, ref_point = None, known_pareto_front = None, weight = None, total_episodes = None, reset_num_timesteps = True, eval_freq = 10000, num_eval_weights_for_front = 100, num_eval_episodes_for_front = 5, num_eval_weights_for_eval = 50, qualitative_preferences=True, reset_learning_starts = False, verbose = False, max_reward_buffer_size=None):
        self.base_dataset = dataset
        self.running_dataset = FixedLengthVSLPreferenceDataset(self.reward_dim, single_agent=False, size= max_reward_buffer_size) if self.running_dataset is None else self.running_dataset
       
        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes
        if reset_learning_starts:  # Resets epsilon-greedy exploration
            self.learning_starts = self.global_step
        
        num_episodes = 0
        eval_weights = self.sample_eval_weights(num_eval_weights_for_front)
        obs, info = self.env.reset()

        w = weight if weight is not None else self.sample_new_weights(1, random=True)
        tensor_w = th.tensor(w).float().to(self.device)
        
        for _ in range(1, total_timesteps + 1):
            
            action_mask = info.get('action_masks', None)
            if self.masked:
                assert action_mask is not None
                action_mask = th.as_tensor(action_mask).float().to(self.device)
            if self.verbose:
                pass
                #print(f"Global step: {self.global_step}, Episode: {num_episodes}, Epsilon: {self.epsilon:.3f}, Homotopy Lambda: {self.homotopy_lambda:.3f}", flush=True)
            if total_episodes is not None and num_episodes == total_episodes:
                break

            if self.global_step < self.learning_starts:
                if not self.masked:
                    action = self.env.action_space.sample()
                else:
                    action =int(np.random.choice(np.where(action_mask > 0.0)[0]))
                
            else:
                action = self.act(th.as_tensor(obs).float().to(self.device), tensor_w)

            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)
            self.global_step += 1
            
            self.replay_buffer.add(obs, action, vec_reward, next_obs, terminated or truncated, original_rew=info['untransformed_reward'], weights=transform_weights_to_tuple(w))
            
            if self.global_step % K == 0:
                self.reward_training(Ns, Nw, H, gamma_preferences, reward_train_callback, max_reward_buffer_size, qualitative_preferences) # ONly change this.
                #input()
            if self.global_step >= self.learning_starts:
                self.update()
            

            if eval_env is not None and self.log and self.global_step % eval_freq == 0:
                eval_weights = self.sample_eval_weights(num_eval_weights_for_front)
                current_front = [
                    self.policy_eval(eval_env, weights=ew, num_episodes=num_eval_episodes_for_front, log=self.log)[3]
                    for ew in eval_weights
                ]
                log_all_multi_policy_metrics(
                    current_front=current_front,
                    hv_ref_point=ref_point,
                    reward_dim=self.reward_dim,
                    global_step=self.global_step,
                    n_sample_weights=num_eval_weights_for_eval,
                    ref_front=known_pareto_front,
                )
            if terminated or truncated:
                obs, _ = self.env.reset()
                num_episodes += 1
                self.num_episodes += 1

                if self.log and "episode" in info.keys():
                    log_episode_info(info["episode"], np.dot, w, self.global_step, verbose=self.verbose)

                if weight is None:
                    w = self.sample_new_weights(1, random=True)
                    tensor_w = th.tensor(w).float().to(self.device)

            else:
                obs = next_obs
        #return super().train(total_timesteps, eval_env, ref_point, known_pareto_front, weight, total_episodes, reset_num_timesteps, eval_freq, num_eval_weights_for_front, num_eval_episodes_for_front, num_eval_weights_for_eval, reset_learning_starts, verbose)
    def sample_new_weights(self, n, random=False):
        return random_weights(self.reward_dim, n, dist="gaussian", rng=self.np_random)

    def __sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)
        #return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)

    @override
    def update(self):
        critic_losses = []
        for g in range(self.gradient_updates):
            if not self.replay_buffer.buffer_with_weights:
                if self.per:
                        (
                            b_obs,
                            b_actions,
                            b_rewards,
                            b_next_obs,
                            b_dones,
                            b_inds,
                        ) = self.__sample_batch_experiences()
                else:
                    (
                        b_obs,
                        b_actions,
                        b_rewards,
                        b_next_obs,
                        b_dones,
                    ) = self.__sample_batch_experiences()

            
                sampled_w = (
                    th.tensor(self.sample_new_weights(self.num_sample_w))
                    .float()
                    .to(self.device)
                )  # sample num_sample_w random weights
                w = sampled_w.repeat_interleave(b_obs.size(0), 0)  # repeat the weights for each sample
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = (
                    b_obs.repeat(self.num_sample_w, *(1 for _ in range(b_obs.dim() - 1))),
                    b_actions.repeat(self.num_sample_w, 1),
                    b_rewards.repeat(self.num_sample_w, 1),
                    b_next_obs.repeat(self.num_sample_w, *(1 for _ in range(b_next_obs.dim() - 1))),
                    b_dones.repeat(self.num_sample_w, 1),
                )
            else:
                if self.per:
                    (
                            b_obs,
                            b_actions,
                            b_rewards,
                            b_next_obs,
                            b_dones,
                            w,
                            b_inds,
                        ) = self.__sample_batch_experiences()
                else:
                    (
                            b_obs,
                            b_actions,
                            b_rewards,
                            b_next_obs,
                            b_dones,
                            w,
                        ) = self.__sample_batch_experiences()
                w= th.tensor(w).float().to(self.device)
                #sampled_w = w  # Just for logging purposes
                sampled_w = torch.unique(w, dim=0)
            #print(b_obs.shape, b_actions.shape, b_rewards.shape, b_next_obs.shape, b_dones.shape, w.shape, "BATCH SHAPES")
            #print("NEW", b_obs_new.shape, b_actions_new.shape, b_rewards_new.shape, b_next_obs_new.shape, b_dones_new.shape, b_weights.shape, "BATCH SHAPES")
            #input()
            b_obs = b_obs.float()
            b_next_obs = b_next_obs.float()

            with th.no_grad():
                if self.envelope:
                    target = self.envelope_target(b_next_obs, w, sampled_w)
                else:
                    target = self.ddqn_target(b_next_obs, w)
                target_q = b_rewards + (1 - b_dones) * self.gamma * target

            q_values = self.q_net(b_obs, w)
            q_value = q_values.gather(
                1,
                b_actions.long().reshape(-1, 1, 1).expand(q_values.size(0), 1, q_values.size(2)),
            )
            q_value = q_value.reshape(-1, self.reward_dim)
            critic_loss = torch.nn.functional.mse_loss(q_value, target_q)

            if self.homotopy_lambda > 0:
                wQ = th.einsum("br,br->b", q_value, w)
                wTQ = th.einsum("br,br->b", target_q, w)
                auxiliary_loss = torch.nn.functional.mse_loss(wQ, wTQ)
                critic_loss = (1 - self.homotopy_lambda) * critic_loss + self.homotopy_lambda * auxiliary_loss

            self.q_optim.zero_grad()
            critic_loss.backward()
            if self.log and self.global_step % 100 == 0:
                pprint.pprint({
                    "losses/grad_norm": get_grad_norm(self.q_net.parameters()).item(),
                    "global_step": self.global_step,
                })
                wandb.log(
                    {
                        "losses/grad_norm": get_grad_norm(self.q_net.parameters()).item(),
                        "global_step": self.global_step,
                    },
                )
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.q_optim.step()
            critic_losses.append(critic_loss.item())

            if self.per:
                td_err = (q_value[: len(b_inds)] - target_q[: len(b_inds)]).detach()
                priority = th.einsum("sr,sr->s", td_err, w[: len(b_inds)]).abs()
                priority = priority.cpu().numpy().flatten()
                priority = (priority + self.replay_buffer.min_priority) ** self.per_alpha
                self.replay_buffer.update_priorities(b_inds, priority)

        if self.tau != 1 or self.global_step % self.target_net_update_freq == 0:
            polyak_update(self.q_net.parameters(), self.target_q_net.parameters(), self.tau)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                self.learning_starts,
                self.final_epsilon,
            )
            

        if self.homotopy_decay_steps is not None:
            self.homotopy_lambda = linearly_decaying_value(
                self.initial_homotopy_lambda,
                self.homotopy_decay_steps,
                self.global_step,
                self.learning_starts,
                self.final_homotopy_lambda,
            )

        if self.log and self.global_step % 100 == 0:
            if self.verbose:
                pprint.pprint({
                    "losses/critic_loss": np.mean(critic_losses),
                    "metrics/epsilon": self.epsilon,
                    "metrics/homotopy_lambda": self.homotopy_lambda,
                    "global_step": self.global_step,
                })
            wandb.log(
                {
                    "losses/critic_loss": np.mean(critic_losses),
                    "metrics/epsilon": self.epsilon,
                    "metrics/homotopy_lambda": self.homotopy_lambda,
                    "global_step": self.global_step,
                },
            )
            if self.per:
                wandb.log({"metrics/mean_priority": np.mean(priority)})
from stable_baselines3.common.buffers import RolloutBuffer
class RolloutBufferCustomReward(RolloutBuffer):

    def __init__(
        self,
        *args,
        relabel_experience=True,
        weights=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.relabel_experience = relabel_experience
        self.weights = weights

    def set_weights(self, weights):
        self.weights = np.array(weights)
    def set_reward_vector_function(self, reward_vector_function: RewardVectorModule):
        if self.relabel_experience:
            self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
            self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
            self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        assert not reward_vector_function.feature_extractor.use_next_state 
        assert not reward_vector_function.feature_extractor.use_done 
        self.rewards = np.asarray([reward_vector_function.forward(self.observations.reshape((self.buffer_size, self.observations.shape[-1])), self.actions.flatten(), None, None).detach().cpu().numpy().dot(self.weights) for _ in range(self.n_envs)]).T
        assert self.rewards.shape == self.returns.shape, f"Reward shape mismatch: {self.rewards.shape} vs {self.observations.shape}, {self.actions.shape}"

ROLLOUT_BUFFER_CLASSES = {
    "RolloutBufferCustomReward": RolloutBufferCustomReward
}
