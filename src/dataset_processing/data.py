from copy import deepcopy
import dataclasses
from functools import partial
import random
import time
import dill
from typing import List, Self, Sequence, Tuple, TypeVar, overload

import logging
import os
import warnings
from typing import Mapping, Sequence, cast

import datasets
import jsonpickle
import numpy as np

from imitation.data import huggingface_utils
from imitation.data.types import AnyPath, Trajectory, TrajectoryWithRew
from imitation.util import util

import torch
def _vs_traj_validation(vs_rews: np.ndarray | torch.Tensor, value_rews: np.ndarray | torch.Tensor, n_values: int, acts: np.ndarray | torch.Tensor, dones: np.ndarray | torch.Tensor=None, optional=False):
    if optional and vs_rews is None and value_rews is None:
        return
    if vs_rews.shape != (len(acts),):
        raise ValueError(
            "Value system rewards must be 1D array, one entry for each action: "
            f"{vs_rews.shape} != ({len(acts)},)",
        )
    if not isinstance(vs_rews.dtype, np.dtype) and not isinstance(vs_rews.dtype, torch.dtype):
        raise ValueError(f"rewards dtype {vs_rews.dtype} not a float")
    
    if value_rews.shape != (n_values, len(acts)):
        raise ValueError(
            "Individual value rewards must each be 1D array, one entry for each action: "
            f"{value_rews.shape} != ({n_values}, {len(acts)})",
        )
    if not isinstance(value_rews.dtype, np.dtype) and not isinstance(value_rews.dtype, torch.dtype) :
        raise ValueError(f"rewards dtype {value_rews.dtype} not a float")
    if dones is not None:
        if dones.shape != (len(acts),):
            raise ValueError(
                "Dones must each be 1D array, one entry for each action: "
                f"{dones.shape} != ({n_values}, {len(acts)})",
            )
        if not isinstance(dones.dtype, np.dtype) and not isinstance(dones.dtype, torch.dtype) :
            raise ValueError(f"rewards dtype {dones.dtype} not a float")

    
@dataclasses.dataclass(frozen=True, eq=False)
class TrajectoryWithValueSystemRews(Trajectory):
    """A `Trajectory` that additionally includes reward information of the value system rewards (for the value system alignment and each individual value alignment)."""

    
    """Reward, shape (trajectory_len, ). dtype float."""
    n_vals: int
    v_rews: np.ndarray
    rews: np.ndarray
    v_rews_real: np.ndarray
    rews_real: np.ndarray
    dones: np.ndarray
    agent: str
    @property
    def vs_rews(self):
        return self.rews
    @property
    def value_rews(self):
        return self.v_rews
    
    @property
    def vs_rews_real(self):
        return self.rews_real
    @property
    def value_rews_real(self):
        return self.v_rews_real
        
    def __post_init__(self):
        """Performs input validation, including for rews."""
        super().__post_init__()
        _vs_traj_validation(self.vs_rews, self.value_rews, self.n_vals, self.acts)
        _vs_traj_validation(self.vs_rews_real, self.value_rews_real, self.n_vals, self.acts, optional=True)

class TrajectoryValueSystemDatasetSequence(huggingface_utils.TrajectoryDatasetSequence):
    """A wrapper to present an HF dataset as a sequence of trajectories.

    Converts the dataset to a sequence of trajectories on the fly.
    """

    def __init__(self, dataset: datasets.Dataset, dtype=np.float32):
        super().__init__(dataset)
        self._trajectory_class = TrajectoryWithValueSystemRews if 'v_rews' in dataset.features else self._trajectory_class

from typing import Any, Dict, Iterable, Optional, Sequence, cast

import datasets

import numpy as np

from imitation.data import types


T = TypeVar("T")
Pair = Tuple[T, T]
TrajectoryWithValueSystemRewsPair = Pair[TrajectoryWithValueSystemRews]

def vs_trajectories_to_dict(trajectories, use_infos=False, dtype=np.float32):
    has_reward = [isinstance(traj, TrajectoryWithValueSystemRews) for traj in trajectories]
    all_trajectories_have_reward = all(has_reward)
    if not all_trajectories_have_reward and any(has_reward):
        raise ValueError("Some trajectories have VS structure but not all")

    # Convert to dict
    trajectory_dict: Dict[str, Sequence[Any]] = dict(
        obs=[traj.obs for traj in trajectories],
        acts=[traj.acts for traj in trajectories],
        # Replace 'None' values for `infos`` with array of empty dicts
        infos=[
            traj.infos if traj.infos is not None and use_infos else [{}] * len(traj)
            for traj in trajectories
        ] ,
        terminal=[traj.terminal for traj in trajectories],
    )
    if any(isinstance(traj.obs, types.DictObs) for traj in trajectories):
        raise ValueError("DictObs are not currently supported")

    # Encode infos as jsondilld strings
    trajectory_dict["infos"] = [
        [jsonpickle.encode(info) for info in traj_infos]
        for traj_infos in cast(Iterable[Iterable[Dict]], trajectory_dict["infos"])
    ]

    # Add rewards if applicable
    if all_trajectories_have_reward:
        trajectory_dict["rews"] = [
            np.asarray(cast(TrajectoryWithValueSystemRews, traj).rews, dtype) for traj in trajectories
        ]
        trajectory_dict["dones"] = [
            np.asarray(cast(TrajectoryWithValueSystemRews, traj).dones, dtype) for traj in trajectories
        ]

        trajectory_dict["v_rews"] = [
            np.asarray(cast(TrajectoryWithValueSystemRews, traj).v_rews, dtype) for traj in trajectories
        ]
        trajectory_dict["rews_real"] = [
            np.asarray(cast(TrajectoryWithValueSystemRews, traj).rews_real, dtype) if traj.rews_real is not None else None for traj in trajectories
        ]
        trajectory_dict["v_rews_real"] = [
            np.asarray(cast(TrajectoryWithValueSystemRews, traj).v_rews_real, dtype) if traj.v_rews_real is not None else None for traj in trajectories
        ]
        trajectory_dict["n_vals"] = [
            cast(TrajectoryWithValueSystemRews, traj).n_vals for traj in trajectories
        ]
        trajectory_dict["agent"] = [
            cast(TrajectoryWithValueSystemRews, traj).agent for traj in trajectories
        ]

    
    return trajectory_dict
    

def vs_trajectories_to_dataset(
    trajectories: Sequence[types.Trajectory],
    info: Optional[datasets.DatasetInfo] = None,
    dtype = np.float32,
    use_infos= False,
) -> datasets.Dataset:
    """Convert a sequence of trajectories to a HuggingFace dataset."""
    if isinstance(trajectories, TrajectoryValueSystemDatasetSequence):
        return trajectories.dataset
    else:
        dataset= datasets.Dataset.from_dict(vs_trajectories_to_dict(trajectories, dtype=dtype, use_infos=use_infos), info=info)
        return dataset
    
def save_vs_trajectories(path: AnyPath, trajectories: Sequence[TrajectoryWithValueSystemRews], dtype=np.float32, use_infos=False) -> None:
    """Save a sequence of Trajectories to disk using HuggingFace's datasets library.

    Args:
        path: Trajectories are saved to this path.
        trajectories: The trajectories to save.
    """
    p = util.parse_path(path)
    vs_trajectories_to_dataset(trajectories, dtype=dtype, use_infos=use_infos).save_to_disk(str(p))
    logging.info(f"Dumped demonstrations to {p}.")



def load_vs_trajectories(path: AnyPath) -> Sequence[Trajectory]:
    """Loads a sequence of trajectories saved by `save()` from `path`."""
    # Interestingly, np.load will just silently load a normal dill file when you
    # set `allow_dill=True`. So this call should succeed for both the new compressed
    # .npz format and the old dill based format. To tell the difference, we need to
    # look at the type of the resulting object. If it's the new compressed format,
    # it should be a Mapping that we need to decode, whereas if it's the old format,
    # it's just the sequence of trajectories, and we can return it directly.
    
    if os.path.isdir(path):  # huggingface datasets format
        dataset = datasets.load_from_disk(str(path))
        if not isinstance(dataset, datasets.Dataset):  # pragma: no cover
            raise ValueError(
                f"Expected to load a `datasets.Dataset` but got {type(dataset)}",
            )

        return TrajectoryValueSystemDatasetSequence(dataset)
    raise NotImplementedError("Only huggingface datasets format is supported")

    data = np.load(path, allow_dill=True)  # works for both .npz and .pkl

    if isinstance(data, Sequence):  # dill format
        warnings.warn("Loading old dill version of Trajectories", DeprecationWarning)
        return data
    if isinstance(data, Mapping):  # .npz format
        warnings.warn("Loading old npz version of Trajectories", DeprecationWarning)
        num_trajs = len(data["indices"])
        fields = [
            # Account for the extra obs in each trajectory
            np.split(data["obs"], data["indices"] + np.arange(num_trajs) + 1),
            np.split(data["acts"], data["indices"]),
            np.split(data["infos"], data["indices"]),
            data["terminal"],
        ]
        if 'vs_rews' in data:
            fields = [
                *fields,
                np.split(data["vs_rews"], data["indices"]),
            ]
            for k in data["value_rews"].keys():
                fields = [
                    *fields,
                    np.split(data["value_rews"][k], data["indices"]),
                ]
            return [TrajectoryWithRew(*args) for args in zip(*fields)]
        elif "rews" in data:
            fields = [
                *fields,
                np.split(data["rews"], data["indices"]),
            ]
            return [TrajectoryWithRew(*args) for args in zip(*fields)]
        else:
            return [Trajectory(*args) for args in zip(*fields)]  # pragma: no cover
    else:  # pragma: no cover
        raise ValueError(
            f"Expected either an .npz file or a dilld sequence of trajectories; "
            f"got a dilld object of type {type(data).__name__}",
        )
    



from imitation.algorithms import preference_comparisons
from sklearn.model_selection import KFold

class VSLPreferenceDataset(preference_comparisons.PreferenceDataset):
    """A PyTorch Dataset for preference comparisons.

    Each item is a tuple consisting of two trajectory fragments
    and a probability that fragment 1 is preferred over fragment 2.

    This dataset is meant to be generated piece by piece during the
    training process, which is why data can be added via the .push()
    method.
    """
    @property
    def fragments_best(self):
        idxs1 = self.preferences > 0.5
        idxs2 = self.preferences < 0.5
        ret = np.concatenate([self.fragments1[idxs1], self.fragments2[idxs2]], axis=0)
        assert np.all(np.array([sum(f.rews) for f in self.fragments1[idxs1]]) >= np.array([sum(f.rews) for f in self.fragments2[idxs1]]))
        # Check that for idxs2, sum(frag1.rews) < sum(frag2.rews)
        frag1_sums = np.array([sum(f.rews) for f in self.fragments1[idxs2]])
        frag2_sums = np.array([sum(f.rews) for f in self.fragments2[idxs2]])
        """try:
            np.testing.assert_array_less(frag1_sums, frag2_sums, verbose=True)
        except AssertionError as e:
            # Print exactly which values do not hold
            failed = np.where(frag1_sums >= frag2_sums)[0]
            for i in failed:
                print(f"Failed at index {i}: sum(frag1.rews)={frag1_sums[i]}, sum(frag2.rews)={frag2_sums[i]}")
            raise"""
        
        #input()
        assert ret.shape[0] == idxs1.sum() + idxs2.sum()
        return ret

    def __init__(self, n_values: int, single_agent=False) -> None:
        """Builds an empty PreferenceDataset for Value System Learning
        """
        self.l_fragments1: List[TrajectoryWithValueSystemRews] = []
        self.l_fragments2: List[TrajectoryWithValueSystemRews] = []
        self.preferences: np.ndarray = np.array([], dtype=np.float32)
        self.list_preferences_with_grounding: list = [np.array([], dtype=np.float32)] * n_values
        self.n_values = n_values
        self.l_agent_ids = []
        self.agent_data = {}
        self.fidxs_per_agent = {}
        if not single_agent:
            self.data_per_agent = {}
    @property
    def preferences_with_grounding(self):
        return np.asarray(self.list_preferences_with_grounding, dtype=np.float32).T

    @property
    def agent_ids(self):
        return np.asarray(self.l_agent_ids)
    
    @property
    def n_agents(self):
        return len(set(self.l_agent_ids))
   
    @property
    def fragments1(self):
        return np.asarray(self.l_fragments1)
    @property
    def fragments2(self):
        return np.asarray(self.l_fragments2)
    @property
    def states1(self):
        return self._states1
    @property
    def states2(self):
        return self._states2
    @property
    def acts1(self):
        return self._acts1
    @property
    def acts2(self):
        return self._acts2
    @property
    def rews1(self):
        return self._rews1
    @property
    def rews2(self):
        return self._rews2
    @property
    def dones1(self):
        return self._dones1
    @property
    def dones2(self):
        return self._dones2
    @property
    def value_rews1(self):
        return self._value_rews1
    @property
    def value_rews2(self):
        return self._value_rews2
    @property
    def vs_rews1(self):
        return self._vs_rews1
    @property
    def vs_rews2(self):
        return self._vs_rews2
    
    @property
    def next_states1(self):
        return self._next_state1

    @property
    def next_states2(self):
        return self._next_state2

    def transition_mode(self, device):
        with torch.no_grad():
            self_fragments1 = self.fragments1
            self_fragments2 = self.fragments2
            self._frag_idxs1 = [0]*(len(self_fragments1)+1)
            for i, frag in enumerate(self_fragments1):
                self._frag_idxs1[i+1] = len(frag) + self._frag_idxs1[i]
            self._frag_idxs2 = [0]*(len(self_fragments2)+1)
            for i, frag in enumerate(self_fragments2):
                self._frag_idxs2[i+1] = len(frag) + self._frag_idxs2[i]
            self._frag_idxs1 = np.array(self._frag_idxs1)
            self._frag_idxs2 = np.array(self._frag_idxs2)
            # Check if all fragments have the same shape for obs[:-1]
            obs_shapes = [frag.obs[:-1].shape for frag in self_fragments1]
            if len(set(obs_shapes)) > 1:
               
                self._states1 = np.asarray([util.safe_to_tensor(frag.obs[:-1], device=device, dtype=torch.float32) for frag in self_fragments1], dtype=torch.Tensor)
                self._states2 = np.asarray([util.safe_to_tensor(frag.obs[:-1], device=device, dtype=torch.float32) for frag in self_fragments2], dtype=torch.Tensor)
                self._acts1 = np.asarray([util.safe_to_tensor(frag.acts, device=device, dtype=torch.float32) for frag in self_fragments1], dtype=torch.Tensor)
                self._acts2 = np.asarray([util.safe_to_tensor(frag.acts, device=device, dtype=torch.float32) for frag in self_fragments2], dtype=torch.Tensor)
                self._next_state1 = np.asarray([util.safe_to_tensor(frag.obs[1:], device=device, dtype=torch.float32) for frag in self_fragments1], dtype=torch.Tensor)
                self._next_state2 = np.asarray([util.safe_to_tensor(frag.obs[1:], device=device, dtype=torch.float32) for frag in self_fragments2], dtype=torch.Tensor)

                self._rews1 = np.asarray([util.safe_to_tensor(frag.rews, device=device, dtype=torch.float32) for frag in self_fragments1], dtype=torch.Tensor)
                self._rews2 = np.asarray([util.safe_to_tensor(frag.rews, device=device, dtype=torch.float32) for frag in self_fragments2], dtype=torch.Tensor)
                self._dones1 = np.asarray([util.safe_to_tensor(frag.dones, device=device, dtype=torch.float32) for frag in self_fragments1], dtype=torch.Tensor)
                self._dones2 = np.asarray([util.safe_to_tensor(frag.dones, device=device, dtype=torch.float32) for frag in self_fragments2], dtype=torch.Tensor)
                self._value_rews1 = np.asarray([util.safe_to_tensor(frag.value_rews.T, device=device, dtype=torch.float32) for frag in self_fragments1], dtype=torch.Tensor)
                self._value_rews2 = np.asarray([util.safe_to_tensor(frag.value_rews.T, device=device, dtype=torch.float32) for frag in self_fragments2], dtype=torch.Tensor)
                self._vs_rews1 = np.asarray([util.safe_to_tensor(frag.vs_rews, device=device, dtype=torch.float32) for frag in self_fragments1], dtype=torch.Tensor)
                self._vs_rews2 = np.asarray([util.safe_to_tensor(frag.vs_rews, device=device, dtype=torch.float32) for frag in self_fragments2], dtype=torch.Tensor)
            else:
                self._states1 = util.safe_to_tensor([frag.obs[:-1] for frag in self_fragments1], device=device, dtype=torch.float32)
                self._states2 = util.safe_to_tensor([frag.obs[:-1] for frag in self_fragments2], device=device, dtype=torch.float32)
                self._acts1 = util.safe_to_tensor([frag.acts for frag in self_fragments1], device=device, dtype=torch.float32)
                self._acts2 = util.safe_to_tensor([frag.acts for frag in self_fragments2], device=device, dtype=torch.float32)
                self._next_state1 = util.safe_to_tensor([frag.obs[1:] for frag in self_fragments1], device=device, dtype=torch.float32)
                self._next_state2 = util.safe_to_tensor([frag.obs[1:] for frag in self_fragments2], device=device, dtype=torch.float32)
                self._dones1 = util.safe_to_tensor([frag.dones for frag in self_fragments1]).to(device=device, dtype=torch.float32)
                self._dones2 = util.safe_to_tensor([frag.dones for frag in self_fragments2]).to(device=device, dtype=torch.float32)
                

                self._rews1 = torch.stack([frag.rews for frag in self_fragments1]).to(device=device, dtype=torch.float32)
                self._rews2 = torch.stack([frag.rews for frag in self_fragments2]).to(device=device, dtype=torch.float32)
                self._value_rews1 = torch.stack([frag.value_rews.T for frag in self_fragments1]).to(device=device, dtype=torch.float32)
                self._value_rews2 = torch.stack([frag.value_rews.T for frag in self_fragments2]).to(device=device, dtype=torch.float32)
                self._vs_rews1 = torch.stack([frag.vs_rews for frag in self_fragments1]).to(device=device, dtype=torch.float32)
                self._vs_rews2 = torch.stack([frag.vs_rews for frag in self_fragments2]).to(device=device, dtype=torch.float32)

            # Flattened versions of the fragment data
            self._fstates1 = torch.cat([util.safe_to_tensor(frag.obs[:-1], device=device, dtype=torch.float32) for frag in self_fragments1], dim=0)
            self._fstates2 = torch.cat([util.safe_to_tensor(frag.obs[:-1], device=device, dtype=torch.float32) for frag in self_fragments2], dim=0)
            self._facts1 = torch.cat([util.safe_to_tensor(frag.acts, device=device, dtype=torch.float32) for frag in self_fragments1], dim=0)
            self._facts2 = torch.cat([util.safe_to_tensor(frag.acts, device=device, dtype=torch.float32) for frag in self_fragments2], dim=0)
            self._fnext_state1 = torch.cat([util.safe_to_tensor(frag.obs[1:], device=device, dtype=torch.float32) for frag in self_fragments1], dim=0)
            self._fnext_state2 = torch.cat([util.safe_to_tensor(frag.obs[1:], device=device, dtype=torch.float32) for frag in self_fragments2], dim=0)

            self._frews1 = torch.cat([util.safe_to_tensor(frag.rews, device=device, dtype=torch.float32) for frag in self_fragments1], dim=0)
            self._frews2 = torch.cat([util.safe_to_tensor(frag.rews, device=device, dtype=torch.float32) for frag in self_fragments2], dim=0)
            self._fdones1 = torch.cat([util.safe_to_tensor(frag.dones, device=device, dtype=torch.float32) for frag in self_fragments1], dim=0)
            self._fdones2 = torch.cat([util.safe_to_tensor(frag.dones, device=device, dtype=torch.float32) for frag in self_fragments2], dim=0)
            self._fvalue_rews1 = torch.cat([util.safe_to_tensor(frag.value_rews, device=device, dtype=torch.float32) for frag in self_fragments1], dim=1)
            self._fvalue_rews2 = torch.cat([util.safe_to_tensor(frag.value_rews, device=device, dtype=torch.float32) for frag in self_fragments2], dim=1)
            self._fvs_rews1 = torch.cat([util.safe_to_tensor(frag.vs_rews, device=device, dtype=torch.float32) for frag in self_fragments1], dim=0)
            self._fvs_rews2 = torch.cat([util.safe_to_tensor(frag.vs_rews, device=device, dtype=torch.float32) for frag in self_fragments2], dim=0)

            # Optionally, validate that the flattened slices match the original lists
            for frag_list, flat, name in [
                (self._states1, self._fstates1, "states1"),
                (self._acts1, self._facts1, "acts1"),
                (self._dones1, self._fdones1, "dones1"),
                (self._next_state1, self._fnext_state1, "next_state1"),
            ]:
                offset = 0
                for frag in frag_list:
                    frag_len = frag.shape[0]
                    assert torch.allclose(flat[offset:offset+frag_len], frag), f"Mismatch in {name}"
                    offset += frag_len
    
    def certify_data_consistency(self):
        #print(set([pi for p in self.fidxs_per_agent.values() for pi in p]))
        #print(set(np.arange(len(self)).tolist()))
        #assert set([pi for p in self.fidxs_per_agent.values() for pi in p]) == set(np.arange(len(self)).tolist()), "Data inconsistency: fidxs_per_agent does not cover all fragments"
        
        for ag in self.fidxs_per_agent.keys():
            fidxs_ag = self.fidxs_per_agent[ag]
            if hasattr(self, 'data_per_agent'):
                assert len(fidxs_ag) == len(self.data_per_agent[ag]), f"Data inconsistency for agent {ag}"
                assert all(self.fragments1[fidxs_ag] == self.data_per_agent[ag].fragments1), f"Data inconsistency for agent {ag}"
                assert all(self.fragments2[fidxs_ag] == self.data_per_agent[ag].fragments2), f"Data inconsistency for agent {ag}"
                assert np.all(self.preferences[fidxs_ag] == self.data_per_agent[ag].preferences), f"Data inconsistency for agent {ag}"
                assert self.agent_data.get(ag, None) is not None
                assert len(self.agent_ids[fidxs_ag]) == len(fidxs_ag)
                assert all(self.agent_ids[fidxs_ag] == ag)
            else:
                assert len(fidxs_ag) == self.l_agent_ids.count(ag), f"Data inconsistency for agent {ag}"
    def push(
        self,
        fragments: Sequence[TrajectoryWithValueSystemRewsPair],
        preferences: np.ndarray,
        preferences_with_grounding: np.ndarray,
        agent_ids = None,
        agent_data = None,
    ) -> Self:
        """Add more samples to the dataset.

        Args:
            fragments: list of pairs of trajectory fragments to add
            preferences: corresponding preference probabilities (probability
                that fragment 1 is preferred over fragment 2)

        Raises:
            ValueError: `preferences` shape does not match `fragments` or
                has non-float32 dtype.
        """
        #print(preferences_with_grounding.shape, len(preferences), self.n_values)
        preferences = np.asarray(preferences, dtype=np.float32)
        assert len(preferences_with_grounding.shape) == 2 and preferences_with_grounding.shape == (len(preferences), self.n_values)
        
        if agent_ids is not None:
            self.l_agent_ids.extend(agent_ids)
            for agent_id in set(agent_ids):
                if agent_id not in self.fidxs_per_agent.keys():
                    self.fidxs_per_agent[agent_id] = []
                self.fidxs_per_agent[agent_id].extend(np.where(np.asarray(agent_ids) == agent_id)[0] + len(self))   
        if agent_data is not None:
            self.agent_data.update(agent_data)
            for agent_id in set(agent_ids):
                if agent_id not in self.data_per_agent.keys():
                    self.data_per_agent[agent_id] = VSLPreferenceDataset(self.n_values, single_agent=True)
                idxs_agent_id = np.where(np.asarray(agent_ids) == agent_id)[0]
                self.data_per_agent[agent_id].push(fragments[idxs_agent_id], preferences[idxs_agent_id], preferences_with_grounding[idxs_agent_id, :], agent_ids=[agent_id]*len(idxs_agent_id), agent_data=None)
        if len(fragments) == 0:
            self.certify_data_consistency()
            return
        fragments1, fragments2 = zip(*fragments)
        if preferences.shape != (len(fragments),):
            raise ValueError(
                f"Unexpected preferences shape {preferences.shape}, "
                f"expected {(len(fragments),)}",
            )
            #raise ValueError("preferences should have dtype float32")

        self.l_fragments1.extend(fragments1)
        self.l_fragments2.extend(fragments2)
        self.preferences = np.concatenate((self.preferences, preferences), dtype=np.float32)
        assert self.preferences.dtype == np.float32, f"Expected dtype {np.float32}, but got {self.preferences.dtype}"
        for i in range(self.n_values):
            self.list_preferences_with_grounding[i] = np.concatenate((self.list_preferences_with_grounding[i], preferences_with_grounding[:, i]))
        self._states1 = None # to force transition mode next time.
        self.certify_data_consistency()
        return self

    def pop(self, n_fragments):
        if len(self.l_fragments1) == 0:
            raise IndexError("pop from empty dataset")
        eliminated_fidxs = np.arange(n_fragments)
        ags = self.l_agent_ids[n_fragments:]
        former_agids = deepcopy(self.l_agent_ids)
        for ag in set(former_agids):
            if ag not in ags:
                self.agent_data.pop(ag, None)
                self.data_per_agent.pop(ag, None)
                self.fidxs_per_agent.pop(ag, None)
            else:
                plen = len(self.fidxs_per_agent[ag])
                self.fidxs_per_agent[ag] = np.setdiff1d(self.fidxs_per_agent[ag], eliminated_fidxs)
                if plen != len(self.fidxs_per_agent[ag]):
                    
                    self.data_per_agent[ag].l_fragments1 = np.asarray(self.l_fragments1)[self.fidxs_per_agent[ag]].tolist()
                    self.data_per_agent[ag].l_fragments2 = np.asarray(self.l_fragments2)[self.fidxs_per_agent[ag]].tolist()
                    self.data_per_agent[ag].preferences = np.asarray(self.preferences)[self.fidxs_per_agent[ag]]

                    for i in range(self.n_values):
                        self.data_per_agent[ag].list_preferences_with_grounding[i] = np.asarray(self.list_preferences_with_grounding[i])[self.fidxs_per_agent[ag]].tolist()


                    self.data_per_agent[ag].l_agent_ids = np.asarray(self.l_agent_ids)[self.fidxs_per_agent[ag]].tolist()
                self.fidxs_per_agent[ag] -= n_fragments
                self.fidxs_per_agent[ag] = self.fidxs_per_agent[ag].tolist()
        for i in range(self.n_values):
            self.list_preferences_with_grounding[i] = self.list_preferences_with_grounding[i][n_fragments:]
        self.l_fragments1 = self.l_fragments1[n_fragments:]
        self.l_fragments2 = self.l_fragments2[n_fragments:]
        self.preferences = self.preferences[n_fragments:]
        self.l_agent_ids = self.l_agent_ids[n_fragments:]

        self._states1 = None # to force transition mode next time.
        #self.transition_mode(device='cpu')
        self.certify_data_consistency()
        return self

    @overload
    def __getitem__(self, key: int) -> Tuple[TrajectoryWithValueSystemRewsPair, float]:
        pass

    @overload
    def __getitem__(
        self,
        key: slice,
    ) -> Tuple[types.Pair[Sequence[TrajectoryWithValueSystemRews]], Sequence[float], Sequence[Sequence[float]]]:
        pass

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, np.integer):
            return (self.l_fragments1[key], self.l_fragments2[key]), self.preferences[key], self.preferences_with_grounding[key], self.agent_ids[key]
        else:
            
            return np.asarray(list(zip(self.fragments1[key], self.fragments2[key]))), self.preferences[key], self.preferences_with_grounding[key], self.agent_ids[key]

    def __len__(self) -> int:
        assert len(self.fragments1) == len(self.fragments2) == len(self.preferences)
        return len(self.fragments1)

    def save(self, path: AnyPath) -> None:
        with open(path, "wb") as file:
            dill.dump(self, file)

    @staticmethod
    def load(path: AnyPath) -> "VSLPreferenceDataset":
        with open(path, "rb") as file:
            return dill.load(file)

    def select_batch(self, batch_size: int = "full", transitions=True, device=None, per_agent=True, replace=False) -> "VSLPreferenceDataset":
        """Selects a random batch of size `batch_size` from the dataset, ensuring equal representation of all agents.

        Args:
            batch_size_per_agent: The number of samples to select per agent.

        Returns:
            fragment_pairs, preferences, preferences_with_grounding, agent_ids: A tuple containing the selected preferences, preferences with grounding, and agent IDs.
        """
        if transitions:
                if (not hasattr(self, '_states1')) or self.states1 is None:
                    self.transition_mode(device=device)
        fidxs_per_agent = dict()
        if isinstance(batch_size, float):
            if per_agent:
                batch_size = int(batch_size * len(self.fragments1) / self.n_agents)
            else:
                batch_size = int(batch_size * len(self.fragments1))
            #print(batch_size_per_agent, len(self.l_fragments1), self.n_agents)
        if batch_size != 'full':
            aggregated_indices = []

            if per_agent:
                for iag, agent in enumerate(list(self.fidxs_per_agent.keys())):
                    agent_indices = self.fidxs_per_agent[agent]
                    selected_indices = np.random.choice(agent_indices, size=batch_size, replace=replace or batch_size >= len(agent_indices))
                    fidxs_per_agent[agent] = np.arange(len(selected_indices))+iag*len(selected_indices)
                    aggregated_indices.extend(selected_indices)
            else:
                selected_indices = np.random.choice(np.arange(len(self.fragments1)), size=batch_size, replace=replace or batch_size >= len(self.fragments1))
                aggregated_indices = selected_indices

                for iag, agent in enumerate(list(self.fidxs_per_agent.keys())):
                    fidxs_per_agent[agent], indices_in_si, _= np.intersect1d(selected_indices, self.fidxs_per_agent[agent], return_indices=True)
                    assert fidxs_per_agent[agent].shape[0] <= len(selected_indices), f"Expected {len(selected_indices)}, but got {fidxs_per_agent[agent].shape[0]}"
                    assert set(self.fidxs_per_agent[agent]).issuperset(fidxs_per_agent[agent]), f"Expected {self.fidxs_per_agent[agent]}, but got {fidxs_per_agent[agent]}"
                    fidxs_per_agent[agent] = indices_in_si
            preferences = self.preferences[aggregated_indices].astype(np.float32)
            #assert preferences.dtype == np.float32, f"Expected dtype {np.float32}, but got {preferences.dtype}"
            preferences_with_grounding = self.preferences_with_grounding[aggregated_indices]
            agent_ids = self.agent_ids[aggregated_indices]
            if transitions:
                states1 = self.states1[aggregated_indices]
                states2 = self.states2[aggregated_indices]
                acts1 = self.acts1[aggregated_indices]
                acts2 = self.acts2[aggregated_indices]
                next_states1 = self.next_states1[aggregated_indices]
                next_states2 = self.next_states2[aggregated_indices]
                dones1 = self.dones1[aggregated_indices]
                dones2 = self.dones2[aggregated_indices]
                return ((states1, acts1, next_states1, dones1), (states2, acts2, next_states2, dones2), preferences, preferences_with_grounding, agent_ids), fidxs_per_agent
            else:
                fragments1 = self.fragments1[aggregated_indices]
                fragments2 = self.fragments2[aggregated_indices]
                
                return (fragments1, fragments2, preferences, preferences_with_grounding, agent_ids), fidxs_per_agent
        else:
            if transitions:
                return ((self.states1, self.acts1, self.next_states1, self.dones1), (self.states2, self.acts2, self.next_states2, self.dones2), self.preferences, self.preferences_with_grounding, self.agent_ids), self.fidxs_per_agent
            else:
                return (self.fragments1, self.fragments2, self.preferences, self.preferences_with_grounding, self.agent_ids), self.fidxs_per_agent

    def k_fold_split(self, k: int, generate_val_dataset: bool = True) -> List[Tuple["VSLPreferenceDataset", "VSLPreferenceDataset"]]:
        """Generates k-fold train and validation datasets, ensuring equal agent representation.

        Args:
            k: Number of folds.

        Returns:
            A list of tuples, where each tuple contains a train and validation VSLPreferenceDataset.
        """

        unique_agents = np.random.permutation(np.unique(self.agent_ids))
        agent_indices = {agent: np.where(self.agent_ids == agent)[0] for agent in unique_agents}

        if k >= 2:
            kf = KFold(n_splits=k, shuffle=True, random_state=np.random.randint(0, 10000))
        
            # Create agent-specific splits
            agent_splits = {agent: list(kf.split(np.random.permutation(agent_indices[agent]))) for agent in unique_agents}
        else:
            # If k is 1, just use a single random split 
            agent_splits = {agent: [np.array_split(np.random.permutation(len(agent_indices[agent])), 2)] for agent in unique_agents}
        
        folds = []
        for fold_idx in range(k):
            train_indices = []
            val_indices = []

            # Collect train and val indices for each agent
            for agent in unique_agents:
                train_idx, val_idx = agent_splits[agent][fold_idx]
                train_indices.extend(agent_indices[agent][train_idx])
                val_indices.extend(agent_indices[agent][val_idx])

                # Create train and validation datasets
                train_dataset = VSLPreferenceDataset(self.n_values)
                val_dataset = VSLPreferenceDataset(self.n_values)

                # Populate train dataset
            train_dataset.push(
            np.asarray([(self.fragments1[i], self.fragments2[i]) for i in train_indices]),
            self.preferences[train_indices],
            self.preferences_with_grounding[train_indices, :],
            agent_ids=self.agent_ids[train_indices],
            agent_data=self.agent_data
            )
            
            if generate_val_dataset:
                # Populate validation dataset
                val_dataset.push(
                np.asarray([(self.fragments1[i], self.fragments2[i]) for i in val_indices]),
                self.preferences[val_indices],
                self.preferences_with_grounding[val_indices, :],
                agent_ids=self.agent_ids[val_indices],
                agent_data=self.agent_data
                )

            folds.append((train_dataset, val_dataset))
            
        
        # Validation script to check splits
        for fold_idx, (train_dataset, val_dataset) in enumerate(folds):
            train_agents, train_counts = np.unique(train_dataset.agent_ids, return_counts=True)
            val_agents, val_counts = np.unique(val_dataset.agent_ids, return_counts=True)

            assert set(train_agents) == set(unique_agents), f"Fold {fold_idx}: Missing agents in train set"
            if generate_val_dataset:
                assert set(val_agents) == set(unique_agents), f"Fold {fold_idx}: Missing agents in val set"

            for agent in unique_agents:
                train_count = train_counts[np.where(train_agents == agent)[0][0]]
                if generate_val_dataset:
                    val_count = val_counts[np.where(val_agents == agent)[0][0]]
                    total_count = len(agent_indices[agent])
                    assert train_count + val_count == total_count, (
                        f"Fold {fold_idx}: Incorrect split for agent {agent}. "
                        f"Train: {train_count}, Val: {val_count}, Total: {total_count}"
            )

        random.shuffle(folds)   
        return folds



class FixedLengthVSLPreferenceDataset(VSLPreferenceDataset):
    def __init__(self, n_values, single_agent=False, size=10000):
        super().__init__(n_values, single_agent)
        self.l_fragments1 = np.zeros(size, dtype=object)
        self.l_fragments2 = np.zeros(size, dtype=object)
        self.l_agent_ids = np.zeros(size, dtype=object)
        self.preferences = np.zeros(size, dtype=np.float32)
        self.list_preferences_with_grounding = np.zeros((size, self.n_values), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size
        self.fidxs_per_agent = {}

   
    
    @property
    def preferences_with_grounding(self):
        return self.list_preferences_with_grounding[0:self.size]

    @property
    def agent_ids(self):
        return self.l_agent_ids[0:self.size]
    
   
    @property
    def fragments1(self):
        return self.l_fragments1[0:self.size]
    @property
    def fragments2(self):
        return self.l_fragments2[0:self.size]
    
    def __len__(self):
        return self.size
    
    def certify_data_consistency(self):
        #print(set([pi for p in self.fidxs_per_agent.values() for pi in p]))
        #print(set(np.arange(len(self)).tolist()))
        #assert set([pi for p in self.fidxs_per_agent.values() for pi in p]) == set(np.arange(len(self)).tolist()), "Data inconsistency: fidxs_per_agent does not cover all fragments"
        print("SIZE", self.size)
        print("PTR", self.ptr)
        print("MAX SIZE", self.max_size)
        for ag in self.fidxs_per_agent.keys():
            
            fidxs_ag = self.fidxs_per_agent[ag]
            #assert len(fidxs_ag) == len(self.data_per_agent[ag]), f"Data inconsistency for agent {ag}"
                #assert all(self.fragments1[fidxs_ag] == self.data_per_agent[ag].fragments1), f"Data inconsistency for agent {ag}"
                #assert all(self.fragments2[fidxs_ag] == self.data_per_agent[ag].fragments2), f"Data inconsistency for agent {ag}"
                #assert np.all(self.preferences[fidxs_ag] == self.data_per_agent[ag].preferences), f"Data inconsistency for agent {ag}"
            assert self.agent_data.get(ag, None) is not None, (self.agent_data.keys(), self.fidxs_per_agent.keys())
            assert len(self.agent_ids[fidxs_ag]) == len(fidxs_ag)
            assert all(self.agent_ids[fidxs_ag] == ag), f"Data inconsistency for agent {ag}, {self.agent_ids[fidxs_ag]}"
            assert len(fidxs_ag) == np.where(self.l_agent_ids == ag)[0].size, f"Data inconsistency for agent {ag}"
    def push(
        self,
        fragments: Sequence[TrajectoryWithValueSystemRewsPair],
        preferences: np.ndarray,
        preferences_with_grounding: np.ndarray,
        agent_ids = None,
        agent_data = None,
    ) -> Self:
        self.size = self.size + min(len(fragments), self.max_size - self.size)
        """Add more samples to the dataset.

        Args:
            fragments: list of pairs of trajectory fragments to add
            preferences: corresponding preference probabilities (probability
                that fragment 1 is preferred over fragment 2)

        Raises:
            ValueError: `preferences` shape does not match `fragments` or
                has non-float32 dtype.
        """
        if len(fragments) == 0:
            self.certify_data_consistency()
            return
        #print(preferences_with_grounding.shape, len(preferences), self.n_values)
        preferences = np.asarray(preferences, dtype=np.float32)
        assert len(preferences_with_grounding.shape) == 2 and preferences_with_grounding.shape == (len(preferences), self.n_values)
        len_data = len(fragments)
        assert agent_ids is not None and len(agent_ids) == len_data, "agent_ids must be provided and have the same length as fragments"
        print(len_data, self.ptr, self.max_size)
        end_ptr = self.ptr + len_data
        if end_ptr <= self.max_size:
            self.l_agent_ids[self.ptr:end_ptr] = agent_ids
        else:
            first_part = self.max_size - self.ptr
            self.l_agent_ids[self.ptr:self.max_size] = agent_ids[:first_part]
            self.l_agent_ids[0:end_ptr % self.max_size] = agent_ids[first_part:]
        if agent_data is not None:
            assert agent_ids is not None, "agent_ids must be provided if agent_data is provided"
            assert set(agent_data.keys()) == set(agent_ids), "agent_data keys must match agent_ids"
            self.agent_data.update(agent_data)
            
        
        fragments1, fragments2 = zip(*fragments)
        if preferences.shape != (len(fragments),):
            raise ValueError(
                f"Unexpected preferences shape {preferences.shape}, "
                f"expected {(len(fragments),)}",
            )
            #raise ValueError("preferences should have dtype float32")
        assert len(fragments1) == len_data
        if end_ptr <= self.max_size:
            self.l_fragments1[self.ptr:end_ptr] = fragments1
            self.l_fragments2[self.ptr:end_ptr] = fragments2
            self.preferences[self.ptr:end_ptr] = preferences
            assert self.preferences.dtype == np.float32, f"Expected dtype {np.float32}, but got {self.preferences.dtype}"
            for i in range(self.n_values):
                self.list_preferences_with_grounding[self.ptr:end_ptr, i] = preferences_with_grounding[:, i]
        else:
            first_part = self.max_size - self.ptr
            self.l_fragments1[self.ptr:self.max_size] = fragments1[:first_part]
            self.l_fragments1[0:end_ptr % self.max_size] = fragments1[first_part:]
            self.l_fragments2[self.ptr:self.max_size] = fragments2[:first_part]
            self.l_fragments2[0:end_ptr % self.max_size] = fragments2[first_part:]
            assert self.preferences.dtype == np.float32, f"Expected dtype {np.float32}, but got {self.preferences.dtype}"
            for i in range(self.n_values):
                self.list_preferences_with_grounding[self.ptr:self.max_size, i] = preferences_with_grounding[:first_part, i]
                self.list_preferences_with_grounding[0:end_ptr % self.max_size, i] = preferences_with_grounding[first_part:, i]
        self._states1 = None # to force transition mode next time.
        
        ags_new = set(self.l_agent_ids)
        for agent_id in ags_new:
            if isinstance(agent_id, str):
                self.fidxs_per_agent[agent_id] = np.where(self.l_agent_ids == agent_id)[0]
        old_agents = deepcopy(list(self.fidxs_per_agent.keys()))
        for agent_id in old_agents:
            if agent_id not in ags_new:
                self.fidxs_per_agent.pop(agent_id)
                self.agent_data.pop(agent_id, None)

        self.ptr += len_data
        self.ptr = self.ptr % self.max_size
        
        self.certify_data_consistency()
        return self

    def pop(self, n_fragments):
        pass
    

    
    