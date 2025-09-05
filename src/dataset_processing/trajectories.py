
from src.dataset_processing.utils import TRAJECTORIES_DATASETS_PATH, USEINFO, calculate_trajectory_save_path
from src.dataset_processing.data import TrajectoryWithValueSystemRews, load_vs_trajectories, save_vs_trajectories


import numpy as np
import torch


import os
from typing import Sequence



def compare_trajectories(traj_i, traj_j, epsilon=0.0001):
    """
    Compare two trajectories based on their discounted sums.
    Args:
        traj_i (float): Discounted sum of the first trajectory.
        traj_j (float): Discounted sum of the second trajectory.
        epsilon (float): Threshold for comparison.
    Returns:
        float: Comparison flag (1.0, 0.5, 0.0).
    """
    comparison = traj_i - traj_j
    if abs(comparison) <= epsilon :
        return 0.5
    elif comparison > 0:
        return 1.0
    else:
        return 0.0


def load_trajectories(dataset_name, ag, environment_data, society_data, override_dtype=np.float32) -> Sequence[TrajectoryWithValueSystemRews]:
    path = calculate_trajectory_save_path(
        dataset_name, ag, environment_data, society_data)
    trajs = load_vs_trajectories(
        os.path.join(TRAJECTORIES_DATASETS_PATH, path))
    new_trajs = [0]*len(trajs)
    for i, t in enumerate(trajs):
        t: TrajectoryWithValueSystemRews
        if isinstance(override_dtype, torch.dtype):
            new_trajs[i] = TrajectoryWithValueSystemRews(
                t.obs, 
                t.acts, 
                t.infos, t.terminal, 
                t.n_vals,
                torch.tensor(t.value_rews, dtype=override_dtype, requires_grad=False), 
                torch.tensor(t.vs_rews, dtype=override_dtype, requires_grad=False), 
                torch.tensor(t.v_rews_real, dtype=override_dtype, requires_grad=False) if t.v_rews_real is not None else None,
                torch.tensor(t.rews_real, dtype=override_dtype, requires_grad=False) if t.rews_real is not None else None,
                torch.tensor(t.dones, dtype=np.float32, requires_grad=False),
                agent=t.agent)

        else:
            assert np.issubdtype(override_dtype, np.floating)
            new_trajs[i] = TrajectoryWithValueSystemRews(t.obs, t.acts, t.infos, t.terminal, 
                                                         rews=np.array(
                t.vs_rews, dtype=override_dtype), v_rews=np.array(t.value_rews, dtype=override_dtype),
                dones=np.array(t.dones, dtype=np.float32),
                v_rews_real=np.array(t.v_rews_real, dtype=override_dtype) if t.v_rews_real is not None else None,
                rews_real=np.array(t.rews_real, dtype=override_dtype) if t.rews_real is not None else None,
                n_vals=t.n_vals, agent=t.agent)



    return new_trajs


def save_trajectories(trajectories: Sequence[TrajectoryWithValueSystemRews], dataset_name, ag, environment_data, society_data, dtype=np.float32):
    path = calculate_trajectory_save_path(
        dataset_name, ag, environment_data, society_data)
    path = os.path.join(TRAJECTORIES_DATASETS_PATH, path)
    os.makedirs(path, exist_ok=True)
    save_vs_trajectories(path=path, trajectories=trajectories,
                         dtype=dtype, use_infos=USEINFO)