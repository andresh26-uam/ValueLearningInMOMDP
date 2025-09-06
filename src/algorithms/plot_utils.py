from collections import OrderedDict
from copy import deepcopy
import csv
from functools import partial
import math
import pprint
from typing import Dict, List
import gymnasium
import matplotlib.pyplot as plt
import itertools
from matplotlib import cm, pyplot as plt
import numpy as np
import pandas as pd
import torch
from defines import transform_weights_to_tuple
from src.algorithms.clustering_utils_simple import ClusterAssignment, ClusterAssignmentMemory
from src.algorithms.preference_based_vsl_simple import PVSL
from src.algorithms.utils import  mce_partition_fh, mce_occupancy_measures
from src.dataset_processing.data import TrajectoryWithValueSystemRews, VSLPreferenceDataset
from src.dataset_processing.utils import calculate_expert_policy_save_path
from src.policies.mushroom_agent_mobaselines import MOBaselinesAgent, ProxyPolicy, obtain_trajectories_and_eval_mo
from src.policies.vsl_policies import VAlignedDictSpaceActionPolicy
import os
from imitation.data import rollout
from collections import defaultdict
import re

import matplotlib.pyplot as plt
from utils import visualize_pareto_front


def get_color_gradient(c1, c2, mix):
    """
    Given two hex colors, returns a color gradient corresponding to a given [0,1] value
    """
    c1_rgb = np.array(c1)
    c2_rgb = np.array(c2)
    mix = torch.softmax(torch.tensor(np.array(mix)),dim=0).detach().numpy()
    return (mix[0]*c1_rgb + ((1-mix[0])*c2_rgb))


def get_linear_combination_of_colors(keys, color_from_keys, mix_weights):

    return np.average(np.array([color_from_keys[key] for key in keys]), 
                      weights=torch.softmax(torch.tensor(np.array(mix_weights)),dim=0).detach().numpy(), axis=0)


def pad(array, length):
    new_arr = np.zeros((length,))
    new_arr[0:len(array)] = np.asarray(array)
    if len(new_arr) > len(array):
        new_arr[len(array):] = array[-1]
    return new_arr

"""
def plot_learning_curves(algo: PVSL, historic_metric, name_metric='Linf', name_method='test_lc', align_func_colors=lambda al: 'black', ylim=(0.0,1.1), show=False, usecmap='viridis'):
    plt.figure(figsize=(6, 6))
    plt.title(
        f"Learning curve for {name_metric}\nover {len(historic_metric)} repetitions.")
    plt.xlabel("Training Iteration")
    plt.ylabel(f"{name_metric}")
    n_possible_al = len(historic_metric[0].keys())
    viridis = cm.get_cmap(usecmap, n_possible_al)  # Get 'viridis' colormap with number of AL strategies
    

    for idx, al in enumerate(historic_metric[0].keys()):
        if al not in historic_metric[0].keys() or len(historic_metric[0][al]) == 0 :
            continue
        if usecmap is None or (np.sum(al) == 1.0 and 1.0 in al):
            color = align_func_colors(al)
        else:
            color = viridis(idx / (n_possible_al - 1))

        max_length = np.max([len(historic_metric[rep][al])
                            for rep in range(len(historic_metric))])

        vals = np.asarray([pad(historic_metric[rep][al], max_length)
                          for rep in range(len(historic_metric))])
        avg_vals = np.mean(vals, axis=0)
        std_vals = np.std(vals, axis=0)

        #color = align_func_colors(al)
        plt.plot(avg_vals,
                 color=color,
                 label=f'{tuple([float("{0:.3f}".format(t)) for t in al])} Last: {float(avg_vals[-1]):0.2f}'
                 )
        # plt.plot(avg_grad_norms,color=align_func_colors.get(tuple(al), 'black'), label=f'Grad Norm: {float(avg_grad_norms[-1]):0.2f}'

        plt.fill_between([i for i in range(len(avg_vals))], avg_vals-std_vals,
                         avg_vals+std_vals, edgecolor=color, alpha=0.3, facecolor=color)
        # plt.fill_between([i for i in range(len(avg_grad_norms))], avg_grad_norms-std_grad_norms, avg_grad_norms+std_grad_norms,edgecolor='black',alpha=0.1, facecolor='black')
        
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    plt.grid()
    plt.savefig(f'plots/Learning_curves_{name_method}.pdf')
    if show:
        plt.show()
    plt.close()"""


def plot_learned_to_expert_policies(expert_policy, vsl_algo: PVSL, vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=None, namefig='mce_vsl_test', show=False, learnt_policy=None, targets=None):
    targets = targets if targets is not None else (vsl_algo.vsi_target_align_funcs if vsi_or_vgl in ['vsi','sim'] else vsl_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(
        vsl_algo.vsi_target_align_funcs, vsl_algo.vgl_target_align_funcs))
    learnt_policy = learnt_policy if learnt_policy is not None else vsl_algo.learned_policy_per_va

    fig = plt.figure(constrained_layout=True, figsize=(16, 8))
    subfigs = fig.subfigures(nrows=2, ncols=1)
    subfigs[0].suptitle(f'Predicted Policy Matrix ({vsi_or_vgl} - {namefig})')
    subfigs[1].suptitle(f'Real Reward Matrix ({vsi_or_vgl} - {namefig})')

    axesUp = subfigs[0].subplots(nrows=1, ncols=len(targets), sharey=True)
    axesDown = subfigs[1].subplots(nrows=1, ncols=len(targets), sharey=True)

    for i, al in enumerate(targets):
        # Plot the first matrix
        all_learned_al = al if vsi_or_vgl == 'vgl' else [ta_to_la[al] for ta_to_la in target_align_funcs_to_learned_align_funcs] if isinstance(
            target_align_funcs_to_learned_align_funcs, list) else target_align_funcs_to_learned_align_funcs[al]
        if isinstance(target_align_funcs_to_learned_align_funcs, list):
            learned_al = np.mean(all_learned_al, axis=0)
            std_learned_al = np.std(all_learned_al, axis=0)
        else:
            learned_al = all_learned_al
            std_learned_al = [0 for _ in learned_al]

        if isinstance(learnt_policy, list):
            if isinstance(target_align_funcs_to_learned_align_funcs, list):
                pol_per_round = [learnt_policy[j].policy_per_va(
                    all_learned_al[j]) for j in range(len(learnt_policy))]
            else:
                pol_per_round = [learnt_policy[j].policy_per_va(
                    all_learned_al) for j in range(len(learnt_policy))]
            lpol = np.mean(pol_per_round, axis=0)
            std_lpol = np.mean(np.std(pol_per_round, axis=0))
            # We plot the average policy and the average learned alignment function which may not correspond to each other directly.

        else:
            learned_al = al if vsi_or_vgl == 'vgl' else target_align_funcs_to_learned_align_funcs[
                al]
            
            lpol = learnt_policy.policy_per_va(learned_al if isinstance(learned_al[0],str) else learned_al)
        if len(lpol.shape) == 3:
            lpol = lpol[0, :, :]

        im1 = axesUp[i].imshow(lpol, cmap='viridis', vmin=0, vmax=1,
                               interpolation='nearest', aspect=lpol.shape[1]/lpol.shape[0])
        learned_al = learned_al[1] if isinstance(learned_al[0],str) else learned_al

        axesUp[i].set_title(
            f'{tuple([float("{0:.3f}".format(v)) for v in learned_al])}\nSTD: {tuple([float("{0:.3f}".format(v)) for v in std_learned_al])}')
        axesUp[i].set_xlabel('Action')
        axesUp[i].set_ylabel(
            f'State\nSTD: {float("{0:.4f}".format(std_lpol)) if isinstance(learnt_policy, list) else 0.0}')

        
        # Plot the second matrix
        pol = expert_policy.policy_per_va(al)
        if len(pol.shape) == 3:
            pol = pol[0, :, :]

        im2 = axesDown[i].imshow(pol, cmap='viridis', interpolation='nearest',
                                 vmin=0, vmax=1, aspect=pol.shape[1]/pol.shape[0])

        axesDown[i].set_title(f'{al}')
        axesDown[i].set_xlabel('Action')
        axesDown[i].set_ylabel('State')

    subfigs[0].colorbar(im1, ax=axesUp, orientation='vertical',
                        label='State-Action Prob.')
    subfigs[1].colorbar(
        im2, ax=axesDown, orientation='vertical', label='State-Action Prob.')
    # Adjust layout to prevent overlap
    # fig.tight_layout()
    dirr = os.path.join('test_results', namefig)
    os.makedirs(dirr, exist_ok=True)
    fig.savefig('test_results/' + namefig + '_policy_dif.pdf')
    # Show the plot
    if show:
        fig.show()
        plt.show()
    plt.close()


def remove_outliers(data1, data2, threshold=1.1):
    """Remove outliers using the IQR method, applied to both datasets together."""
    # Compute IQR for both datasets
    q1_data1, q3_data1 = np.percentile(data1, [25, 75])
    iqr_data1 = q3_data1 - q1_data1
    lower_bound_data1 = q1_data1 - threshold * iqr_data1
    upper_bound_data1 = q3_data1 + threshold * iqr_data1

    q1_data2, q3_data2 = np.percentile(data2, [25, 75])
    iqr_data2 = q3_data2 - q1_data2
    lower_bound_data2 = q1_data2 - threshold * iqr_data2
    upper_bound_data2 = q3_data2 + threshold * iqr_data2

    # Create masks for both datasets
    mask_data1 = (data1 >= lower_bound_data1) & (data1 <= upper_bound_data1)
    mask_data2 = (data2 >= lower_bound_data2) & (data2 <= upper_bound_data2)

    # Combine the masks: only keep points that are not outliers in both datasets
    combined_mask = mask_data1 & mask_data2
    return combined_mask


def filter_values(data1, data2, min_value=-10):
    """Filter out values smaller than min_value in both datasets."""
    mask_data1 = data1 >= min_value
    mask_data2 = data2 >= min_value
    # Combine the masks: only keep points where both data1 and data2 are >= min_value
    combined_mask = mask_data1 & mask_data2
    return combined_mask



def generate_expert_and_learned_trajs(vsl_algo: PVSL, assignment_test: ClusterAssignment, expert_policy: MOBaselinesAgent, eval_env: gymnasium.Env, agent_data: Dict, epsilon=0.05, n_trajs_per_agent=100, seed=0):
    # Select target alignment functions based on vsi_or_vgl

    
    assignment_test.grounding.set_mode('test')
    print("Generating trajectories")
    generated_experttrajs_per_agent = {}
    generated_learnedtrajs_per_cluster = [None for _ in range(len(assignment_test.assignments_vs))]
    for cluster_id, agents in enumerate(assignment_test.assignments_vs):
        if len(agents) > 0:  
            print("For cluster", cluster_id, f"{len(agents)} agents assigned: {agents}")
            cluster_weights = transform_weights_to_tuple(assignment_test.get_value_system(cluster_id))
            
            trajs_learned, (scalarized_return,
                                scalarized_discounted_return,
                                vec_return,
                                disc_vec_return, scalarized_return_real,
                                scalarized_discounted_return_real,
                                vec_return_real,
                                disc_vec_return_real) = obtain_trajectories_and_eval_mo(
                                    n_seeds=n_trajs_per_agent, agent=vsl_algo.mobaselines_agent, 
                                    env=eval_env, 
                                    exploration=epsilon,
                                    reward_vector=assignment_test.grounding,
                                    ws=[cluster_weights], 
                                    ws_eval=[cluster_weights], seed=seed)
            print("LEARNED TRAJS: ", "REAL RETURN", vec_return_real, "DISCOUNTED REAL RETURN", disc_vec_return_real)
            print("LEARNED TRAJS: ", "LEARNED RETURN", vec_return, "DISCOUNTED LEARNED RETURN", disc_vec_return)
            for agent_id in agents:
                agent_weights = transform_weights_to_tuple(agent_data[agent_id]['value_system'])

                if expert_policy.is_single_objective:
                    assert expert_policy.weights_to_algos.contains(agent_weights)
                expert_policy.set_weights(agent_weights)
                if vsl_algo.mobaselines_agent.is_single_objective:
                    assert vsl_algo.mobaselines_agent.weights_to_algos.contains(cluster_weights)

                vsl_algo.mobaselines_agent.set_weights(agent_weights)
                
                trajs_agent, (scalarized_return,
                                    scalarized_discounted_return,
                                    vec_return,
                                    disc_vec_return, scalarized_return_real,
                                    scalarized_discounted_return_real,
                                    vec_return_real,
                                    disc_vec_return_real) = obtain_trajectories_and_eval_mo(
                                        n_seeds=n_trajs_per_agent, agent=expert_policy, 
                                        exploration=epsilon,
                                        env=eval_env, 
                                        reward_vector=assignment_test.grounding,
                                        ws=[agent_weights], 
                                        warn_relabel_not_possible=False,
                                        ws_eval=[agent_weights], seed=seed)
                generated_experttrajs_per_agent[agent_id] = trajs_agent
                print(f"EXPERT TRAJS: {agent_id}", "REAL RETURN", vec_return_real, "DISCOUNTED REAL RETURN", disc_vec_return_real)
                print(f"EXPERT TRAJS: {agent_id}", "LEARNED RETURN", vec_return, "DISCOUNTED LEARNED RETURN", disc_vec_return)
            
        else:
            trajs_learned = []
            trajs_agent = []
        generated_learnedtrajs_per_cluster[cluster_id] = trajs_learned
            
    
        
    return generated_experttrajs_per_agent, generated_learnedtrajs_per_cluster


def plot_return_pairs(trajs_by_agent, trajs_by_cluster, cluster_colors, run_dir, agent_data, cluster_assignment: ClusterAssignment, namefig='return_pairs', show=False, gamma=1.0):
    """
    Plots learned vs real returns for trajectories, colored by cluster.

    Args:
        trajs_by_agent: dict mapping agent_id to list of trajectories (each trajectory should have .real_return and .learned_return attributes or keys)
        trajs_by_cluster: list of lists of trajectories, one list per cluster
        cluster_to_agents: dict mapping cluster_id to list of agent_ids
        cluster_colors: dict mapping cluster_id to color (e.g., RGB tuple or color string)
        namefig: filename for saving the plot
        show: whether to display the plot
    """
    cluster_to_agents = cluster_assignment.assignments_vs

    fig, axs = plt.subplots(
        nrows=1, ncols=len(agent_data[list(agent_data.keys())[0]]['value_system']),
        figsize=(8 * len(agent_data[list(agent_data.keys())[0]]['value_system']), 8)
    )
    if len(agent_data[list(agent_data.keys())[0]]['value_system']) == 1:
        axs = [axs]
    value_dim = len(agent_data[list(agent_data.keys())[0]]['value_system'])

    for i in range(value_dim):
        ax = axs[i]
        ax.set_title(f"Learned vs Real Returns (Value {i+1}) by Cluster")
        ax.set_xlabel("Real Return")
        ax.set_ylabel("Learned Return")

        
        # Normalize learned rewards to match the range of real rewards, then plot x=y line for reference
        all_real = [rollout.discounted_sum(traj.v_rews_real[i], gamma=gamma) for aid, trajs in trajs_by_agent.items() for traj in trajs]
        all_learned = [rollout.discounted_sum(traj.v_rews[i], gamma=gamma) for aid, trajs in trajs_by_agent.items() for traj in trajs ]
        all_real.extend([rollout.discounted_sum(traj.v_rews_real[i], gamma=gamma) for cid in range(len(trajs_by_cluster)) for traj in trajs_by_cluster[cid]])
        all_learned.extend([rollout.discounted_sum(traj.v_rews[i], gamma=gamma) for cid in range(len(trajs_by_cluster)) for traj in trajs_by_cluster[cid]])

        if all_real and all_learned:
            # Normalize learned rewards to real rewards range
            min_real, max_real = min(all_real), max(all_real)
            min_learned, max_learned = min(all_learned), max(all_learned)
            # Avoid division by zero
            if max_learned - min_learned > 1e-8:
                all_learned_norm = [
                    (x - min_learned) / (max_learned - min_learned) * (max_real - min_real) + min_real
                    for x in all_learned
                ]
            else:
                all_learned_norm = all_learned
            min_val = min(min_real, min(all_learned_norm))
            max_val = max(max_real, max(all_learned_norm))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
        for cid, trajs_cis in enumerate(trajs_by_cluster):
            if len(trajs_cis) == 0:
                continue
            color = cluster_colors[cid] if cid in cluster_colors else None
            real_returns = []
            learned_returns = []
            for traj in trajs_cis:
                traj: TrajectoryWithValueSystemRews
                assert hasattr(traj, 'v_rews') and hasattr(traj, 'v_rews_real'), "Trajectories must have v_rews and v_rews_real attributes"
                #print("TRAJ V_REWS", traj.v_rews, traj.v_rews.shape)
                #print("TRAJ V_REWS REAL", traj.v_rews_real, traj.v_rews_real.shape)

                real = rollout.discounted_sum(traj.v_rews_real[i,:], gamma=gamma)
                learned = rollout.discounted_sum(traj.v_rews[i,:], gamma=gamma)
                if max_learned - min_learned > 1e-8:
                    learned = (learned - min_learned) / (max_learned - min_learned) * (max_real - min_real) + min_real
                real_returns.append(real)
                learned_returns.append(learned)
            ax.scatter(real_returns, learned_returns, color=color, alpha=0.5, label=f'Cluster {cid+1}')

        ax.legend()
        ax.grid(True)
    plt.tight_layout()

    # Save the plot
    dirr = os.path.join(run_dir, 'cluster_rewards', namefig)
    os.makedirs(dirr, exist_ok=True)
    plt.savefig(os.path.join(dirr, 'return_pairs_learned_policies_per_value.pdf'))

    if show:
        plt.show()
    plt.close()
    
    # OPCION A1: plot every agent trajectory with real and learned reward instead. Each cluster will have a different subplot.
    usable_clusters = [cid for cid, agents_cid in enumerate(cluster_to_agents) if len(agents_cid) > 0]
    n_clusters = len(usable_clusters)
    ncols = min(4, n_clusters)
    nrows = (n_clusters + ncols - 1) // ncols
    fig, axs = plt.subplots(figsize=(4 * (ncols + 2), 4 * (nrows+1)), nrows=nrows, ncols=ncols)
    axs = np.array(axs).reshape(-1)  # Flatten in case of single row/col
    #print(usable_clusters, [1 for i in trajs_by_cluster if len(i) > 0])
    #input()
    for cid_index, cid in enumerate(usable_clusters):
        agents_cid = cluster_to_agents[cid]
        assert len(agents_cid) > 0
        ax = axs[cid_index]
        ax.set_title(f"Cluster {cid+1} - Learned vs Real Returns")
        ax.set_xlabel("Real Return")
        ax.set_ylabel("Learned Return")
        
        all_real = [rollout.discounted_sum(traj.rews_real, gamma=gamma) for aid, trajs in trajs_by_agent.items() for traj in trajs if aid in cluster_to_agents[cid]]
        all_learned = [rollout.discounted_sum(traj.rews, gamma=gamma) for aid, trajs in trajs_by_agent.items() for traj in trajs if aid in cluster_to_agents[cid]]
        all_real.extend([rollout.discounted_sum(traj.rews_real, gamma=gamma) for traj in trajs_by_cluster[cid]])
        all_learned.extend([rollout.discounted_sum(traj.rews, gamma=gamma) for traj in trajs_by_cluster[cid]])
        min_val_real = min(all_real)
        max_val_real = max(all_real)
        min_val_learned = min(all_learned)
        max_val_learned = max(all_learned)
        
        
        agent_colors = plt.cm.viridis(np.linspace(0, 1, len(agents_cid)))
        for iad, aid in enumerate(agents_cid):
            trajs_ag = trajs_by_agent[aid]
            real_returns = []
            learned_returns = []
            for traj in trajs_ag:
                traj: TrajectoryWithValueSystemRews
                real = rollout.discounted_sum(traj.rews_real, gamma=gamma)
                learned = rollout.discounted_sum(traj.rews, gamma=gamma)

                if max_val_learned - min_val_learned > 1e-8:
                    learned = (learned - min_val_learned) / (max_val_learned - min_val_learned) * (max_val_real - min_val_real) + min_val_real
                if real is not None and learned is not None:
                    real_returns.append(real)
                    learned_returns.append(learned)
            if real_returns and learned_returns:
                agent_vs = agent_data[aid]['value_system']
                label = f'Agent: '
                if agent_vs is not None:
                    label += f'{tuple([float("{0:.3f}".format(v)) for v in agent_vs])}'
                ax.scatter(real_returns, learned_returns, color=agent_colors[iad], alpha=0.5, label=label, marker='o')
        # add the cluster trajectories.
        trajs_cluster = trajs_by_cluster[cid]
        real_returns = []
        learned_returns = []
        for traj in trajs_cluster:
            traj: TrajectoryWithValueSystemRews
            real = rollout.discounted_sum(traj.rews_real, gamma=gamma)
            learned = rollout.discounted_sum(traj.rews, gamma=gamma)
            if max_val_learned - min_val_learned > 1e-8:
                learned = (learned - min_val_learned) / (max_val_learned - min_val_learned) * (max_val_real - min_val_real) + min_val_real
            if real is not None and learned is not None:
                real_returns.append(real)
                learned_returns.append(learned)
        if real_returns and learned_returns:
            agent_vs = transform_weights_to_tuple(cluster_assignment.get_value_system(cid))
            label = f'Cluster VS: '
            if agent_vs is not None:
                label += f'{tuple([float("{0:.3f}".format(v)) for v in agent_vs])}'
            ax.scatter(real_returns, learned_returns, color=cluster_colors[cid], alpha=0.5, label=label, marker='x')
        
        
        min_val = min(min_val_real, min_val_learned)
        max_val = max(max_val_real, max_val_learned)

        for ax in axs.flatten():
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
            ax.legend()
            ax.grid(True)
    plt.tight_layout()

    # Save the plot
    dirr = os.path.join(run_dir, 'cluster_rewards', namefig)
    os.makedirs(dirr, exist_ok=True)
    plt.savefig(os.path.join(dirr, f'return_pairs_per_cluster.pdf'))

    if show:
        plt.show()
    plt.close()

def plot_cluster_representation(vsl_algo: PVSL):
    pass
    # Validacion de clusters: pintar todas las trayectorias pero en el plano o 3d por objetivos, pintando los clusters.

def TAC_KENDALL(vsl_algo):
    # Compute Kendall's Tau for the learned policies to see if they respect the pareto efficiency.
    # Also compute it to compare with the clusters: try to compute how good the representation is in A1 and A3.
    # Put in the tables too.
    # TODO..
    pass

def plot_learned_and_expert_reward_pairs(vsl_algo: PVSL, learned_rewards_per_al_func, vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=None, namefig='reward_pairs', show=False, targets=None, trajs_sampled=None):
    # Select target alignment functions based on vsi_or_vgl
    targets = targets if targets is not None else (vsl_algo.vsi_target_align_funcs if vsi_or_vgl == 'vsi' else vsl_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(
        vsl_algo.vsi_target_align_funcs, vsl_algo.vgl_target_align_funcs))

    num_targets = len(targets)

    # Adjust subplot grid based on the number of targets
    if num_targets > 3:
        cols = (num_targets + 1) // 2  # Calculate the number of rows needed
        rows = 2  # We want 2 columns
    else:
        rows = 1  # If there are 3 or fewer targets, use a single row
        cols = num_targets
        
    # Create figure and subplots with dynamic layout
    fig, axes = plt.subplots(rows, cols, figsize=(
        16, 8), constrained_layout=True)
    fig.suptitle(
        f'Reward Coincidence Plots ({vsi_or_vgl} - {namefig})', fontsize=16)
    fig.supylabel("Learned Rewards")
    fig.supxlabel("Ground Truth Rewards")

    # Flatten axes if needed (for consistency when accessing them)
    axes = axes.flatten() if num_targets > 1 else [axes]

    for i, al in enumerate(targets):
        ax = axes[i]  # Get the current subplot axis

        # Get learned rewards
        all_learned_al = al if vsi_or_vgl == 'vgl' else [ta_to_la[al] for ta_to_la in target_align_funcs_to_learned_align_funcs] if isinstance(
            target_align_funcs_to_learned_align_funcs, list) else target_align_funcs_to_learned_align_funcs[al]

        if isinstance(target_align_funcs_to_learned_align_funcs, list):
            learned_al = np.mean(all_learned_al, axis=0)
            std_learned_al = np.std(all_learned_al, axis=0)
        else:
            learned_al = all_learned_al
            std_learned_al = [0 for _ in learned_al]
        
        # TODO: put it callable learner reward?
        # If the environment does not have state_dim, we assume it is a single value
            
        if isinstance(learned_rewards_per_al_func, list):
            lr_per_round = [learned_rewards_per_al_func[j](
                al) for j in range(len(learned_rewards_per_al_func))]
            
        else:
            lr_per_round  = [learned_rewards_per_al_func(al), ]
        traj_expert_rewards_al = [0.0 for _ in range(len(lr_per_round))]
        traj_learned_rewards_al = [0.0 for _ in range(len(lr_per_round))]
        expert_reward_al = partial(vsl_algo.env.get_reward_per_align_func, align_func=al)
            
        
        for round_ in range(len(lr_per_round)):
            learned_reward_al = lr_per_round[round_]
            traj_expert_rewards_al[round_] = np.array(
                [np.sum(
                    [float(expert_reward_al(state=s, action=a, next_state=ns, done=None)) for s,a,ns in zip(
                        trajs_sampled[j].obs[:-1], trajs_sampled[j].acts, trajs_sampled[j].obs[1:])])
                            for j in range(len(trajs_sampled))])
            traj_learned_rewards_al[round_] = np.array(
                [np.sum(
                    [float(learned_reward_al(state=s, action=a, next_state=ns, done=None)) for s,a,ns in zip(
                        trajs_sampled[j].obs[:-1], trajs_sampled[j].acts, trajs_sampled[j].obs[1:])])
                            for j in range(len(trajs_sampled))])
        
        traj_learned_rewards_al = np.mean(traj_learned_rewards_al, axis=0)
        traj_expert_rewards_al = np.mean(traj_expert_rewards_al, axis=0)
            

        # Flatten rewards for plotting as pairs
        learned_rewards_flat = traj_learned_rewards_al.flatten()
        expert_rewards_flat = traj_expert_rewards_al.flatten()

        # Remove outliers and filter values smaller than a threshold
        """combined_mask = remove_outliers(
            expert_rewards_flat, learned_rewards_flat)
        learned_rewards_flat = learned_rewards_flat[combined_mask]
        expert_rewards_flat = expert_rewards_flat[combined_mask]"""

        combined_mask = filter_values(
            expert_rewards_flat, learned_rewards_flat, min_value=-1000)
        learned_rewards_flat = learned_rewards_flat[combined_mask]
        expert_rewards_flat = expert_rewards_flat[combined_mask]

        # Scatter plot for reward pairs
        ax.scatter(expert_rewards_flat, learned_rewards_flat,
                   color='blue', alpha=0.5, label='Reward Pairs')

        # Plot x = y line
        min_val = min(min(expert_rewards_flat), min(learned_rewards_flat))
        max_val = max(max(expert_rewards_flat), max(learned_rewards_flat))
        ax.plot([min_val, max_val], [min_val, max_val],
                color='red', linestyle='--', alpha=0.7, label='x=y')

        # Set labels and title for each subplot
        al = al[1] if isinstance(al[0],str) else al 
        learned_al = learned_al[1] if isinstance(learned_al[0],str) else learned_al
        if vsi_or_vgl != 'vgl':
            ax.set_title(
                f'Original: {tuple([float("{0:.3f}".format(v)) for v in al])}\nLearned: {tuple([float("{0:.3f}".format(v)) for v in learned_al])}\nSTD: {tuple([float("{0:.3f}".format(v)) for v in std_learned_al])}')
        else:
            ax.set_title(
                f'VS aggregation: {tuple([float("{0:.3f}".format(v)) for v in al])}')
        
        ax.legend()

    # Remove unused axes if the number of targets is odd
    if num_targets % 2 != 0 and num_targets > 3:
        fig.delaxes(axes[-1])

    # Save the figure
    dirr = os.path.join('test_results', namefig)
    os.makedirs(dirr, exist_ok=True)
    fig.savefig(os.path.join(dirr, 'reward_dif_correlation.pdf'))

    # Show the plot if requested
    if show:
        fig.show()
        plt.show()
    plt.close()


def plot_learned_and_expert_rewards(vsl_algo, learned_rewards_per_al_func, cmap='viridis',  vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=None, namefig='mce_vsl_test', show=False, targets=None):
    targets = targets if targets is not None else (vsl_algo.vsi_target_align_funcs if vsi_or_vgl == 'vsi' else vsl_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(
        vsl_algo.vsi_target_align_funcs, vsl_algo.vgl_target_align_funcs))

    # fig, axes = plt.subplots(2, len(targets), figsize=(16, 8))
    fig = plt.figure(constrained_layout=True, figsize=(16, 8))
    subfigs = fig.subfigures(nrows=2, ncols=1)
    subfigs[0].suptitle(f'Predicted Reward Matrix ({vsi_or_vgl} - {namefig})')
    subfigs[1].suptitle(f'Real Reward Matrix ({vsi_or_vgl} - {namefig})')

    axesUp = subfigs[0].subplots(nrows=1, ncols=len(targets), sharey=True)
    axesDown = subfigs[1].subplots(nrows=1, ncols=len(targets), sharey=True)
    for i, al in enumerate(targets):
        # Plot the learned matrix
        all_learned_al = al if vsi_or_vgl == 'vgl' else [ta_to_la[al] for ta_to_la in target_align_funcs_to_learned_align_funcs] if isinstance(
            target_align_funcs_to_learned_align_funcs, list) else target_align_funcs_to_learned_align_funcs[al]
        if isinstance(target_align_funcs_to_learned_align_funcs, list):
            learned_al = np.mean(all_learned_al, axis=0)
            std_learned_al = np.std(all_learned_al, axis=0)
        else:
            learned_al = all_learned_al
            std_learned_al = [0 for _ in learned_al]

        if isinstance(learned_rewards_per_al_func, list):
            lr_per_round = [learned_rewards_per_al_func[j](
                al)() for j in range(len(learned_rewards_per_al_func))]
            learned_reward_al = np.mean(lr_per_round, axis=0)
            std_reward_al = np.mean(np.std(lr_per_round, axis=0))
        else:
            learned_reward_al = learned_rewards_per_al_func(al)()
        im1 = axesUp[i].imshow(learned_reward_al, cmap=cmap, interpolation='nearest',
                               aspect=learned_reward_al.shape[1]/learned_reward_al.shape[0])
        learned_al = learned_al[1] if isinstance(learned_al[0],str) else learned_al

        axesUp[i].set_title(
            f'{tuple([float("{0:.3f}".format(v)) for v in learned_al])}\nSTD: {tuple([float("{0:.3f}".format(v)) for v in std_learned_al])}')
        axesUp[i].set_xlabel('Action')
        axesUp[i].set_ylabel(
            f'State\nSTD: {float("{0:.4f}".format(std_reward_al)) if isinstance(learned_rewards_per_al_func, list) else 0.0}')

        # Plot the expert matrix
        im2 = axesDown[i].imshow(vsl_algo.env.reward_matrix_per_align_func(
            al), cmap=cmap, interpolation='nearest', aspect=learned_reward_al.shape[1]/learned_reward_al.shape[0])
        axesDown[i].set_title(f'{al}')
        axesDown[i].set_xlabel('Action')
        axesDown[i].set_ylabel('State')

    subfigs[0].colorbar(im1, ax=axesUp, orientation='vertical', label='Reward')
    subfigs[1].colorbar(
        im2, ax=axesDown, orientation='vertical', label='Reward')
    # subfigs[0].tight_layout(pad=3.0)
    # subfigs[1].tight_layout(pad=3.0)
    # Adjust layout to prevent overlap
    dirr = os.path.join('test_results', namefig)
    os.makedirs(dirr, exist_ok=True)
    fig.savefig(os.path.join(dirr, 'reward_dif.pdf'))
    # Show the plot
    if show:
        fig.show()
        plt.show()
    plt.close()

def plot_learned_and_expert_occupancy_measures(vsl_algo: PVSL, 
        learned_rewards_per_al_func,  vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=None, 
        namefig='mce_vsl_test', show=False, targets=None, assumed_expert_pi: VAlignedDictSpaceActionPolicy = None):
    targets = targets if targets is not None else (vsl_algo.vsi_target_align_funcs if vsi_or_vgl == 'vsi' else vsl_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(
        vsl_algo.vsi_target_align_funcs, vsl_algo.vgl_target_align_funcs))

    fig = plt.figure(constrained_layout=True, figsize=(16, 8))
    subfigs = fig.subfigures(nrows=2, ncols=1)
    subfigs[0].suptitle(
        f'Predicted Occupancy Matrix ({vsi_or_vgl} - {namefig})')
    subfigs[1].suptitle(f'Real Occupancy Matrix ({vsi_or_vgl} - {namefig})')

    axesUp = subfigs[0].subplots(nrows=1, ncols=len(targets), sharey=True)
    axesDown = subfigs[1].subplots(nrows=1, ncols=len(targets), sharey=True)

    assert isinstance(vsl_algo.learned_policy_per_va, VAlignedDictSpaceActionPolicy)
    for i, al in enumerate(targets):

        # Plot the first matrix
        all_learned_al = al if vsi_or_vgl == 'vgl' else [ta_to_la[al] for ta_to_la in target_align_funcs_to_learned_align_funcs] if isinstance(
            target_align_funcs_to_learned_align_funcs, list) else target_align_funcs_to_learned_align_funcs[al]
        if isinstance(target_align_funcs_to_learned_align_funcs, list):
            learned_al = np.mean(all_learned_al, axis=0)
            std_learned_al = np.std(all_learned_al, axis=0)
        else:
            learned_al = all_learned_al
            std_learned_al = [0 for _ in learned_al]

        use_action_visitations = vsl_algo.reward_net.use_action or vsl_algo.reward_net.use_next_state

        if isinstance(learned_rewards_per_al_func, list):
            occupancies = []
            for j in range(len(learned_rewards_per_al_func)):
                occupancies.append(mce_occupancy_measures(
                    env=vsl_algo.env,
                    reward=learned_rewards_per_al_func[j](al)(),
                    pi = vsl_algo.learned_policy_per_va.policy_per_va(target_align_funcs_to_learned_align_funcs[al]),
                    discount=vsl_algo.discount,
                    deterministic=not vsl_algo.learn_stochastic_policy,
                    initial_state_distribution=vsl_algo.env.initial_state_dist,
                    use_action_visitations=use_action_visitations)[1])
            learned_oc = np.mean(occupancies, axis=0)
            std_oc = np.mean(np.std(occupancies, axis=0))

        else:

            std_oc = 0.0
            learned_oc = mce_occupancy_measures(env=vsl_algo.env,
                                                reward=learned_rewards_per_al_func(al)(),
                                                discount=vsl_algo.discount,
                                                pi=vsl_algo.learned_policy_per_va.policy_per_va(target_align_funcs_to_learned_align_funcs[al]),
                                                deterministic=not vsl_algo.learn_stochastic_policy,
                                                initial_state_distribution=vsl_algo.env.initial_state_dist,
                                                use_action_visitations=use_action_visitations)[1]
        ocs = np.transpose(learned_oc)

        

        eocs = np.transpose(mce_occupancy_measures(env=vsl_algo.env,
                                                   reward=vsl_algo.env.reward_matrix_per_align_func(al),
                                                   discount=vsl_algo.discount,
                                                   pi=assumed_expert_pi.policy_per_va(al),
                                                   deterministic=not vsl_algo.learn_stochastic_policy,
                                                   initial_state_distribution=vsl_algo.env.initial_state_dist,
                                                   use_action_visitations=use_action_visitations)[1])
        # eocs2 = np.transpose(vsl_algo.mce_occupancy_measures(pi=expert_policy.policy_per_va(al), deterministic=not vsl_algo.expert_is_stochastic,  use_action_visitations=use_action_visitations)[1])
        # eocs3 = np.transpose(vsl_algo.mce_occupancy_measures(pi=assumed_expert_pi, deterministic=not vsl_algo.expert_is_stochastic,  use_action_visitations=use_action_visitations)[1])

        # assert np.allclose(eocs3, eocs2)
        # assert np.allclose(eocs , eocs2)

        if not use_action_visitations:
            ocs = ocs[:, None]
            eocs = eocs[:, None]

        im1 = axesUp[i].imshow(ocs, cmap='viridis', interpolation='nearest',
                               aspect=vsl_algo.env.state_dim/vsl_algo.env.action_dim)
        learned_al = learned_al[1] if isinstance(learned_al[0],str) else learned_al
        axesUp[i].set_title(
            f'{tuple([float("{0:.3f}".format(v)) for v in learned_al])}\nSTD: {tuple([float("{0:.3f}".format(v)) for v in std_learned_al])}')
        axesUp[i].set_xlabel(
            f'State\nSTD: {float("{0:.4f}".format(std_oc)) if isinstance(learned_rewards_per_al_func, list) else 0.0}')
        if use_action_visitations:
            axesUp[i].set_ylabel('Act')

        # Plot the second matrix
        im2 = axesDown[i].imshow(eocs, cmap='viridis', interpolation='nearest',
                                 aspect=vsl_algo.env.state_dim/vsl_algo.env.action_dim)
        axesDown[i].set_title(f'{al}')
        axesDown[i].set_xlabel('State')
        if use_action_visitations:
            axesDown[i].set_ylabel('Act')

    subfigs[0].colorbar(
        im1, ax=axesUp, orientation='vertical', label='Visitation Freq.')
    subfigs[1].colorbar(
        im2, ax=axesDown, orientation='vertical', label='Visitation Freq.')
    # Adjust layout to prevent overlap
    # fig.tight_layout()
    dirr = os.path.join('test_results', namefig)
    os.makedirs(dirr, exist_ok=True)
    fig.savefig(os.path.join(dirr, 'occupancy_dif.pdf'))
    # Show the plot
    if show:
        fig.show()
        plt.show()
    plt.close()


def compute_stats(data_dict, metric_name='acc'):
        ratios = []

        if isinstance(data_dict[list(data_dict.keys())[0]][metric_name][list(data_dict[list(data_dict.keys())[0]][metric_name].keys())[0]][0], dict):
            epsilons = data_dict[list(data_dict.keys())[0]][metric_name][list(data_dict[list(data_dict.keys())[0]][metric_name].keys())[0]][0].keys()
        
            means_per_al = {eps: {al: {} for al in data_dict[list(data_dict.keys())[0]][metric_name].keys()} for eps in epsilons}
            stds_per_al = {eps: {al: {} for al in data_dict[list(data_dict.keys())[0]][metric_name].keys()} for eps in epsilons}
            labels_per_al = {eps: {al: [] for al in data_dict[list(data_dict.keys())[0]][metric_name].keys()} for eps in epsilons} 
            
            for ratio in sorted(data_dict.keys()):
                ratios.append(ratio)
                for eps in epsilons:
                    for al, values in data_dict[ratio][metric_name].items():
                        n = len(values)
                        values_for_eps = [v[eps] for v in values]
                        
                        means_per_al[eps][al][ratio] = np.mean(values_for_eps)
                        stds_per_al[eps][al][ratio] = np.std(values_for_eps)
                        labels_per_al[eps][al].append(f'{tuple([float("{0:.3f}".format(t)) for t in al])}')  # Convert the tuple key to a string for labeling
            #vector of means, stds and labels and number of repetitions per ratio.
            
            return ratios, means_per_al, stds_per_al, labels_per_al, n
        
        else:
            means_per_al = {al: {} for al in data_dict[list(data_dict.keys())[0]][metric_name].keys()}
            stds_per_al = {al: {} for al in data_dict[list(data_dict.keys())[0]][metric_name].keys()}
            labels_per_al = {al: [] for al in data_dict[list(data_dict.keys())[0]][metric_name].keys()}
            ratios = []
            
            for ratio in sorted(data_dict.keys()):
                ratios.append(ratio)
                for al, values in data_dict[ratio][metric_name].items():
                    n = len(values)
                    means_per_al[al][ratio] = np.mean(values)
                    stds_per_al[al][ratio] = np.std(values)
                    labels_per_al[al].append(f'{tuple([float("{0:.3f}".format(t)) for t in al])}')  # Convert the tuple key to a string for labeling
            #vector of means, stds and labels and number of repetitions per ratio.
            
            return ratios, means_per_al, stds_per_al, labels_per_al, n

def plot_vs_preference_metrics(metrics_per_ratio, namefig='test_metrics', show=False,
                                        align_func_colors=None,
                                        usecmap = 'viridis',
                                        value_expectations_per_ratio=None,
                                        value_expectations_per_ratio_expert=None,
                                        target_align_funcs_to_learned_align_funcs = None,
                                        values_names = None,
                                    ):
    
    # Compute stats for 'f1' and 'ce'
    #ratios, f1_means, f1_stds, f1_labels, n = compute_stats(metrics_per_ratio, 'f1')
    #ratios, jsd_means, jsd_stds, ce_labels, n = compute_stats(metrics_per_ratio, 'jsd')
    ratios, acc_means_per_epsilon, acc_stds_per_epsilon, acc_labels_per_epsilon, n = compute_stats(metrics_per_ratio, 'acc')
    _, n_repescados_means_per_epsilon, n_repescados_stds_per_epsilon, _, n = compute_stats(metrics_per_ratio, 'repescados')
    _, jsd_means_per_epsilon, jsd_stds_per_epsilon, jsd_labels_per_epsilon, n = compute_stats(metrics_per_ratio, 'jsd')
    
    # Plot 'acc'
    namefig_base = namefig
    for eps in acc_means_per_epsilon.keys():
        namefig = namefig_base +f'_EPS{eps}_'
        plt.figure(figsize=(16, 8))
        
        viridis = cm.get_cmap(usecmap, len(acc_means_per_epsilon[eps]))  # Get 'viridis' colormap with number of AL strategies
        target_al_func_ro_mean_align_func = None

        for idx, al in enumerate(acc_means_per_epsilon[eps].keys()):
            if usecmap is None or (np.sum(al) == 1.0 and 1.0 in al):
                color = align_func_colors(al)
            else:
                color = viridis(idx / (len(acc_means_per_epsilon[eps]) - 1))

            if target_align_funcs_to_learned_align_funcs is not None:
                if target_al_func_ro_mean_align_func is None:
                    target_al_func_ro_mean_align_func = {}
                all_learned_al = [ta_to_la[al] for ta_to_la in target_align_funcs_to_learned_align_funcs] if isinstance(
                    target_align_funcs_to_learned_align_funcs, list) else target_align_funcs_to_learned_align_funcs[al]

                if isinstance(target_align_funcs_to_learned_align_funcs, list):
                    learned_al = np.mean(all_learned_al, axis=0)
                    std_learned_al = np.std(all_learned_al, axis=0)
                else:
                    learned_al = all_learned_al
                    std_learned_al = [0 for _ in learned_al]
                
                orig_al = tuple([float("{0:.3f}".format(v)) for v in al])
                std_learned_al = tuple([float("{0:.3f}".format(v)) for v in std_learned_al])
                learned_al = tuple([float("{0:.3f}".format(v)) for v in learned_al])
                label = f'Original: {orig_al}\nLearned: {learned_al}\nSTD: {std_learned_al}'
                target_al_func_ro_mean_align_func[al] = learned_al
            else:
                label = f'Target al: {tuple([float("{0:.3f}".format(v)) for v in al])}'
            assert ratios == sorted(ratios)
            plt.errorbar(ratios, [acc_means_per_epsilon[eps][al][r] for r in ratios], yerr=[acc_stds_per_epsilon[eps][al][r] for r in ratios], label=label
                        , capsize=5, marker='o',color=color,ecolor=color)

        plt.title(f'Avg. preference comparisons accuracy over {n} runs (tol: {eps})')
        #plt.ylabel('F1 score (weighted)')
        plt.ylabel('Accuracy')
        plt.xlabel('Proportion of pairs of trajectories chosen by the expert versus randomly')
        plt.ylim((0.0,1.1))
        
        
        plt.legend(title="VS aggregation function", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        #plt.savefig('results/'+ 'f1_score_' + namefig + f'_{n}_runs.pdf')
        dirr = os.path.join('test_results', namefig)
        os.makedirs(dirr, exist_ok=True)
        plt.savefig(os.path.join(dirr, 'acc_score_' + f'_{n}_runs.pdf'))
        # Show the plot
        if show:
            plt.show()

        plt.figure(figsize=(16, 8))
        

        for idx, al in enumerate(jsd_means_per_epsilon[eps].keys()):

            if usecmap is None or (np.sum(al) == 1.0 and 1.0 in al):
                color = align_func_colors(al)
            else:
                color = viridis(idx / (len(jsd_means_per_epsilon[eps]) - 1))


            if target_align_funcs_to_learned_align_funcs is not None:
                all_learned_al = [ta_to_la[al] for ta_to_la in target_align_funcs_to_learned_align_funcs] if isinstance(
                    target_align_funcs_to_learned_align_funcs, list) else target_align_funcs_to_learned_align_funcs[al]

                if isinstance(target_align_funcs_to_learned_align_funcs, list):
                    learned_al = np.mean(all_learned_al, axis=0)
                    std_learned_al = np.std(all_learned_al, axis=0)
                else:
                    learned_al = all_learned_al
                    std_learned_al = [0 for _ in learned_al]

                orig_al = tuple([float("{0:.3f}".format(v)) for v in al])
                std_learned_al = tuple([float("{0:.3f}".format(v)) for v in std_learned_al])
                learned_al = tuple([float("{0:.3f}".format(v)) for v in learned_al])
                label = f'Original: {orig_al}\nLearned: {learned_al}\nSTD: {std_learned_al}'
            else:
                label = f'Target al: {tuple([float("{0:.3f}".format(v)) for v in al])}'
            assert ratios == sorted(ratios)
            plt.errorbar(ratios,[jsd_means_per_epsilon[eps][al][r] for r in ratios], yerr=[jsd_stds_per_epsilon[eps][al][r] for r in ratios], label=label
                        , capsize=5, marker='o',color=color,ecolor=color)

        plt.title(f'Avg. Jensen Shannon div. over {n} runs')
        plt.ylabel('JSD')
        #plt.title(f'Avg. number of reinstated pairs: negative pairs treated as positives due to the tolerance {eps} over {n} runs')
        plt.xlabel('Proportion of pairs of trajectories chosen by the expert versus randomly')
        plt.legend(title="VS aggregation function", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        #plt.savefig('results/' + 'jsd_score_' + namefig + f'_{n}_runs.pdf')
        dirr = os.path.join('test_results', namefig)
        os.makedirs(dirr, exist_ok=True)
        plt.savefig(os.path.join(dirr, 'jsd_score_' + f'_{n}_runs.pdf'))
        # Show the plot
        if show:
            plt.show()
            
        plt.close()

        save_stats_to_csv_and_latex(acc_means_per_epsilon[eps], acc_stds_per_epsilon[eps], jsd_means_per_epsilon[eps], jsd_stds_per_epsilon[eps], n_repescados_means_per_epsilon[eps], n_repescados_stds_per_epsilon[eps],  acc_labels_per_epsilon[eps], namefig, n, value_expectations_per_ratio, value_expectations_per_ratio_expert, values_names, target_align_funcs_to_learned_align_funcs=target_al_func_ro_mean_align_func)


def save_stats_to_csv_and_latex(metric1_means, metric1_stds, metric2_means, metric2_stds, metric3_means, metric3_stds, labels, namefig, n, value_expectations_per_ratio, value_expectations_per_ratio_expert, values_names, target_align_funcs_to_learned_align_funcs=None):
    # File names
    csv_file_1 = f'results/tables/{namefig}_{n}_runs_metrics_table.csv'
    csv_file_2 = f'results/tables/{namefig}_{n}_runs_expected_alignments_table.csv'
    latex_file_1 = f'results/tables/{namefig}_{n}_runs_metrics_table.tex'
    latex_file_2 = f'results/tables/{namefig}_{n}_runs_expected_alignments_table.tex'

    # Process value expectations for the learned and expert policies
    value_expectations_learned = value_expectations_per_ratio
    value_expectations_expert = value_expectations_per_ratio_expert

    # Initialize data rows for each table
    metrics_rows = []
    expected_alignments_rows = []

    for target_vs_function in labels:
        # Gather F1 and JSD stats for metrics table
        m1_at_0 = f'{metric1_means[target_vs_function][0.0]:.3f} ± {metric1_stds[target_vs_function][0.0]:.2f}'
        m1_at_1 = f'{metric1_means[target_vs_function][1.0]:.3f} ± {metric1_stds[target_vs_function][1.0]:.2f}'
        m2_at_0 = f'{metric2_means[target_vs_function][0.0]:.2e} ± {metric2_stds[target_vs_function][0.0]:.2e}'
        m2_at_1 = f'{metric2_means[target_vs_function][1.0]:.2e} ± {metric2_stds[target_vs_function][1.0]:.2e}'

        m3_at_0 = f'{metric3_means[target_vs_function][0.0]:.2e} ± {metric3_stds[target_vs_function][0.0]:.2e}'
        m3_at_1 = f'{metric3_means[target_vs_function][1.0]:.2e} ± {metric3_stds[target_vs_function][1.0]:.2e}'

        # Prepare metrics row with conditional "Learned VS" column
        metrics_row = [str(target_vs_function), m1_at_0, m1_at_1, m2_at_0, m2_at_1,m3_at_0, m3_at_1]
        if target_align_funcs_to_learned_align_funcs:
            learned_vs = target_align_funcs_to_learned_align_funcs[target_vs_function]
            metrics_row.insert(1, learned_vs)  # Insert "Learned VS" as the second column
        metrics_rows.append(metrics_row)

        # Prepare expert and learned policy averages for policy table
        expert_values = []
        learned_values = []
        for alb in values_names.keys():
            #expert_data = [(np.mean(data_rep[alb])-np.mean([d[alb] for d in value_expectations_expert[alb]]))/np.std([d[alb] for d in value_expectations_expert[alb]]) for data_rep in value_expectations_expert[target_vs_function]]
            #learned_data = [(np.mean(data_rep[alb])-np.mean([d[alb] for d in value_expectations_expert[alb]]))/np.std([d[alb] for d in value_expectations_expert[alb]]) for data_rep in value_expectations_learned[target_vs_function]]
            expert_data = [data_rep[alb] for data_rep in value_expectations_expert[target_vs_function]] 
            learned_data = [data_rep[alb] for data_rep in value_expectations_learned[target_vs_function]]
            
            expert_values.append(f'{np.mean(expert_data):.3f} ± {np.std(expert_data):.3f}')
            learned_values.append(f'{np.mean(learned_data):.3f} ± {np.std(learned_data):.3f}')
        
        expected_alignments_row = [str(target_vs_function), *expert_values, *learned_values]
        if target_align_funcs_to_learned_align_funcs:
            expected_alignments_row.insert(1, str(learned_vs))  # Insert "Learned VS" as the second column
        expected_alignments_rows.append(expected_alignments_row)
        

    # Write metrics table to CSV
    with open(csv_file_1, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['VS Function']
        if target_align_funcs_to_learned_align_funcs:
            header.append('Learned VS')
        #header.extend(['F1 (random)', 'F1 (expert)', 'JSD (random)', 'JSD (expert)'])
        header.extend(['Acc (random)', 'Acc (expert)', 'JSD (random)', 'JSD (expert)', '#Reins (random)', '#Reins (expert)'])
        
        writer.writerow(header)
        writer.writerows(metrics_rows)

    # Write policy table to CSV
    with open(csv_file_2, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['VS Function']
        if target_align_funcs_to_learned_align_funcs:
            header.append('Learned VS')
        header.extend([f'Expert Policy Avg. \A_{{{v}}}' for v in values_names.values()])
        header.extend([f'Learned Policy Avg. \A_{{{v}}}' for v in values_names.values()])
        writer.writerow(header)
        writer.writerows(expected_alignments_rows)

    # Write metrics table to LaTeX
    with open(latex_file_1, 'w') as f:
        f.write('\\begin{table}[ht]\n')
        f.write('\\centering\n')
        f.write(f'\\caption{{Metrics Results for {namefig} over {n} runs}}\n')
        col_format = '|l|'
        if target_align_funcs_to_learned_align_funcs:
            col_format += 'l|'
        col_format += 'c|c|c|c|'
        f.write(f'\\begin{{tabular}}{{{col_format}}}\n')
        f.write('\\hline\n')
        header_latex = 'VS Function'
        if target_align_funcs_to_learned_align_funcs:
            header_latex += ' & Learned VS'
        #header_latex += ' & F1 (random) & F1 (expert) & JSD (random) & JSD (expert)'
        header_latex += ' & Acc (random) & Acc (expert) & JSD (random) & JSD (expert) & #Reins (random) & #Reins (expert)'
        f.write(header_latex + ' \\\\\n')
        f.write('\\hline\n')
        for row in metrics_rows:
            f.write(' & '.join([str(v) for v in row]) + ' \\\\\n')
        f.write('\\hline\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')

    # Write policy table to LaTeX
    with open(latex_file_2, 'w') as f:
        f.write('\\begin{table}[ht]\n')
        f.write('\\centering\n')
        f.write(f'\\caption{{Policy Averages for {namefig} over {n} runs}}\n')
        col_format = '|l|'
        if target_align_funcs_to_learned_align_funcs:
            col_format += 'l|'
        col_format += 'c|' * (len(values_names) * 2)
        f.write(f'\\begin{{tabular}}{{{col_format}}}\n')
        f.write('\\hline\n')
        header_latex = 'VS Function'
        if target_align_funcs_to_learned_align_funcs:
            header_latex += ' & Learned VS'
        header_latex += ''.join([f' & Expert Policy Avg. $\\A_{{{v}}}$' for v in values_names.values()])
        header_latex += ''.join([f' & Learned Policy Avg. $\\A_{{{v}}}$' for v in values_names.values()])
        f.write(header_latex + ' \\\\\n')
        f.write('\\hline\n')
        for row in expected_alignments_rows:
            f.write(' & '.join([str(v) for v in row]) + ' \\\\\n')
        f.write('\\hline\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')

    print(f"Results saved in CSV: {csv_file_1}, {csv_file_2} and LaTeX: {latex_file_1}, {latex_file_2}")



def plot_metrics_for_experiments(historic_assignments_per_lre: Dict[str, Dict[str, List]], experiment_names: List[str], run_dir: str, maximum_conciseness_per_ename: Dict[str, ClusterAssignmentMemory], n_iterations_real=500, fontsize=16):
    """
    Plots conciseness, representativity, and combined score vs for each assignment in historic assignments
    of each experiment. Each experiment has a different color, and each metric is a different line shape.
    """
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors
    line_styles = {
        "Conciseness": "--",
        "Representativity": "-.",
        "Ray-Turi": "-x",
        #"Ray-Turi Index": "-x",
        "Grounding Coherence": "-o",
    }

    plt.figure(figsize=(10, 6))

    # Calculate mean and standard error
    metrics = {
        "Conciseness": [],
        "Representativity": [],
        "Ray-Turi": [],
        #"Ray-Turi Index": [],
        "Grounding Coherence": [],
    }

    """for idx, ename in enumerate(experiment_names):
        maximum_conciseness_vs = maximum_conciseness_per_ename[ename]
        historic_assignments = historic_assignments_per_lre[ename]

        # Calculate metrics
        conciseness_ename = [min(assignment.conciseness_vs(
        ), maximum_conciseness_vs) for assignment in historic_assignments]
        representativity_ename = [assignment.representativity_vs(
            aggr=np.min) for assignment in historic_assignments]
        combined_score_ename = [assignment.combined_cluster_score_vs(
            ) for assignment in historic_assignments]
        combined_score_ename_rt = [1.0/assignment.combined_cluster_score_vs(
            ) for assignment in historic_assignments]
        grounding_scores_ename = [
            np.mean(assignment.coherence()) for assignment in historic_assignments]
        isl1 = [assignment.L == 1 for assignment in historic_assignments]
        metrics['Ray-Turi Index'].append(combined_score_ename)
        #metrics['Ray-Turi Index'].append(combined_score_ename_rt)
        metrics['Conciseness'].append(conciseness_ename)
        metrics['Representativity'].append(representativity_ename)
        metrics['Grounding Coherence'].append(grounding_scores_ename) TODO: esto."""

    for idx, ename in enumerate(experiment_names):
        maximum_conciseness_vs = maximum_conciseness_per_ename[ename]
        historic_assignments = historic_assignments_per_lre[ename]

        metrics['Ray-Turi'].append(np.array(historic_assignments['ray_turi']))
        #metrics['Ray-Turi Index'].append(combined_score_ename_rt)
        metrics['Conciseness'].append(np.array(historic_assignments['conciseness']))
        metrics['Representativity'].append(np.array(historic_assignments['representativity']))
        metrics['Grounding Coherence'].append(np.array(historic_assignments['coherence']))

    for idx_color, (metric_name, values) in enumerate(metrics.items()):
        # Divide the x-axis into 10 evenly spaced points
        values = np.clip(values, -10.0, 10.0)
        total_points = np.array(values).shape[-1]
        num_points = 10
        x_indices = np.linspace(0, total_points - 1, num_points, dtype=int)
        x = [i + 1 for i in x_indices]

        mean_values = np.mean(np.array(values), axis=0)[x_indices]
        std_error = np.std(np.array(values), axis=0)[x_indices] / np.sqrt(len(experiment_names))

        plt.plot(x, mean_values, line_styles[metric_name], color=colors[idx_color % len(colors)],
             alpha=0.5, label=f"{metric_name}")
        plt.fill_between(x, mean_values - std_error, mean_values + std_error, color=colors[idx_color % len(colors)],
                 alpha=0.2)

    plt.xlabel("Iteration", fontsize=fontsize)
    plt.xticks(x, labels=[int(xi/len(x)*len(historic_assignments))
               for xi in x], fontsize=fontsize)
    plt.ylabel("Metric Value", fontsize=fontsize)
    plt.title("Learning curves", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(False)
    plt.tight_layout()

    # Save the plot
    plot_dir = os.path.join(run_dir, 'learning_curves')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "metrics_plot.pdf")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved metrics plot to {plot_path}")


def plot_di_scores_for_experiments(run_dir, scores, repres, conc, n_clusters, fontsize=16):


    sorted_keys = sorted(
        list(scores.keys()))
    sorted_scores = {key: scores[key] for key in sorted_keys}
    sorted_repres = {key: repres[key] for key in sorted_keys}
    sorted_conc = {key: conc[key] for key in sorted_keys}
    sorted_n_clusters = {key: n_clusters[key] for key in sorted_keys}
    # Prepare data for plotting
    x_labels = [str(k) + f"\nLearned:\n{[f"{Li} ({((ni/sum(sorted_n_clusters[k].values()))*100.0):.1f}%)\n" for Li, ni in sorted_n_clusters[k].items()]}" for k in sorted_scores.keys()]

    x_positions = range(len(x_labels))
    # Plot the scores
    plt.figure(figsize=(10, 6))

    max_score = 0
    for values in sorted_scores.values():
        max_score = max(max_score, max(values))
    if isinstance(max_score, torch.Tensor):
        max_score = float(max_score.detach().cpu().numpy())
    means = [np.mean(np.asarray(values)/max_score)
             for values in sorted_scores.values()]
    errors = [np.std(np.asarray(values)/max_score)/np.sqrt(len(values))
              if len(values) > 1 else 0 for values in sorted_scores.values()]
    
    plt.errorbar(x_positions, means, yerr=errors, fmt='o-',
                 capsize=5, label="Ray-Turi Index", color='green', alpha=0.7)
    
    sorted_scores_rt = {key: [1.0/score for score in values] for key, values in sorted_scores.items()}
    max_score_rt = 0
    for values in sorted_scores_rt.values():
        max_score_rt = max(max_score_rt, max(values))
    if isinstance(max_score_rt, torch.Tensor):
        max_score_rt = float(max_score_rt.detach().cpu().numpy())
    #print(sorted_scores_rt, max_score_rt)
    means = [np.mean(np.array(values)/max_score_rt)
             for values in sorted_scores_rt.values()]
    errors = [np.std(np.array(values)/max_score_rt)/np.sqrt(len(values))
              if len(values) > 1 else 0 for values in sorted_scores_rt.values()]
    #plt.errorbar(x_positions, means, yerr=errors, fmt='o-',
    #             capsize=5, label="Ray-Turi Index", color='green', alpha=0.7)

    means = [np.mean(values) for values in sorted_repres.values()]
    errors = [np.std(values)/np.sqrt(len(values)) if len(values)
              > 1 else 0 for values in sorted_repres.values()]
    plt.errorbar(x_positions, means, yerr=errors, fmt='o-', capsize=5,
                 label="Representativeness", color='red', alpha=0.7)

    means = [np.mean(values) for values in sorted_conc.values()]
    errors = [np.std(values)/np.sqrt(len(values)) if len(values)
              > 1 else 0 for values in sorted_conc.values()]
    plt.errorbar(x_positions, means, yerr=errors, fmt='o-',
                 capsize=5, label="Conciseness", color='blue', alpha=0.7)

    plt.xticks(x_positions, x_labels, rotation=45,
               fontsize=fontsize)  # Increased font size by 75%
    plt.xlabel("L / Number of Clusters",
               fontsize=fontsize)  # Increased font size by 75%
    plt.ylabel("Cluster Score", fontsize=fontsize)  # Increased font size by 75%
    plt.ylim(0, 1.05)
    plt.yticks(np.arange(0, 1.05, 0.1),
               fontsize=fontsize)  # Increased font size by 75%
    plt.title("Scores by Number of Clusters",
              fontsize=fontsize)  # Increased font size by 75%
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=fontsize)  # Increased font size by 75%
    plt.tight_layout()

    # Save the plot
    plot_dir = os.path.join(run_dir, 'di_scores')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "di_scores_plot.pdf")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved scores plot to {plot_path}")



def generate_assignment_tables(assignment_identifier_to_assignment: Dict[str, ClusterAssignment] | Dict[str, List[ClusterAssignment]], experiment_name, output_columns, output_dir="test_results", values_names=None, label='train_set'):

    # Ensure output directories exist
    csv_dir = os.path.join(output_dir, experiment_name,
                           label, 'tables', 'general', "csv")
    latex_dir = os.path.join(output_dir, experiment_name,
                             label, 'tables', 'general', "latex")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(latex_dir, exist_ok=True)

    for pi, (position, assignments) in enumerate(assignment_identifier_to_assignment.items()):
        # Ensure assignments is always a list for simplicity
        is_list = True
        if not isinstance(assignments, list):
            is_list = False
            assignments = [assignments]

        # If is_list is True and position is over 2, skip processing
        if is_list and pi > 1:
            continue

        # Prepare data for the table
        data = []

        for cluster_idx, agents_inside in enumerate(assignments[0].active_vs_clusters()):
            row = {}
            row["Cluster"] = cluster_idx + 1
            # Sort clusters by the number of agents in descending order
            corresponding_cidx = []
            for assignment in assignments:
                assignment.certify_cluster_assignment_consistency()
                sorted_clusters = sorted(
                    range(assignments[0].assignments_vs), key=lambda idx: -len(assignment.assignments_vs[idx]))
                cidx = sorted_clusters[cluster_idx]
                corresponding_cidx.append(cidx)

            if output_columns.get("value_systems", False):
                value_systems = [assignment.get_value_system(
                    cidx) for assignment, cidx in zip(assignments, corresponding_cidx)]
                means = np.mean(value_systems, axis=0)
                stds = np.std(value_systems, axis=0)
                row["Value System"] = ", ".join(
                    f"{mean:.3f} ± {std:.3f}" for mean, std in zip(means, stds))
            if output_columns.get("num_agents", False):
                num_agents = [len(assignment.assignments_vs[cidx]) for assignment, cidx in zip(
                    assignments, corresponding_cidx)]
                row["Number of Agents"] = f"{np.mean(num_agents):.1f} ± {np.std(num_agents):.1f}"
            if output_columns.get("representativity", False):
                assignment.certify_cluster_assignment_consistency()
                representativities = [
                    ClusterAssignment._representativity_cluster(
                        [d for agent, d in assignment.intra_discordances_vs_per_agent.items(
                        ) if agent in assignment.assignments_vs[cidx]]
                    ) for assignment, cidx in zip(assignments, corresponding_cidx)
                ]
                row["Representativeness"] = f"{np.mean(representativities):.3f} ± {np.std(representativities):.3f}"
            if output_columns.get("conciseness", False):
                conciseness_values = [
                    ClusterAssignment._conciseness(
                        [d for kpair, d in assignment.inter_discordances_vs_per_cluster_pair.items(
                        ) if kpair[0] == cidx or kpair[1] == cidx],
                        assignment.L
                    ) if assignment.L > 1 else '-' for assignment, cidx in zip(assignments, corresponding_cidx)
                ]
                print(conciseness_values, assignment.L)
                row["Conciseness"] = f"{np.mean(conciseness_values):.3f} ± {np.std(conciseness_values):.3f}" if assignments[0].L > 1 else '-'
            if output_columns.get("combined_score", False):
                representativities = np.array([
                    ClusterAssignment._representativity_cluster(
                        [d for agent, d in assignment.intra_discordances_vs_per_agent.items(
                        ) if agent in assignment.assignments_vs[cidx]]
                    ) for assignment, cidx in zip(assignments, corresponding_cidx)
                ])
                conciseness_values = np.array([
                    ClusterAssignment._conciseness(
                        [d for kpair, d in assignment.inter_discordances_vs_per_cluster_pair.items(
                        ) if kpair[0] == cidx or kpair[1] == cidx],
                        assignment.L
                    ) if assignment.L > 1 else '-' for assignment, cidx in zip(assignments, corresponding_cidx)
                ])
                combined_scores = [
                    conciseness_values /
                    (1-representativities) if assignment.L > 1 else '-'
                    for assignment in assignments
                ]
                row["Ray-Turi Index"] = f"{np.mean(combined_scores):.3f} ± {np.std(combined_scores):.3f}" if assignments[0].L > 1 else '-'
            if output_columns.get("grounding_coherence", False):
                coherence_values = [
                    [
                        ClusterAssignment._representativity_cluster(
                            [d for agent, d in assignment.intra_discordances_gr_per_agent[i].items(
                            ) if agent in assignment.assignments_vs[cidx]]
                        ) for i in range(len(assignment.gr_score))
                    ] for assignment, cidx in zip(assignments, corresponding_cidx)
                ]
            coherence_means = np.mean(coherence_values, axis=0)
            coherence_stds = np.std(coherence_values, axis=0)
            for i, (mean, std) in enumerate(zip(coherence_means, coherence_stds)):
                row[f"Coherence V{i + 1}" if values_names is None else f"Coherence {values_names[i]}"] = f"{mean:.3f} ± {std:.3f}"
            data.append(row)

        # Assignment-level information
        row = {}
        row["Cluster"] = "Total"

        if output_columns.get("value_systems", False):
            avg_value_systems = [assignment.average_value_system()
                                 for assignment in assignments]
            means = np.mean(avg_value_systems, axis=0)
            stds = np.std(avg_value_systems, axis=0)
            row["Value System"] = ", ".join(
                f"{mean:.3f} ± {std:.3f}" for mean, std in zip(means, stds))
        if output_columns.get("num_agents", False):
            num_agents = [assignment.n_agents for assignment in assignments]
            row["Number of Agents"] = f"{np.mean(num_agents):.1f} ± {np.std(num_agents):.1f}"
        if output_columns.get("representativity", False):
            representativities = [assignment.representativity_vs(
                aggr='weighted') for assignment in assignments]
            row["Representativeness"] = f"{np.mean(representativities):.3f} ± {np.std(representativities):.3f}"
        if output_columns.get("conciseness", False):
            conciseness_values = [assignment.conciseness_vs(
            ) if assignment.L > 1 else '-' for assignment in assignments]
            row["Conciseness"] = f"{np.mean(conciseness_values):.3f} ± {np.std(conciseness_values):.3f}" if assignments[0].L > 1 else '-'
        if output_columns.get("combined_score", False):
            combined_scores = [
                assignment.conciseness_vs() / (1.0 - assignment.representativity_vs()
                                               ) if assignment.L > 1 else '-'
                for assignment in assignments
            ]
            row["Ray-Turi Index"] = f"{np.mean(combined_scores):.3f} ± {np.std(combined_scores):.3f}" if assignments[0].L > 1 else '-'
        if output_columns.get("grounding_coherence", False):
            coherence_values = [
                assignment.gr_score for assignment in assignments]
            coherence_means = np.mean(coherence_values, axis=0)
            coherence_stds = np.std(coherence_values, axis=0)
            for i, (mean, std) in enumerate(zip(coherence_means, coherence_stds)):
                row[f"Coherence V{i + 1}" if values_names is None else f"Coherence {values_names[i]}"] = f"{mean:.3f} ± {std:.3f}"
        data.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Save to CSV

        csv_path = os.path.join(
            csv_dir, f"{position}_list.csv" if is_list else f"{position}.csv")
        df.to_csv(csv_path, index=False)

        # Save to LaTeX
        latex_path = os.path.join(
            latex_dir, f"{position}_list.tex" if is_list else f"{position}.tex")
        with open(latex_path, "w") as f:
            f.write(df.to_latex(index=False, escape=False))

        print(
            f"Saved {position} assignment table to {csv_path} and {latex_path}")

from scipy import stats
import pandas as pd

def generate_assignment_tables_v2(table_data: dict, Lmax_to_enames: dict, values_short_names: list, run_dir: str, label: str) :
    """ DATA IS LIKE: 
    train_data_table[ename][best.Lmax] = {
                "L": best.L,
                "VS": best.value_systems,
                "Repr": best.representativity_vs(aggr=np.mean),
                "Conc": best.conciseness_vs(),
                "RT": best.combined_cluster_score_vs(, conciseness_if_L_is_1=maximum_conciseness_per_ename[ename]),
                **{f"Chr {values_short_names[i]}": best.coherence()[i] for i in range(len(values_names))},
                "HYPERVOLUME??" TODO,
                "DISPERSION??" TODO,
            })

            output_columns = {
            "value_systems": True,
            "num_agents": True,
            "representativity": True,
            "conciseness": True,
            "combined_score": True,
            "grounding_coherence": True,
        }
            """
    
    example_ename = list(table_data.keys())[0]
    n_rows = len(Lmax_to_enames) # 1 row per Lmax
    Lmaxes = list(Lmax_to_enames.keys())
    table = []
    for row_index in range(n_rows):
        row = {}
        Lmax_now = Lmaxes[row_index]
        data_for_row = [table_data[ename][Lmax_now] for ename in Lmax_to_enames[Lmax_now]]
        print("DR", data_for_row)
        row["Lmax"] = Lmax_now
        Lslearned = [data_for_row[i]["L"] for i in range(len(data_for_row))]
        # Compute mode and frequency for Lslearned
        mode_val, mode_count = stats.mode(Lslearned)
        mode_val = mode_val[0]
        mode_count = mode_count[0]
        row["L"] = f"{mode_val}, ({mode_count/len(Lslearned)*100.0:.1f}%)"
        # Add the rest of values and their frequency
        unique_vals, counts = np.unique(Lslearned, return_counts=True)
        for val, count in zip(unique_vals, counts):
            freq = count / len(Lslearned) * 100.0
            row["L"] += f", {val} ({freq:.1f}%)"
        row["Repr"] = f"{np.mean([data_for_row[i]['Repr'] for i in range(len(data_for_row))]):.3f} ± {np.std([data_for_row[i]['Repr'] for i in range(len(data_for_row))]):.3f}"
        row["Conc"] = f"{np.mean([data_for_row[i]['Conc'] for i in range(len(data_for_row))]):.3f} ± {np.std([data_for_row[i]['Conc'] for i in range(len(data_for_row))]):.3f}"
        row["RT"] = f"{np.mean([data_for_row[i]['RT'] for i in range(len(data_for_row))]):.3f} ± {np.std([data_for_row[i]['RT'] for i in range(len(data_for_row))]):.3f}"
        for v in values_short_names:
            row[f"Chr {v}"] = f"{np.mean([data_for_row[i][f'Chr {v}'] for i in range(len(data_for_row))]):.3f} ± {np.std([data_for_row[i][f'Chr {v}'] for i in range(len(data_for_row))]):.3f}"
        table.append(row)
    # Convert to DataFrame
    table.sort(key=lambda x: x["Lmax"], reverse=True)
    df = pd.DataFrame(table)

    csv_dir = os.path.join(run_dir,
                           label, 'tables', 'general', "csv")
    latex_dir = os.path.join(run_dir,
                             label, 'tables', 'general', "latex")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(latex_dir, exist_ok=True)

    # Save to CSV
    csv_path = os.path.join(
        csv_dir, f"summary.csv")
    df.to_csv(csv_path, index=False)

    # Save to LaTeX
    latex_path = os.path.join(
        latex_dir, f"summary.tex")
    with open(latex_path, "w") as f:
        f.write(df.to_latex(index=False, escape=False))

    print(
        f"Saved assignment table to {csv_path} and {latex_path}")

def evaluate(vsl_algo: PVSL, algo_name, enames_all, test_dataset, ref_env, ref_eval_env, run_dir: str, discount, environment_data, expert_policy: MOBaselinesAgent, known_pareto_front=None, plot_di_scores=False, plot_returns=True, num_eval_weights_for_front=20, num_eval_episodes_for_front=20, fontsize=12, sampling_trajs_per_agent=100, sampling_epsilon=0.05, show=False):

        values_names = environment_data['values_names']
        values_short_names = environment_data['values_short_names']
        
        vsl_algo_per_ename = dict()
        best_per_ename = {}
        test_best_per_ename = {}
        enames_per_l = {}
        historic_per_ename = {}
        maximum_conciseness_per_ename = {}
        train_data_table = {}
        test_data_table = {}
        n_clusters = {}
        
        for ename in enames_all:
            best_assignments_list, historic, agent, final_global_step, config =  PVSL.load_state(ename=ename, agent_name=vsl_algo.mobaselines_agent.name, ref_env=ref_env, ref_eval_env=ref_eval_env)
            best,_ = best_assignments_list.get_best_assignment(lexicographic_vs_first=False)
            assert best.is_equivalent_assignment(historic[-1], exhaustive=True)
            
            best_assignments_list: ClusterAssignmentMemory
            best_assignments_list.update_maximum_conciseness(best)

            maximum_conciseness_per_ename[ename] = best_assignments_list.maximum_conciseness_vs

            match = re.search(r"_L(\d+)", ename)
            
            if match:
                l_num = int(match.group(1))
            else:
                l_num = 0

            if l_num==0:
                l_num = best_assignments_list.memory[0].Lmax
            if algo_name != 'pbmorl':
                print(list(best_assignments_list.memory[0].value_systems), )
                assert l_num == best_assignments_list.memory[0].Lmax, f"Expected Lmax {l_num}, but got {best_assignments_list.memory[0].Lmax} for {ename}"
            l_num = best_assignments_list.memory[0].Lmax # Use the real Lmax
            if l_num not in enames_per_l.keys():
                enames_per_l[l_num] = []
            enames_per_l[l_num].append(ename)
            print(f"Evaluating {ename} with L={l_num} and {len(historic)} historic assignments")
            vsl_algo_ename: PVSL = PVSL.load_from_state(best_assignments_list, historic, agent, final_global_step, config)
            vsl_algo_ename.mobaselines_agent.set_envs(env=ref_env, eval_env=ref_eval_env)
            vsl_algo_ename.current_assignment.certify_cluster_assignment_consistency()

            assert best.Lmax == vsl_algo_ename.Lmax, f"Expected Lmax {vsl_algo_ename.Lmax}, but got {best.Lmax} for {ename}"
            if algo_name != 'pbmorl': assert vsl_algo_ename.Lmax == l_num, f"Expected Lmax {l_num}, but got {vsl_algo_ename.Lmax} for {ename}"
            vsl_algo_per_ename[ename] = vsl_algo_ename
            best_per_ename[ename] = best
            assert best.is_equivalent_assignment(vsl_algo_ename.current_assignment, exhaustive=True), f"Best assignment {best} is not equivalent to current assignment {vsl_algo_ename.current_assignment} for {ename}"
            # Save a dict with conciseness, Ray-Turi index, representativity_vs, and coherence for each member of the historic
            historic_per_ename[ename] = {
                    "conciseness": [assignment.conciseness_vs() for assignment in historic],
                    "ray_turi": [assignment.combined_cluster_score_vs() for assignment in historic],
                    "representativity": [assignment.representativity_vs(aggr=np.mean) for assignment in historic],
                    "coherence": [np.mean(assignment.coherence()) for assignment in historic]
                }
                
            
            del historic
            best_t: ClusterAssignment = best.copy()
            best_t.recalculate_discordances(test_dataset, vsl_algo_ename.loss.model_indifference_tolerance)
            best_t.certify_cluster_assignment_consistency()
            test_best_per_ename[ename]=best_t

            if ename not in train_data_table.keys():
                train_data_table[ename] = {}
                test_data_table[ename] = {}
            if best.Lmax not in train_data_table[ename].keys():
                train_data_table[ename][best.Lmax] = []
                test_data_table[ename][best.Lmax] = []

            train_data_table[ename][best.Lmax] = {
                "L": best.L,
                "VS": best.value_systems,
                "Repr": best.representativity_vs(aggr=np.mean),
                "Conc": best.conciseness_vs(),
                "RT": best.combined_cluster_score_vs(conciseness_if_L_is_1=maximum_conciseness_per_ename[ename]),
                **{f"Chr {values_short_names[i]}": best.coherence()[i] for i in range(len(values_names))},
            }
            test_data_table[ename][best.Lmax] = {
                "L": best_t.L,
                "VS": best_t.value_systems,
                "Repr": best_t.representativity_vs(aggr=np.mean),
                "Conc": best_t.conciseness_vs(),
                "RT": best_t.combined_cluster_score_vs(conciseness_if_L_is_1=maximum_conciseness_per_ename[ename]),
                **{f"Chr {values_short_names[i]}": best_t.coherence()[i] for i in range(len(values_names))},
            }
        generate_assignment_tables_v2(test_data_table, Lmax_to_enames=enames_per_l, values_short_names=values_short_names, run_dir=run_dir, label='test')
        generate_assignment_tables_v2(train_data_table, Lmax_to_enames=enames_per_l, values_short_names=values_short_names, run_dir=run_dir, label='train')

        if plot_di_scores:
            scores = {}
            repres = {}
            conc = {}
            n_clusters = {}
            max_conciseness = float('-inf')
            for ename_clean in enames_all:
                best = best_per_ename[ename_clean]
                max_conciseness = max(max_conciseness, maximum_conciseness_per_ename[ename_clean])
                assert max_conciseness < 1, f"Max conciseness is {max_conciseness} for {ename_clean}"
                
                
            

            for best in best_per_ename.values():
                key = best.Lmax
                best: ClusterAssignment
                if key not in scores.keys():
                    scores[key] = []
                    repres[key] = []
                    conc[key] = []
                    n_clusters[key] = []
                scores[key].append(best.combined_cluster_score_vs( conciseness_if_L_is_1=max_conciseness if max_conciseness > 0 else None))
                repres[key].append(best.representativity_vs(aggr=np.min))
                conc[key].append(best.conciseness_vs() if best.L >
                                1 else max_conciseness)
                n_clusters[key].append(best.L)
            for key in scores.keys():
                n_clusters[key] = {
                    Li: n_clusters[key][i] for i, Li in enumerate(sorted(n_clusters[key]))
                }



            plot_di_scores_for_experiments(run_dir, scores, repres, conc, n_clusters=n_clusters, fontsize=fontsize)

        
        
        
        for l_num, enames in enames_per_l.items():
            
            ename = enames[0]

            
            vsl_algo = vsl_algo_per_ename[ename]
            best_gr_then_vs_assignment = best_per_ename[ename]
            
            
            if plot_returns:
                for ename in enames:
                    if '_seed' in ename:
                        seed_ename = ename.split('_seed')[-1]
                        assert int(seed_ename) >= 0, f"Expected seed_ename to be a non-negative integer, but got {seed_ename} for {ename}"
                        seed_ename = '_' + str(seed_ename)
                    else:
                        seed_ename = "_unk"
                    trajs_per_agent, trajs_per_cluster = generate_expert_and_learned_trajs(
                        vsl_algo=vsl_algo_per_ename[ename], 
                        assignment_test=test_best_per_ename[ename], expert_policy=expert_policy,
                        eval_env=ref_eval_env, agent_data=test_dataset.agent_data, 
                        epsilon=sampling_epsilon, n_trajs_per_agent=sampling_trajs_per_agent)
                
                        
                    plot_return_pairs(trajs_per_agent, trajs_per_cluster, cluster_assignment=test_best_per_ename[ename], cluster_colors=vsl_algo.get_cluster_colors('matplotlib'), 
                                    run_dir=run_dir, agent_data=test_dataset.agent_data, namefig=os.path.join("return_pairs_all_clusters", "seed" + seed_ename), show=show)

                    vsl_algo_per_ename[ename].mobaselines_agent.set_envs(env=ref_eval_env, eval_env=ref_eval_env)
                    pareto_front_and_weights, unfiltered_front_and_weights = vsl_algo_per_ename[ename].mobaselines_agent.pareto_front(num_eval_weights_for_front=num_eval_weights_for_front,
                                                                                                            num_eval_episodes_for_front=num_eval_episodes_for_front,
                                                                                                            discount=discount, use_weights=vsl_algo_per_ename[ename].get_value_systems())


                    print("Solution set:", unfiltered_front_and_weights)
                    print("LEARNED PARETO FRONT:", pareto_front_and_weights)
                    print("REAL PARETO FRONT:", known_pareto_front)
                    ppath = os.path.join(run_dir, f'learned_pareto_front')
                    plearned = os.path.join(run_dir, f'learned_solutions_front')
                    os.makedirs(ppath, exist_ok=True)
                    os.makedirs(plearned, exist_ok=True)
                    visualize_pareto_front(title="Learned Pareto Front Comparison",
                                        learned_front_data=pareto_front_and_weights,
                                        known_front_data=known_pareto_front,
                                        with_clusters=best_gr_then_vs_assignment,
                                        objective_names=values_names,
                                        cluster_colors=vsl_algo.get_cluster_colors('matplotlib'),fontsize=fontsize,
                                        save_path=os.path.join(ppath, "seed" + seed_ename), show=show)
                    visualize_pareto_front(title="Learned Solutions",
                                        learned_front_data=unfiltered_front_and_weights,
                                        known_front_data=known_pareto_front,
                                        with_clusters=best_gr_then_vs_assignment,
                                        objective_names=values_names,
                                        cluster_colors=vsl_algo.get_cluster_colors('matplotlib'),fontsize=fontsize,
                                        save_path=os.path.join(plearned, "seed" + seed_ename), show=show)

            
            # plot di scores:
            
            n_iter = max([len(v) for k,v in historic_per_ename.items() if k in enames])

            plot_metrics_for_experiments(historic_per_ename, enames,
                                        run_dir=os.path.join(run_dir, 'train', 'plot_metrics'),
                                        maximum_conciseness_per_ename=maximum_conciseness_per_ename, n_iterations_real=n_iter,  fontsize=fontsize)
            
            

            test_dataset: VSLPreferenceDataset = test_dataset
            target_agent_and_vs_to_learned_ones = {}
            for ag, agdata in test_dataset.agent_data.items():
                cluster_id_ag = best_gr_then_vs_assignment.agent_to_vs_cluster_assignments[ag]
                weight_learned = transform_weights_to_tuple(best_gr_then_vs_assignment.get_value_system(cluster_id_ag))
                target_agent_and_vs_to_learned_ones[(ag, transform_weights_to_tuple(agdata['value_system']))] = weight_learned
            target_al_aid, learned_al_aid = list(target_agent_and_vs_to_learned_ones.items())[0]
            unique_al = set([t[1] for t in target_agent_and_vs_to_learned_ones.keys()])

            # Save table in LaTeX format: original value systems to learned ones
            orig_vs = []
            learned_vs = []
            agent_ids = []
            for (ag, orig), learned in target_agent_and_vs_to_learned_ones.items():
                agent_ids.append(ag)
                orig_vs.append(tuple([float(f"{v:.3f}") for v in orig]))
                learned_vs.append(tuple([float(f"{v:.3f}") for v in learned]))
            df_vs = pd.DataFrame({
                "Agent": agent_ids,
                "Original Value System": orig_vs,
                "Learned Value System": learned_vs
            })
            table_dir = os.path.join(run_dir, "tables")
            os.makedirs(table_dir, exist_ok=True)
            latex_path = os.path.join(table_dir, f"original_vs_to_learned_vs_L{l_num}.tex")
            with open(latex_path, "w") as f:
                f.write(df_vs.to_latex(index=False, escape=False))
            print(f"Saved original vs to learned vs table to {latex_path}")
            targets_all = []
            for t,v in  target_agent_and_vs_to_learned_ones.items():
                if t[1] in unique_al and t[1] not in [tt[1] for tt in targets_all]:
                    targets_all.append(t)
            print("TARGET", targets_all)



            
            
            
        

            num_digits = len(str(len(best_assignments_list.memory)))


            # 2: tables.
            # Put for the first, middle, and last assignment in separated tables.
            # For each assignment, put in a table the value systems of each cluster, the number of agents, the representativity of each cluster regarding value systems, average distance to other clusters, the combined score, the representativity and conciseness of the assignment, and the grouinding coherence (given by the .gr_score).
            #  Make every single column modular, i.e. to activate or deactivate it with a flag.
            # Output the tables in latex anc csv in the test_results/{experiment_name}/csv and test_results/{experiment_name}/latex folders.

 
