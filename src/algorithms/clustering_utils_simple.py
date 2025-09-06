from copy import deepcopy
from functools import cmp_to_key
import itertools
import os
import random
from typing import Dict, List, Mapping, Self, Set, Tuple

from colorama import Fore, init
import dill
from matplotlib import pyplot as plt
import numpy as np
from morl_baselines.common.morl_algorithm import MOAgent
from defines import transform_weights_to_tuple
from src.algorithms.preference_based_vsl_lib import OPTION3, PreferenceModelClusteredVSL, discordance, probability_BT
from src.dataset_processing.data import VSLPreferenceDataset
from src.reward_nets.vsl_reward_functions import RewardVectorModule, VectorModule, LinearAlignmentLayer, create_alignment_layer
import torch as th

from imitation.util import util

def check_grounding_value_system_networks_consistency_with_optim(grounding, value_system_per_cluster, optimizer, only_grounding: bool=False, copies: bool=False, check_grads: bool=False):
    if __debug__:
        """Checks if the optimizer parameters match the networks' parameters."""
        optimizer_params = {param for group in optimizer.param_groups for param in group['params']}
        if check_grads:
            assert all([param.requires_grad for param in optimizer_params]), "Some value system parameters are missing in the optimizer."
            
        #network_params = {param for network in grounding.networks for param in network.parameters()}
        #network_params.update({param for network in value_system_per_cluster for param in network.parameters()})
        network_params = {param for param in grounding.parameters()}
        
        if not only_grounding:
            vs_params = {param for network in value_system_per_cluster for param in network.parameters()}
            if check_grads:
                assert all([param.requires_grad for param in vs_params]), "Some value system parameters are missing in the optimizer."
            network_params.update({param for network in value_system_per_cluster for param in network.parameters()})
        #print(len(optimizer_params), len(network_params))
        if not copies:
            assert optimizer_params == network_params, "Optimizer parameters do not match the networks' parameters."
        else:
            # Check that all parameter values are equal (not just references)
            # Compare parameter values (not ids)
            print("OP", optimizer_params, len(optimizer_params))
            print("NP",network_params, len(network_params))
            assert len(optimizer_params) == len(network_params), "Number of optimizer and network parameters do not match."
            # Match parameters by their content, not just shape
            unmatched_network_params = list(network_params)
            for p_opt in optimizer_params:
                found = False
                for i, p_net in enumerate(unmatched_network_params):
                    if th.allclose(p_opt.data, p_net.data):
                        found = True
                        unmatched_network_params.pop(i)
                        break
                assert found, "Optimizer parameter values do not match any network parameter values."



from morl_baselines.common.weights import equally_spaced_weights,random_weights

def generate_random_assignment(dataset: VSLPreferenceDataset, l_max, alignment_layer_class, alignment_layer_kwargs, ref_grounding: VectorModule, seed: int, evenly_spaced=False):
        
        value_system_per_cluster_c = None
        value_system_per_cluster_c = []
        if not evenly_spaced:
            if l_max > 1:
                new_weights = random_weights(dataset.n_values, l_max, rng=np.random.default_rng(seed=seed))
            else:
                new_weights = [random_weights(dataset.n_values, 1, rng=np.random.default_rng(seed=seed))]
        else:
            new_weights = equally_spaced_weights(dataset.n_values, l_max)
        for cs in range(l_max):
            new_align_func = transform_weights_to_tuple(new_weights[cs])
            # This creates a new random alignment function.
            alignment_layer_c = create_alignment_layer(new_align_func, alignment_layer_class, alignment_layer_kwargs)
            value_system_per_cluster_c.append(
                            alignment_layer_c)


        agent_to_vs_cluster_assignments = {}
        assignments_vs = [list() for _ in range(len(value_system_per_cluster_c))]

        for aid in dataset.agent_data.keys():
            agent_to_vs_cluster_assignments[aid] = {}
                        
            cs = np.random.choice(l_max)
            assignments_vs[cs].append(aid)
            agent_to_vs_cluster_assignments[aid] = cs
                
        
        new_gr = ref_grounding.copy_new()
        reference_assignment = ClusterAssignment(weights_per_cluster=value_system_per_cluster_c, 
                                                 grounding=new_gr,
                                                 agent_to_vs_cluster_assignments=agent_to_vs_cluster_assignments, assignments_vs=assignments_vs)
                                                
        return reference_assignment


def assign_colors_matplotlib(num_coordinates,color_map=plt.cm.tab10.colors):
    colors =  color_map # Use the 'tab10' colormap from matplotlib
    assigned_colors = [colors[i % len(colors)] for i in range(num_coordinates)]
    return assigned_colors
def assign_colors_colorama(num_coordinates):
    init()
    colors = [Fore.RED, Fore.GREEN, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.YELLOW, Fore.WHITE, Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTMAGENTA_EX, Fore.LIGHTCYAN_EX, Fore.LIGHTYELLOW_EX, Fore.LIGHTWHITE_EX]
    assigned_colors = [colors[i % len(colors)] for i in range(num_coordinates)]
    return assigned_colors


COLOR_PALETTE = {
    'colorama': assign_colors_colorama,
    'matplotlib': assign_colors_matplotlib
}
"""def plot_vs_assignments(self, save_path="demo.pdf", pie_and_hist_path="pie_and_histograms.pdf", show=False, subfig_multiplier=5.0, values_color_map=plt.cm.tab10.colors, 
                                values_names=None, values_short_names=None, fontsize=12):
            
            #Plots the agents-to-value-system (VS) assignments in 2D space.
            #Each cluster is represented as a point, and agents are plotted around the cluster center
            #based on their intra-cluster distances. Clusters are separated by their inter-cluster distances.

            #Args:
            #    save_path (str, optional): Path to save the cluster plot. If None, the plot is shown interactively.
            #    pie_and_hist_path (str, optional): Path to save the combined pie charts and histograms. If None, the plot is shown interactively.
            
            if self.inter_discordances_vs is None or self.intra_discordances_vs is None:
                raise ValueError("Inter-cluster and intra-cluster distances must be defined to plot VS assignments.")

            # Extract cluster coordinates
            cluster_idx_to_label, cluster_positions, calculated_distances = extract_cluster_coordinates(
                self.inter_discordances_vs_per_cluster_pair, [cid for (cid, _) in self.active_vs_clusters()]
            )

            cluster_colors_vs = assign_colors_matplotlib(self.L)

            # Create the figure for the cluster plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            max_intra_dist = max(max(self.intra_discordances_vs), 1.0)
            max_radius = 0

            for idx, (x, y) in enumerate(cluster_positions):
                cluster_idx = cluster_idx_to_label[idx]
                if len(calculated_distances) > 0:
                    min_inter_dist = min(d for (i, j), d in calculated_distances.items() if i == cluster_idx or j == cluster_idx)
                else:
                    min_inter_dist = 1.0
                radius = min_inter_dist / 2.0
                max_radius = max(radius, max_radius)

            for idx, (x, y) in enumerate(cluster_positions):
                cluster_idx = cluster_idx_to_label[idx]
                ax.scatter(x, y, color=cluster_colors_vs[idx], label=f"Cluster {cluster_idx}", s=100, zorder=3, marker='x')

                agents = self.assignment_vs[cluster_idx]
                intra_distances = self.intra_discordances_vs_per_agent
                if len(calculated_distances) > 0:
                    min_inter_dist = min(d for (i, j), d in calculated_distances.items() if i == cluster_idx or j == cluster_idx)
                else:
                    min_inter_dist = 1.0

                radius = min_inter_dist / 2.0
                circle = plt.Circle((x, y), radius, color=cluster_colors_vs[idx], fill=False, linestyle='--', alpha=0.5)
                ax.add_artist(circle)

                for agent_idx, agent in enumerate(agents):
                    agent_angle = 2 * np.pi * agent_idx / len(agents)
                    agent_x = x + ((intra_distances[agent] / max_intra_dist) * min_inter_dist / 2) * np.cos(agent_angle)
                    agent_y = y + ((intra_distances[agent] / max_intra_dist) * min_inter_dist / 2) * np.sin(agent_angle)
                    ax.scatter(agent_x, agent_y, color=cluster_colors_vs[idx], s=50, zorder=2)

            ax.set_title("Agents-to-VS Assignments")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xlim(min(-3 * max_radius * 1.3 - fontsize / 200, ax.get_xlim()[0]),
                        max(3 * max_radius * 1.3 + fontsize / 200, ax.get_xlim()[1]))
            ax.set_ylim(min(-3 * max_radius * 1.0, ax.get_ylim()[0]),
                        max(3 * max_radius * 1.0, ax.get_ylim()[1]))
            ax.legend()
            ax.grid(False)

            # Save or show the cluster plot
            if save_path is not None:
                if os.path.dirname(save_path) != '':
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches="tight")
            if show or save_path is None:
                plt.show()
            plt.close()

            # Create the figure for combined pie charts and histograms
            fig_combined = plt.figure(figsize=(12, 12))
            n_clusters = len(cluster_idx_to_label)
            for idx, cluster_idx in enumerate(cluster_idx_to_label):
                # Pie chart
                pie_ax = fig_combined.add_subplot(2, n_clusters, idx + 1, aspect='equal')
                value_system_weights = self.get_value_system(cluster_idx)
                pie_ax.pie(value_system_weights,
                            labels=[f"V{i}" for i in range(len(value_system_weights))] if values_short_names is None else [
                                values_short_names[i] for i in range(len(value_system_weights))],
                            autopct='%f',
                            startangle=90, colors=assign_colors_matplotlib(self.n_values, color_map=values_color_map),
                            textprops={'fontsize': fontsize})
                pie_ax.set_title(f"Cluster {cluster_idx} Value System", fontsize=fontsize)

                # Histogram
                hist_ax = fig_combined.add_subplot(2, n_clusters, n_clusters + idx + 1)
                agents = self.assignment_vs[cluster_idx]
                cluster_representativity = [1.0 - self.intra_discordances_vs_per_agent[agent] for agent in agents]
                hist_ax.hist(cluster_representativity, bins=5, color=cluster_colors_vs[idx], alpha=1.0)
                hist_ax.set_xlim(0, 1.0)
                hist_ax.set_ylim(0, len(agents))
                hist_ax.tick_params(axis='both', which='major', labelsize=fontsize)
                hist_ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
                hist_ax.set_yticks(np.linspace(0, len(agents), num=8, endpoint=True, dtype=np.int64))
                hist_ax.set_title(f"Cluster {cluster_idx} Representativity", fontsize=fontsize)

            # Ensure the pie chart and histogram have the same width
            pie_ax.set_box_aspect(1)
            hist_ax.set_box_aspect(1)  # Save or show the combined pie charts and histograms
            if pie_and_hist_path is not None:
                if os.path.dirname(pie_and_hist_path) != '':
                    os.makedirs(os.path.dirname(pie_and_hist_path), exist_ok=True)
                plt.savefig(pie_and_hist_path, bbox_inches="tight")
            if show or pie_and_hist_path is None:
                plt.show()
            plt.close()"""
class ClusterAssignment(object):

    @property
    def Lmax(self):
        return len(self.weights_per_cluster)
    def find_cluster_with_weights(self, w, only_used_clusters=True):

        diffs = []
        for cluster_idx, weights in enumerate(self.value_systems):
            if only_used_clusters:
                if len(self.assignments_vs[cluster_idx]) == 0:
                    continue
            diffs.append(np.abs(weights - w).sum())
        index_ = np.where(np.array(diffs) == max(diffs))[0][0] if len(diffs) > 0 else None
        return (self.value_systems[index_] if index_ is not None else None), index_

    def _default_aggr_on_gr_scores(x):
                return np.mean(x, axis=0)
    @property
    def value_systems(self):
        return [transform_weights_to_tuple(w.get_alignment_layer()[0][0]) for w in self.weights_per_cluster]
    @property
    def value_systems_active(self):
        return [v for i,v in enumerate(self.value_systems) if len(self.assignments_vs[i]) > 0]
    def copy(self):
        new_grounding= self.grounding.copy()
        new_value_system_per_cluster = [v.copy() for v in self.weights_per_cluster]

        clust = ClusterAssignment(
                                  grounding=new_grounding,
                                  weights_per_cluster=new_value_system_per_cluster,
                          inter_discordances_vs=deepcopy(self.inter_discordances_vs),

                          intra_discordances_vs_per_agent=deepcopy(self.intra_discordances_vs_per_agent),
                          intra_discordances_gr_per_agent=deepcopy(self.intra_discordances_gr_per_agent),
                          inter_discordances_vs_per_cluster_pair=deepcopy(self._inter_discordances_vs_per_cluster_pair),

                          assignments_vs=deepcopy(self.assignments_vs),
                          agent_to_vs_cluster_assignments=deepcopy(self.agent_to_vs_cluster_assignments),
                          aggregation_on_gr_scores=self.aggregation_on_gr_scores)
        clust.explored = bool(self.explored)
        clust.protected = bool(self.protected)
        clust.optimizer_state = deepcopy(self.optimizer_state)
        if hasattr(self, "n_training_steps"):
            clust.n_training_steps = int(self.n_training_steps)
        else:
            clust.n_training_steps = 0
        
        return clust


    @property
    def inter_discordances_vs(self) -> List[float]:
        return self._inter_discordances_vs

    @property
    def intra_discordances_gr_per_agent(self) -> List[Dict[str, float]]:
        return self._intra_discordances_gr_per_agent

    @property
    def intra_discordances_vs_per_agent(self) -> Dict[str, float]:
        return self._intra_discordances_vs_per_agent

    @property
    def inter_discordances_vs_per_cluster_pair(self) -> Dict[Tuple[int, int], float]:
        return self._inter_discordances_vs_per_cluster_pair


    def recalculate_discordances(self, dataset: VSLPreferenceDataset, indifference_tolerance: float):
        """
        Recalculate all discordance metrics based on the provided RewardVectorFunction.
        Implementation should be provided by the user.
        """

        self.certify_cluster_assignment_consistency()
        grounding1 = self.grounding.forward(dataset.fragments1)
        grounding2 = self.grounding.forward(dataset.fragments2)
        #print("GR", grounding1[-3])
        #print("GR", grounding2[-3])
        #print("FR", dataset.fragments1[-3].v_rews)
        #print("FR", dataset.fragments2[-3].v_rews)


        discordance_gr_per_case = [None for _ in range(self.n_values)]
        dprefs_with_grounding = util.safe_to_tensor(dataset.preferences_with_grounding).requires_grad_(False)
        dpreferences = util.safe_to_tensor(dataset.preferences).requires_grad_(False)

        #print("DPREFS", dprefs_with_grounding[-3])
        #print("DPREFS", dpreferences[-3], dataset.agent_ids[-3])

        #pmodel = [None for _ in range(self.n_values)]
        for vi in range(self.n_values):
            assert grounding1.shape[0] == grounding2.shape[0]
            assert grounding1.shape[1] == self.n_values, f"Grounding1 shape {grounding1.shape} does not match expected {self.n_values} values."
            assert dataset.preferences_with_grounding.shape == grounding1.shape
            pmodel_vi = probability_BT(grounding1[:, vi], grounding2[:, vi])
            #print("PMODEL", vi, pmodel_vi[-3])
            #pmodel[vi] = pmodel_vi
            assert pmodel_vi.shape == dataset.preferences_with_grounding[:, vi].shape, f"Probability model shape {pmodel_vi.shape} does not match preferences shape {dataset.preferences_with_grounding[:, vi].shape}."
            discordance_gr_per_case[vi] = discordance(pmodel_vi, dprefs_with_grounding[:, vi], indifference_tolerance=indifference_tolerance, reduce='none')
            #print("DISCORDANCE GR", vi, discordance_gr_per_case[vi][-3])

        pref_model_per_cid = dict()
        cluster_idxs_per_cid = dict()

        self._inter_discordances_vs_per_agent = dict()
        self._intra_discordances_vs_per_agent = dict()
        self._intra_discordances_gr_per_agent = [dict() for _ in range(self.n_values)]
        self._inter_discordances_vs_per_cluster_pair = dict()
        self._inter_discordances_vs = []

        if len(self.weights_per_cluster) == 1:
            (cluster_id1, w1) = list(enumerate(self.weights_per_cluster))[0]
            aid_in_c1 = self.assignments_vs[cluster_id1]
            if cluster_id1 not in pref_model_per_cid:
                    probs_cid1 = probability_BT(w1.forward(grounding1), w1.forward(grounding2)).squeeze(-1)
                    assert probs_cid1.shape == (len(dataset),), f"Expected shape {(len(dataset),)}, got {probs_cid1.shape}"
                    pref_model_per_cid[cluster_id1] = probs_cid1
                    cluster_idxs_per_cid[cluster_id1] = np.arange(len(dataset))
                    for ag in aid_in_c1:
                        fidxs = dataset.fidxs_per_agent[ag]
                        assert all([dataset.fragments1[fidxs][j].agent in ag for j in range(len(fidxs))]), f"Fragment {dataset.fragments1[fidxs][0].agent} does not match agent {ag} in fidxs_per_agent."
                        #for i in fidxs:
                            #print(dpreferences[i], probs_cid1[i], sum(dataset.fragments1[i].rews), sum(dataset.fragments2[i].rews))
                        #    pass
                        #input()
                        self._intra_discordances_vs_per_agent[ag] = discordance(probs_cid1[fidxs], dpreferences[fidxs], indifference_tolerance=indifference_tolerance, reduce='mean')
                        for vi in range(self.n_values):
                            self._intra_discordances_gr_per_agent[vi][ag] = th.mean(discordance_gr_per_case[vi][fidxs])
                            #assert th.allclose(self._intra_discordances_gr_per_agent[vi][ag], discordance(pmodel[vi][fidxs], dprefs_with_grounding[fidxs, vi], indifference_tolerance=indifference_tolerance, reduce='mean')), f"Expected {self._intra_discordances_gr_per_agent[vi][ag]} to be equal to {discordance(pmodel[vi][fidxs], dprefs_with_grounding[fidxs, vi], indifference_tolerance=indifference_tolerance).mean()}"

        else:
            pairs = itertools.combinations(list(enumerate(self.weights_per_cluster)), 2)

            for (cluster_id1, w1), (cluster_id2, w2) in pairs:
                # GR Discordances
                aid_in_c1 = self.assignments_vs[cluster_id1]
                NC1 = len(aid_in_c1)
                if NC1 > 0:
                    if cluster_id1 not in pref_model_per_cid:
                        probs_cid1 = probability_BT(w1.forward(grounding1), w1.forward(grounding2)).squeeze(-1)
                        assert probs_cid1.shape == (len(dataset),), f"Expected shape {(len(dataset),)}, got {probs_cid1.shape}"
                        pref_model_per_cid[cluster_id1] = probs_cid1
                        print(dataset.fidxs_per_agent.keys())
                        cluster_idxs_per_cid[cluster_id1] = [idx for ag in aid_in_c1 for idx in dataset.fidxs_per_agent[ag]]
                        for ag in aid_in_c1:
                            fidxs = dataset.fidxs_per_agent[ag]
                            assert all([dataset.fragments1[fidxs][j].agent in ag for j in range(len(fidxs))]), f"Fragment {dataset.fragments1[fidxs][0].agent} does not match agent {ag} in fidxs_per_agent."
                            self._intra_discordances_vs_per_agent[ag] = discordance(probs_cid1[fidxs], dpreferences[fidxs], indifference_tolerance=indifference_tolerance)
                            for vi in range(self.n_values):
                                self._intra_discordances_gr_per_agent[vi][ag] = th.mean(discordance_gr_per_case[vi][fidxs])
                aid_in_c2 = self.assignments_vs[cluster_id2] 
                
                NC2 = len(aid_in_c2)
                if NC2 > 0:   
                    if cluster_id2 not in pref_model_per_cid:
                        probs_cid2 = probability_BT(w2.forward(grounding1), w2.forward(grounding2)).squeeze(-1)
                        assert probs_cid2.shape == (len(dataset),), f"Expected shape {(len(dataset),)}, got {probs_cid2.shape}"
                        pref_model_per_cid[cluster_id2] = probs_cid2
                        assert pref_model_per_cid[cluster_id2].shape == (len(dataset),) , f"Expected shape {(len(dataset),)}, got {pref_model_per_cid[cluster_id2].shape}"
                        cluster_idxs_per_cid[cluster_id2] = [idx for ag in aid_in_c2 for idx in dataset.fidxs_per_agent[ag]]
                        for ag in aid_in_c2:
                            fidxs = dataset.fidxs_per_agent[ag]
                            assert all([dataset.fragments1[fidxs][j].agent in ag for j in range(len(fidxs))]), f"Fragment {dataset.fragments1[fidxs][0].agent} does not match agent {ag} in fidxs_per_agent."
                            self._intra_discordances_vs_per_agent[ag] = discordance(probs_cid2[fidxs], dpreferences[fidxs], indifference_tolerance=indifference_tolerance)
                            for vi in range(self.n_values):
                            
                                self._intra_discordances_gr_per_agent[vi][ag] = th.mean(discordance_gr_per_case[vi][fidxs])
                            
                # _inter_discordances_vs_per_agent
                if NC1 > 0 and NC2 > 0:
                    if OPTION3:
                        total_dis_aid_in_c1 = 0.0
                        for ag in aid_in_c1:
                            fidxs = dataset.fidxs_per_agent[ag]
                            if ag not in self._inter_discordances_vs_per_agent:
                                    self._inter_discordances_vs_per_agent[ag] = th.zeros((len(self.weights_per_cluster),), dtype=th.float32)
                            
                            dis = discordance(pref_model_per_cid[cluster_id2][fidxs], dpreferences[fidxs], indifference_tolerance=indifference_tolerance)
                            assert dis.shape == (), f"Expected scalar, got {dis.shape}"
                            self._inter_discordances_vs_per_agent[ag][cluster_id2] = dis
                            total_dis_aid_in_c1 = dis  + total_dis_aid_in_c1
                        # _inter_discordances_vs_per_agent 
                        total_dis_aid_in_c2 = 0.0
                        for ag in aid_in_c2:
                            fidxs = dataset.fidxs_per_agent[ag]
                            if ag not in self._inter_discordances_vs_per_agent:
                                    self._inter_discordances_vs_per_agent[ag] = th.zeros((len(self.weights_per_cluster),), dtype=th.float32)
                            dis = discordance(pref_model_per_cid[cluster_id1][fidxs], dpreferences[fidxs], indifference_tolerance=indifference_tolerance)
                            assert dis.shape == (), f"Expected scalar, got {dis.shape}"
                            self._inter_discordances_vs_per_agent[ag][cluster_id1] = dis
                            total_dis_aid_in_c2 = dis  + total_dis_aid_in_c2

                        disc_c1_c2 = 1/(NC1 + NC2)* (total_dis_aid_in_c1 + total_dis_aid_in_c2)
                        self._inter_discordances_vs.append(disc_c1_c2)
                        self._inter_discordances_vs_per_cluster_pair[(cluster_id1, cluster_id2)] = disc_c1_c2
                    else:
                        dis = discordance(pref_model_per_cid[cluster_id1], pref_model_per_cid[cluster_id2], indifference_tolerance=indifference_tolerance, reduce='none')
                        mdis = th.mean(dis)
                        for ag in aid_in_c1:
                            fidxs = dataset.fidxs_per_agent[ag]
                            if ag not in self._inter_discordances_vs_per_agent:
                                self._inter_discordances_vs_per_agent[ag] = th.zeros((len(self.weights_per_cluster),), dtype=th.float32)
                            self._inter_discordances_vs_per_agent[ag][cluster_id2] = th.mean(dis[fidxs])
                        for ag in aid_in_c2:
                            fidxs = dataset.fidxs_per_agent[ag]
                            if ag not in self._inter_discordances_vs_per_agent:
                                self._inter_discordances_vs_per_agent[ag] = th.zeros((len(self.weights_per_cluster),), dtype=th.float32)
                            self._inter_discordances_vs_per_agent[ag][cluster_id1] = th.mean(dis[fidxs])
                        self._inter_discordances_vs.append(mdis)
                        self._inter_discordances_vs_per_cluster_pair[(cluster_id1, cluster_id2)] = mdis

        self.certify_cluster_assignment_consistency()
        #print(self.intra_discordances_gr_per_agent[0]['A0|[0. 1.]_0'])
        #print(self.intra_discordances_gr_per_agent[1]['A0|[0. 1.]_0'])
        #input("Done recalculating discordances.")
        return pref_model_per_cid, cluster_idxs_per_cid
    def __init__(self, weights_per_cluster: List[LinearAlignmentLayer],
                 grounding: RewardVectorModule,
                 inter_discordances_vs=None,
                 intra_discordances_gr_per_agent = None,
                    intra_discordances_vs_per_agent = None,
                    inter_discordances_vs_per_cluster_pair = None,
                 agent_to_vs_cluster_assignments: Mapping[str, int] = {},
                 assignments_vs: List[List[str]] = None,
                 aggregation_on_gr_scores=None):
        self.grounding = grounding
        self.weights_per_cluster = weights_per_cluster

        self._inter_discordances_vs = inter_discordances_vs
        
        self._intra_discordances_gr_per_agent = intra_discordances_gr_per_agent
        self._intra_discordances_vs_per_agent = intra_discordances_vs_per_agent
        self._inter_discordances_vs_per_cluster_pair = inter_discordances_vs_per_cluster_pair
        self.protected = False

        self.agent_to_vs_cluster_assignments = agent_to_vs_cluster_assignments

        if assignments_vs is None:
            self.assignments_vs = [[] for _ in range(len(weights_per_cluster))]
            sorted_assign = sorted(agent_to_vs_cluster_assignments.items())
            for cl in range(len(self.weights_per_cluster)):
                self.assignments_vs[cl] = [agent for agent, clust in sorted_assign if clust == cl]
        else:
            self.assignments_vs = assignments_vs
            assert len(self.assignments_vs) == len(weights_per_cluster), \
                f"Length of assignments_vs {len(self.assignments_vs)} does not match number of clusters {len(weights_per_cluster)}"
            """assert all([set(self.assignments_vs[c]) == {agent for agent, clust in sorted(agent_to_vs_cluster_assignments.items()) if clust == c} for c in range(len(self.assignments_vs))]), \
                f"Assignments vs {self.assignments_vs} do not match agent to vs cluster assignments {agent_to_vs_cluster_assignments}"
                """
        self.explored = False

        self.optimizer_state = None # This is useful when saving and loading cluster assignments.
        if aggregation_on_gr_scores is None:
            aggregation_on_gr_scores = ClusterAssignment._default_aggr_on_gr_scores
        self.aggregation_on_gr_scores = aggregation_on_gr_scores
        self.n_training_steps = 0
        self.certify_cluster_assignment_consistency()
    @property
    def n_agents(self):
        return len(self.agent_to_vs_cluster_assignments)
    def get_value_system(self, cluster_idx):
        vs = transform_weights_to_tuple(self.weights_per_cluster[cluster_idx].get_alignment_layer()[0][0].detach().numpy().tolist())
        return vs
    def average_value_system(self):
        self.certify_cluster_assignment_consistency()
        average_vs = np.array([0.0]*self.n_values)
        for cluster_idx in range(len(self.assignments_vs)):
            if len(self.assignments_vs[cluster_idx]) > 0:
                vs = np.array(list(self.get_value_system(cluster_idx)))*len(self.assignments_vs[cluster_idx])
                average_vs += vs
        average_vs /= self.n_agents
        self.certify_cluster_assignment_consistency()
        return average_vs
    def save(self, path: str, file_name: str = "cluster_assignment.pkl"):
        self.certify_cluster_assignment_consistency()
        os.makedirs(path, exist_ok=True)
        self.grounding.requires_grad_(False)
        save_path = os.path.join(path, file_name + '.pkl')
        with open(save_path, "wb") as f:
            dill.dump(self, f)
        self.grounding.requires_grad_(True)
        self.certify_cluster_assignment_consistency()
    def _combined_cluster_score(inter_cluster_distances, intra_cluster_distances_per_agent, n_actual_clusters, cluster_to_agents, conciseness_if_1_cluster=None):
        n_actual_clusters = sum([1 for c in range(len(cluster_to_agents)) if len(cluster_to_agents[c]) > 0],0)
        represent = ClusterAssignment._representativity(intra_cluster_distances_per_agent, cluster_to_agents)
        
        if n_actual_clusters <= 1:
            if (conciseness_if_1_cluster is None) or (conciseness_if_1_cluster == float('-inf')):
                return represent
            else:
                conc = (conciseness_if_1_cluster)
        else:
            conc = ClusterAssignment._conciseness(inter_cluster_distances, n_actual_clusters)
        
        val = (1.0-represent)/(conc + 1.0) # + 1 to avoid overflowing over 1.
        return val

    def _intra_cluster_discordances(intra_cluster_distances_per_agent, cluster_to_agents):
        
        intra_cluster_discordances = [-1.0 for _ in range(len(cluster_to_agents))]
        for c, agents in enumerate(cluster_to_agents):
            if len(agents) == 0:
                continue
            intra_cluster_distances_c = sum(intra_cluster_distances_per_agent[agent] for agent in agents)
            intra_cluster_distances_c /= len(agents)
            intra_cluster_discordances[int(c)] = intra_cluster_distances_c
        
        return intra_cluster_discordances
           
    def _conciseness(inter_cluster_distances, n_actual_clusters):
        if n_actual_clusters <= 1:
            return 1.0
        #distances_non_zero = [d for d in inter_cluster_distances if d > 0]
        if len(inter_cluster_distances) > 0:
            conciseness = min(inter_cluster_distances)
        else:
            conciseness = 0.0 # ?????
        return conciseness

    def _representativity_cluster(intra_cluster_distances):
        return np.mean(1.0 - np.asarray(intra_cluster_distances))
    
    def _representativity(intra_cluster_distances_per_agent: Dict[str, float], cluster_to_agents: List[List[str]], aggr=np.mean):
        disc_total = 0.0

        for aid, disc in intra_cluster_distances_per_agent.items():
            disc_total += disc
        disc_total /= len(intra_cluster_distances_per_agent)
        return 1- disc_total
        """else:
            for c in cluster_to_agents:
                if len(c) == 0:
                    continue
                intra_cluster_discordances = [intra_cluster_distances_per_agent[aid] for aid in c]
                disc_total += aggr(intra_cluster_discordances)
            return 1 - (disc_total / len(cluster_to_agents))"""
    @property
    def L(self):
        return sum(1 for c in self.assignments_vs if len(c) > 0) # TODO: take into account inter cluster distances?, if they are 0 they are actually the same cluster...

    def active_vs_clusters(self) -> List[Tuple[int, int]]:
        return [(i, len(self.assignments_vs[i])) for i in range(len(self.assignments_vs)) if len(self.assignments_vs[i]) > 0]
    """
    def active_gr_clusters(self):
        return [[(i, len(self.assignment_gr[value_i][i])) for i in range(len(self.assignment_gr[value_i])) if len(self.assignment_gr[value_i][i]) > 0] for value_i in range(self.n_values)]
    """
    @property
    def n_values(self):
        return len(self.get_value_system(0))
    @property
    def vs_score(self):

        self.certify_cluster_assignment_consistency()
        if self.inter_discordances_vs == float('inf') or self.L == 1:
            return self.representativity_vs(aggr=np.mean)
        else:
            return self.combined_cluster_score_vs()
    @property
    def gr_score(self):

        self.certify_cluster_assignment_consistency()
        # TODO: for now, it is the average (or other aggregation) on the intra scores of the value-based clusters (accuracies)
        return self.coherence()

    
    def representativities_gr(self):

        return [ClusterAssignment._representativity(self.intra_discordances_gr_per_agent[i], None) for i in range(self.n_values)]
    def representativity_gr_aggr(self):
        return self.aggregation_on_gr_scores(self.representativities_gr())

    def representativity_vs(self, aggr=np.mean):
        return ClusterAssignment._representativity(self.intra_discordances_vs_per_agent,None, aggr=aggr)

    def conciseness_vs(self):
        return ClusterAssignment._conciseness(self.inter_discordances_vs, self.L)
    def combined_cluster_score_vs(self,conciseness_if_L_is_1=None):
        return ClusterAssignment._combined_cluster_score(self.inter_discordances_vs, self.intra_discordances_vs_per_agent, self.L, self.assignments_vs,conciseness_if_1_cluster=conciseness_if_L_is_1)

    def coherence(self):
        return self.representativities_gr() # TODO FUTURE WORK aggregation of combined scores is this, or dividing the aggregation?
    def __str__(self):

        self.certify_cluster_assignment_consistency()
        result = "Cluster Assignment:\n"
        
        """
        result += "Grounding Clusters:\n"
        for vi, clusters in enumerate(self.assignment_gr):
            result += f"Value {vi}:\n"
            if self.K[vi] == 1:
                result += f"  Single GR Cluster: {[cix for cix in range(len(clusters)) if len(clusters[cix]) >0][0] } \n"
            else:
                for cluster_idx, agents in enumerate(clusters):
                    if len(agents) > 0:
                        result += f"  Cluster {cluster_idx}: {agents}\n"
                        """
        result += "\nValue System Clusters:\n"
        for cluster_idx, agents in enumerate(self.assignments_vs):
            if self.L == 1:
                result += f"  Single VS Cluster: {cluster_idx}\n"
            else:
                if len(agents) > 0:
                    result += f"  Cluster {cluster_idx} {self.get_value_system(cluster_idx=cluster_idx)}: {agents}\n"
        result += "\nScores:\n"
        try:
            #result += f"Representativities (Grounding): {self.representativities_gr()}\n"
            #result += f"Concisenesses (Grounding): {self.concisenesses_gr()}\n"
            result += f"Coherence (Grounding): {self.coherence()}\n"
            result += f"Representativity (Value System) MIN: {self.representativity_vs(aggr=np.min)}\n"
            result += f"Representativity (Value System) AVG: {self.representativity_vs(aggr=np.mean)}\n"
            #result += f"Representativity (Value System) GLOBAL: {self.representativity_vs(aggr='weighted')}\n"
            result += f"Conciseness (Value System): {self.conciseness_vs()}\n"
            result += f"Combined Score (Value System): {self.combined_cluster_score_vs()}\n"
        except TypeError:
            result += f"Not available\n"

        
        self.certify_cluster_assignment_consistency()
        return result

    def __repr__(self):
        return self.__str__()
    
    def is_equivalent_assignment(self, other: Self, exhaustive=False):

        self.certify_cluster_assignment_consistency()

        other.certify_cluster_assignment_consistency()
        l = self.L
        l_other = other.L
        #k_t = tuple(self.K)
        #k_t_other = tuple(other.K)
        if self.n_values != other.n_values:
            return False
        
        if l != l_other:
            return False
        """if k_t != k_t_other:
            return False"""
        
        if not exhaustive:
            if l == 1 and l_other == 1: # and k_t == tuple([1]*self.n_values) and k_t_other == tuple([1]*self.n_values):
                return True

        same_distribution = self.agent_distribution_vs() == other.agent_distribution_vs()
        if not same_distribution:
            print("Different agent distribution")
            return False

        self.certify_cluster_assignment_consistency()

        other.certify_cluster_assignment_consistency()

        if not exhaustive:
            return same_distribution
        else:
            if self.inter_discordances_vs is None and other.inter_discordances_vs is None:
                return same_distribution
            elif self.inter_discordances_vs is None:
                assert other.inter_discordances_vs is None, "Comparing unfairly exhaustively"
            
            same_conciseness = np.allclose(self.conciseness_vs(), other.conciseness_vs())
            same_representativeness = np.allclose(self.representativity_vs(aggr=np.mean), other.representativity_vs(aggr=np.mean))
            same_grounding = np.allclose(self.coherence(), other.coherence())
            same_n_training_steps = self.n_training_steps == other.n_training_steps
            if not same_n_training_steps:
                print("Different number of training steps")
            if not same_conciseness:
                print("Different conciseness")
            if not same_representativeness:
                print("Different representativeness")
            if not same_grounding:
                print("Different grounding")

            return same_distribution and same_conciseness and same_representativeness and same_grounding and same_n_training_steps

    def cluster_similarity(self, other: Self):

        self.certify_cluster_assignment_consistency()
        l = self.L 
        l_other = other.L
        if self == other:
            return 1000.0
        if self.n_values != other.n_values:
            return 0.0
        
        if l != l_other:
            return 0.0
        
        if l == 1 and l_other == 1:
            return 1.0
        if self.n_agents != other.n_agents:
            raise ValueError("Number of agents is different between the two cluster assignments. This needs a workaround.")
        
        # self.agent_distribution_vs() is a set of tuples. I want to use the edit distance to compare the two distributions
        total_differences = []

        a1 = self.agent_distribution_vs()
        a2 = other.agent_distribution_vs()
        min_total_edit_distance = float('inf')

        # Generate all possible permutations of clusters in a2
        a1_sorted = sorted(a1, key=len)
        a2_sorted = sorted(a2, key=len)
        min_total_edit_distance = sum(len(set(cluster1).symmetric_difference(set(cluster2))) for cluster1, cluster2 in zip(a1_sorted, a2_sorted)) 
        assert min_total_edit_distance <= 2*self.n_agents
        total_differences .append(min_total_edit_distance)
        """if not one_grounding:
            gr_dists = self.agent_distribution_gr()
            gr_dists_other = other.agent_distribution_gr()
            for a1, a2 in zip(gr_dists, gr_dists_other):
                a1 = self.agent_distribution_vs()
                a2 = other.agent_distribution_vs()
                raise ValueError("Do the trick from the value systems")
                min_total_edit_distance = float('inf')
                for perm in permutations(a2):
                    total_edit_distance = 0
                    for cluster1, cluster2 in zip(a1, perm):
                        total_edit_distance += len(set(cluster1).symmetric_difference(cluster2))
                min_total_edit_distance = min(min_total_edit_distance, total_edit_distance)
                total_differences .append(min_total_edit_distance)"""
        diff = np.mean(1.0 - np.array(total_differences)/ (2*self.n_agents) )
        """
        this is just an approximation... Might be differeif diff == 1.0:
            assert self.is_equivalent_assignment(other)
        else:
            assert not self.is_equivalent_assignment(other)
        """ 

        self.certify_cluster_assignment_consistency()
        return diff # TODO: separate grounding?
    
    def agent_distribution_vs(self) -> Set[Tuple]:
        dist = set([tuple(cluster) for cluster in self.assignments_vs])
        return dist

    def certify_cluster_assignment_consistency(self):
        with th.no_grad():
            if self.Lmax != len(self.assignments_vs):
                raise ValueError(f"Number of clusters {len(self.assignments_vs)} does not match Lmax {self.Lmax}.")
            assert self.L == len(self.active_vs_clusters())
            assert self.L > 0 and self.Lmax > 0, f"L {self.L} and Lmax {self.Lmax} must be greater than 0."
            assert self.L <= self.Lmax, f"L {self.L} must be less than or equal to Lmax {self.Lmax}."
            if self.inter_discordances_vs_per_cluster_pair is not None:
                assert self.intra_discordances_vs_per_agent is not None
                assert self.intra_discordances_gr_per_agent is not None
                assert self.inter_discordances_vs is not None

                if len(self._inter_discordances_vs_per_cluster_pair) > len(list(itertools.combinations(range(len(self.active_vs_clusters())), 2))):
                    raise ValueError(f"Different length different inter-cluster assignment {len(self._inter_discordances_vs_per_cluster_pair)}.")
                for c1, c2 in self._inter_discordances_vs_per_cluster_pair:
                    if c1 not in self.agent_to_vs_cluster_assignments.values() or c2 not in self.agent_to_vs_cluster_assignments.values():
                        raise ValueError(f"Cluster pair {c1}, {c2} is not assigned to any agent.")
                    if tuple(self.assignments_vs[c1]) == tuple(self.assignments_vs[c2]):
                        raise ValueError(f"Cluster pair {c1}, {c2} is assigned to same agents.")
                
                for d in self.intra_discordances_gr_per_agent:
                    assert len(d) == len(self.intra_discordances_gr_per_agent[0])
                for ag, disc in self.intra_discordances_vs_per_agent.items():
                    assert ag in self.agent_to_vs_cluster_assignments, f"Agent {ag} is not assigned to any cluster."
                    if disc != 0.0 or disc != 1.0:
                        assert self.agent_to_vs_cluster_assignments[ag] in [c[0] for c in self.active_vs_clusters()]
                disc = ClusterAssignment._intra_cluster_discordances(self.intra_discordances_gr_per_agent[0], self.assignments_vs)
                assert len(disc) == self.Lmax
                assert len([d for d in disc if d > -1.0]) == len(self.active_vs_clusters()), f"Discordances {disc} do not match active clusters {self.active_vs_clusters()}"
            for ag, clustid in self.agent_to_vs_cluster_assignments.items():
                if clustid >= len(self.assignments_vs):
                    raise ValueError(f"Agent {ag} assigned to cluster {clustid} which is out of bounds for the value system clusters {len(self.assignments_vs)}")
                if ag not in self.assignments_vs[clustid]:
                    raise ValueError(f"Agent {ag} not found in cluster {clustid} assignments {self.assignments_vs[clustid]}")
                if self.agent_to_vs_cluster_assignments.get(ag) != clustid:
                    raise ValueError(f"Agent {ag} is assigned to cluster {clustid} but has a different cluster assignment.")
                
            for clustid, agents in enumerate(self.assignments_vs):
                if len(agents) == 0:
                    for ag in self.agent_to_vs_cluster_assignments:
                        assert self.agent_to_vs_cluster_assignments.get(ag) != clustid, f"Cluster {clustid} is empty but has agents assigned to it."
                    #raise ValueError(f"Cluster {clustid} is empty.")
                elif clustid not in self.agent_to_vs_cluster_assignments.values():
                    raise ValueError(f"Cluster {clustid} is not assigned to any agent.")
                for ag in agents:
                    if ag not in self.agent_to_vs_cluster_assignments:
                        raise ValueError(f"Agent {ag} is not assigned to any cluster.")
                    if self.agent_to_vs_cluster_assignments.get(ag) not in list(range(len(self.assignments_vs))):
                        raise ValueError(f"Agent {ag} is assigned to a non-existent cluster.")
                    if ag not in self.assignments_vs[self.agent_to_vs_cluster_assignments.get(ag)]:
                        raise ValueError(f"Agent {ag} is not found in its assigned cluster.")

class ClusterAssignmentMemory():

    def __init__(self, max_size, n_values):
        self.max_size = max_size
        self.memory: List[ClusterAssignment] = []
        self.common_env = None
        self.maximum_conciseness_vs = float('-inf')
        self.maximum_grounding_coherence = [float('-inf') for _ in range(n_values)]
        self.initializing = True

    def __str__(self):
        
        self.sort_lexicographic(lexicographic_vs_first=True)
        result = "Cluster Assignment Memory:\n"
        mgr = self.maximum_conciseness_vs 

        for i, assignment in enumerate(self.memory):
            result += f"Assignment {i} (Explored: {assignment.explored}, {assignment.n_training_steps if hasattr(assignment, 'n_training_steps') else 'unk'}):"
            result += f" VS: RT={assignment.combined_cluster_score_vs(conciseness_if_L_is_1=mgr):1.4f}|RP={assignment.representativity_vs():.4f},RPav={assignment.representativity_vs(aggr=np.mean):.4f}|CN={assignment.conciseness_vs() if assignment.L > 1 else mgr:.4f}, GR: {[f"{float(g):.3f}" for g in assignment.coherence()]}, L: {assignment.L} \n"
            result += f" VS Clusters: {assignment.active_vs_clusters()}\n"
            result += f" OS: {assignment.optimizer_state}\n"
            result += f" Value Systems: {assignment.value_systems_active}\n"
            result += f" Value Systems (ALL): {assignment.value_systems}\n"
            result += "\n"
        return result

    
    def __len__(self):
        return len(self.memory)

    
    
    def compare_assignments(self, x: ClusterAssignment, y: ClusterAssignment, lexicographic_vs_first=False) -> float:

        # first on different grounding scores... then on value system scores.
        assert x.n_values == y.n_values
        assert x.n_values > 0

        mcvs = self.maximum_conciseness_vs 

        difs = []

        x_combined_per_value, y_combined_per_value = x.coherence(), y.coherence()

        for i in range(x.n_values):
            dif_gr_i = x_combined_per_value[i] - y_combined_per_value[i]
            difs.append(dif_gr_i)
            #assert not (has1 and hasmorethan1) # we need to come up with something here. For ECAI we have 1 grounding always, so no problem yet
        
        gr_score_dif = x.aggregation_on_gr_scores(difs) # TODO... maybe aggregation on scores should be modelled outside these two?
        
        #pareto
        vs_score_dif = - (x.combined_cluster_score_vs(conciseness_if_L_is_1=mcvs) - y.combined_cluster_score_vs(conciseness_if_L_is_1=mcvs))
        conc_proxy = (self.maximum_conciseness_vs  if self.maximum_conciseness_vs != float('-inf') else 0.0)
        conc_dif = (x.conciseness_vs() if x.L > 1 else conc_proxy) - (y.conciseness_vs() if y.L > 1 else conc_proxy)
        repr_dif = x.representativity_vs(aggr=np.mean) - y.representativity_vs(aggr=np.mean)
        l_diff = x.L - y.L
        #TODO: TEST PARETO TAKING INTO ACOUNT REPRESENTATIVITY TOO?
        
        pareto_score = 0.0
        lexic_diff = 0.0
        if (l_diff <= 0 and gr_score_dif > 0.0 and conc_dif >= 0.0 and repr_dif >=0) or (l_diff <= 0 and gr_score_dif >= 0.0 and conc_dif > 0.0 and repr_dif >=0) or (l_diff <= 0 and gr_score_dif >= 0.0 and conc_dif >= 0.0 and repr_dif > 0) or (l_diff < 0 and gr_score_dif >= 0.0 and conc_dif >= 0.0 and repr_dif >= 0):
                pareto_score = 1.0
        elif (l_diff >= 0 and gr_score_dif < 0.0 and conc_dif <= 0.0 and repr_dif <=0) or (l_diff >= 0 and gr_score_dif <= 0.0 and conc_dif < 0.0 and repr_dif <=0) or (l_diff >= 0 and gr_score_dif <= 0.0 and conc_dif <= 0.0 and repr_dif < 0) or (l_diff > 0 and gr_score_dif <= 0.0 and conc_dif <= 0.0 and repr_dif <= 0):
            pareto_score    = -1.0
        else:
            pareto_score = 0.0

        if lexicographic_vs_first:
                
                if abs(vs_score_dif) > 0.00: 
                        lexic_diff = vs_score_dif
                else:
                    lexic_diff = gr_score_dif
        else:
            if abs(gr_score_dif) > 0.00: 
                lexic_diff = gr_score_dif  
            else:
                lexic_diff = vs_score_dif
        return lexic_diff, pareto_score
       
    def notify_updated_assignment(self, assignment: ClusterAssignment):
        indeed_notified = False
        for i, p in enumerate(self.memory):
            if p == assignment:
                indeed_notified = True
        assert indeed_notified, f"Assignment {assignment} not found in memory. This should not happen."
        changes_made = self.update_maximum_conciseness(assignment)
        if assignment.L == 1:
            for i in reversed(range(len(self.memory))):
                if self.memory[i] == assignment:
                    continue
                elif self.memory[i].L == 1:
                    cmp_lexico, cmp_pareto = self.compare_assignments(self.memory[i], assignment,lexicographic_vs_first=False,) 
                    
                    if (cmp_pareto <=0 or cmp_lexico <0) and len([a for a in self.memory if a.L == 1]) > 1:
                        self.memory.pop(i)
                        
                    
            
        self.clean_memory(exhaustive=True, force_elimination=False)
        

    def insert_assignment(self, assignment: ClusterAssignment, sim_threshold=0.95) -> Tuple[int, ClusterAssignment]:

        
        """for a,b in itertools.combinations(self.memory, 2):
            assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."""
        for p in self.memory:
            assert p!=assignment, f"Assignment {assignment} already in memory."
        changes_made = self.update_maximum_conciseness(assignment)
        
        
        """for a,b in itertools.combinations(self.memory, 2):
            assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."""
        if all([asa.explored for asa in self.memory]) and len(self.memory) >= self.max_size:
            #self.clean_memory(exhaustive=True)  
            for i in range(len(self.memory)):  
                self.memory[i].explored = False
        
        l_assignment_1 = assignment.L == 1# if it is 1, need to have only one.
        override_and_insert = False

        if l_assignment_1:
            l1_exists = False
            for i in range(len(self.memory)):
                """for a,b in itertools.combinations(self.memory, 2):
                    assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."
                """
                if self.memory[i].L == 1:
                    cmp_lexico, cmp_pareto = self.compare_assignments(self.memory[i], assignment,lexicographic_vs_first=False,) 
                    l1_exists = True
                    """for a,b in itertools.combinations(self.memory, 2):
                        assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."
                    """
                    if self.memory[i].n_training_steps <= assignment.n_training_steps and (cmp_pareto <=0 or cmp_lexico <0):
                        self.memory[i] = assignment
                        self.memory[i].explored = False
                        break
                    
            if not l1_exists:
                """for a,b in itertools.combinations(self.memory, 2):
                    assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."
                """
                self.memory.append(assignment)
                """for a,b in itertools.combinations(self.memory, 2):
                    assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."
                """
        elif self.initializing:
            """for a,b in itertools.combinations(self.memory, 2):
                assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."
            """
            self.memory.append(assignment)
            if len(self.memory) == self.max_size:
                self.initializing = False
            for a,b in itertools.combinations(self.memory, 2):
                assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."
        else:  # general case.
            override_and_insert = False
            for a,b in itertools.combinations(self.memory, 2):
                assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."
            for override_index, a in enumerate(self.memory):
                cmp_lexico, cmp_pareto = self.compare_assignments(a, assignment,lexicographic_vs_first=False,)
                #self.memory[last_index].is_equivalent_assignment
                """for a,b in itertools.combinations(self.memory, 2):
                    assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."
                """
                if (cmp_pareto < 0) and assignment.is_equivalent_assignment(a) or a == assignment:
                    override_and_insert = True
                    break
                    
            if override_and_insert:
                #del self.memory[last_index]
                for p in self.memory:
                    assert p!=assignment, f"this should be it. Assignment {assignment} already in memory."
                self.memory[override_index] = assignment
                for a,b in itertools.combinations(self.memory, 2):
                    assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."
                self.memory[override_index].explored = False
            else:
                #if last_index is not None: self.memory[last_index].explored = True
                for p in self.memory:
                    assert p!=assignment, f"this should be it. Assignment {assignment} already in memory."
                self.memory.append(assignment)
                for a,b in itertools.combinations(self.memory, 2):
                    assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."
                self.memory[-1].explored = False
        """for a,b in itertools.combinations(self.memory, 2):
            assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."""
        if (len(self.memory) > self.max_size) or changes_made:
            self.clean_memory(exhaustive=False)
            print("Memory cleaned")
        """for a,b in itertools.combinations(self.memory, 2):
            assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."""
        return

    def update_maximum_conciseness(self, assignment: ClusterAssignment):

        gr_diffs = np.array(self.maximum_grounding_coherence) - np.array(assignment.gr_score)
        better_grounding_precondition = all(gr_diffs <= 0.0)
        if better_grounding_precondition:
            self.maximum_grounding_coherence = assignment.gr_score
        changes_made = False
        if better_grounding_precondition:
            if assignment.L > 1: 
                new_max_c = max(self.maximum_conciseness_vs, assignment.conciseness_vs()) 
                if new_max_c != self.maximum_conciseness_vs:
                    changes_made = True
                self.maximum_conciseness_vs = new_max_c
            
        
        return changes_made

    def clean_memory(self, exhaustive=False, sim_threshold=0.95, append_made=False, force_elimination=False):
        """for a,b in itertools.combinations(self.memory, 2):
            assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."""
        if len(self.memory) == 1:
            return
        self.sort_lexicographic(lexicographic_vs_first=False)
        self.memory[0].protected = True
        self.sort_lexicographic(lexicographic_vs_first=True)
        self.memory[0].protected = True

        pareto_dominated_counts = [0] * len(self.memory)
        equivalent_assignments_counts = [0] * len(self.memory)
        similarity_index = [0] * len(self.memory)

        grounding_score = [0] * len(self.memory)
        vs_score = [0] * len(self.memory)
        # Calculate pareto dominance and equivalence
        for i in reversed(list(range(len(self.memory)))):
            if self.memory[i].L == 1:
                pareto_dominated_counts[i] = 0
                equivalent_assignments_counts[i] = 0
                similarity_index[i] = 0
                continue
            
            grounding_score[i] = np.mean(self.memory[i].coherence())
            vs_score[i] = self.memory[i].combined_cluster_score_vs(conciseness_if_L_is_1=self.maximum_conciseness_vs)
            for j in range(len(self.memory)):
                if i != j:
                    _, cmp_pareto = self.compare_assignments(self.memory[j], self.memory[i], lexicographic_vs_first=True)
                    sim = self.memory[i].cluster_similarity(self.memory[j])
                    
                    similarity_index[i] += sim
                    if cmp_pareto > 0:
                        pareto_dominated_counts[i] += 1
                    #print("SIM", sim, i ,j)
                    if sim >= sim_threshold:
                        equivalent_assignments_counts[i] += 1 
                    
        # Remove the assignments most dominated among those that are 
        # If still too many examples:
        # Eliminate the one pareto dominated by the most others, or all if exhaustive (only at the end or under all examples explored)
        
        if (len(self.memory) > self.max_size) or (exhaustive and (max(pareto_dominated_counts) > 0 or sum(equivalent_assignments_counts) > 0)):
            
            sorted_indices = sorted(list(range(len(self.memory))), key=lambda x: (
                #int(explored_and_pareto_dominated[x]),
                #explored_and_pareto_dif[x],
                equivalent_assignments_counts[x], 
                similarity_index[x], 
                pareto_dominated_counts[x], 
                
                -grounding_score[x],
                vs_score[x]
                ), reverse=True)

            
            eliminated_index_in_sorted_indices = 0 
            # Special cases guards
            best_sorted_indices_by_grounding_then_vs = [i[0] for i in sorted(enumerate(self.memory), key=lambda x: ( 
            np.mean(x[1].coherence()),
            -x[1].combined_cluster_score_vs(conciseness_if_L_is_1=self.maximum_conciseness_vs),
                                                                                    ), reverse=True)]
            best_sorted_indices_by_vs_then_grounding = [i[0] for i in sorted(enumerate(self.memory), key=lambda x: ( 
                    
            -x[1].combined_cluster_score_vs(conciseness_if_L_is_1=self.maximum_conciseness_vs),
            np.mean(x[1].coherence()),
                                                                                    ), reverse=True)]
        
            
            eliminated_index_in_sorted_indices = 0
            while eliminated_index_in_sorted_indices < len(self.memory):
                eliminated_index = sorted_indices[eliminated_index_in_sorted_indices]
                c1 = (self.memory[eliminated_index].L == 1) and len([0 for m in self.memory if m.L == 1]) == 1
                c2 = best_sorted_indices_by_grounding_then_vs[0] == eliminated_index
                c3 = best_sorted_indices_by_vs_then_grounding[0] == eliminated_index
                c4 = self.memory[eliminated_index].protected

                if c1 or c2 or c3 or c4:
                    eliminated_index_in_sorted_indices+=1
                else: 
                    break
            
            eliminated_index = sorted_indices[eliminated_index_in_sorted_indices % len(sorted_indices)]
            should_be_out_criterion = eliminated_index_in_sorted_indices < len(self.memory) and (equivalent_assignments_counts[eliminated_index] > 0 or pareto_dominated_counts[eliminated_index] > 0) and similarity_index[eliminated_index] != min(similarity_index)
            force_elimination_now = force_elimination or (len(self.memory) > self.max_size)
            #print("Force now", force_elimination_now, len(self.memory), self.max_size, eliminated_index_in_sorted_indices, len(sorted_indices))
            if (should_be_out_criterion and exhaustive) or force_elimination_now :
                self.memory.pop(eliminated_index)
                print("eliminated", eliminated_index, "due to final rule")
            
                if (exhaustive or (len(self.memory) > self.max_size)):
                    self.clean_memory(exhaustive=exhaustive,sim_threshold=sim_threshold, append_made=False, force_elimination=False)
        """for a,b in itertools.combinations(self.memory, 2):
            assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."""
        for p in self.memory:
            p.protected = False 
        return
        
    def sort_lexicographic(self, lexicographic_vs_first=False):
        """for a,b in itertools.combinations(self.memory, 2):
            assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."""
        sorted_memory_with_indices = sorted(enumerate(self.memory), key=lambda x: cmp_to_key(lambda a, b: self.compare_assignments(a, b, lexicographic_vs_first=lexicographic_vs_first)[0])(x[1]), reverse=True)
        self.memory = [item[1] for item in sorted_memory_with_indices]
        new_indices = [item[0] for item in sorted_memory_with_indices]
        """for a,b in itertools.combinations(self.memory, 2):
            assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."""
        return new_indices  
    def get_random_weighted_assignment(self, consider_only_unexplored=False, lexicographic_vs_first=False)-> ClusterAssignment:
        """for a,b in itertools.combinations(self.memory, 2):
            assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."""
        for p in self.memory:
            p.protected = False
        self.sort_lexicographic(lexicographic_vs_first=not lexicographic_vs_first)
        self.memory[0].protected = True
        self.sort_lexicographic(lexicographic_vs_first=lexicographic_vs_first)
        self.memory[0].protected = True

        assert len([p for p in self.memory if p.protected]) <= 2, "There should be at most two protected assignments in the memory."
        assert len([p for p in self.memory if p.protected]) > 0, "There should be at least one protected assignment in the memory."
        if consider_only_unexplored:
            indices_selectable = [i for i, assignment in enumerate(self.memory) if not assignment.explored]
        else:
            indices_selectable = [i for i in range(len(self.memory))]
            
        if len(indices_selectable) == 0:
            self.clean_memory(exhaustive=False)
            for i, assignment in enumerate(self.memory):
                assignment.explored = False
            return self.memory[0], True
        n = len(indices_selectable)
        weights = [2*(n-i)/(n*(n+1)) for i in range(n)] # Linear rank selection Goldberg
        #print(weights)
        
        assignment_index = random.choices(indices_selectable, weights=weights, k=1)[0]
        assignment = self.memory[assignment_index]
        print(assignment.coherence())
        print("Selected index", assignment_index,weights, indices_selectable)
        
        protected = bool(assignment.protected) and len(self.memory) > 1
        assignment.protected = False
        """for a,b in itertools.combinations(self.memory, 2):
            assert a != b, f"Assignments {a} and {b} are the same in memory. This should not happen."""
        return assignment, protected

    def get_best_assignment(self, consider_only_unexplored=False, lexicographic_vs_first=True) -> Tuple[ClusterAssignment, bool]:
        self.sort_lexicographic(lexicographic_vs_first=lexicographic_vs_first)
        indices_non_explored = [i for i, assignment in enumerate(self.memory) if not assignment.explored]
        if not consider_only_unexplored:
            return self.memory[0], True
        if len(indices_non_explored) == 0:
            self.clean_memory(exhaustive=False)
            for i, assignment in enumerate(self.memory):
                assignment.explored = False
            return self.memory[0], True
        
        assignment = self.memory[indices_non_explored[0]]
        assignment.certify_cluster_assignment_consistency()
        return assignment, True
        
    
    

def generate_mutated_assignment(reference_assignment: ClusterAssignment,
                                    mutation_scale=5.0,
                                    mutation_prob=0.1):
        
        with th.no_grad():
            grounding_deviation = 1.0-np.mean(reference_assignment.gr_score)
            value_system_deviation = 2.0-reference_assignment.representativity_vs(aggr=np.min)-reference_assignment.conciseness_vs()

            new_grounding = reference_assignment.grounding.copy()
            value_system_per_cluster_c = [reference_assignment.weights_per_cluster[i].copy() for i in range(len(reference_assignment.weights_per_cluster))]
            
            

            assignment_vs_new = [list() for _ in range(len(value_system_per_cluster_c))]
            l_prev = reference_assignment.L

            max_clusters = len(reference_assignment.weights_per_cluster)

            if l_prev == 1:
                expand_or_reduce = np.random.choice(['expand', 'same_l'])
            elif l_prev == max_clusters:
                expand_or_reduce = np.random.choice(['reduce', 'same_l'])
            else:
                expand_or_reduce = np.random.choice(['expand', 'reduce', 'same_l'])

            if len(value_system_per_cluster_c) >1:
                valid_new_clusters = [a[0] for a in reference_assignment.active_vs_clusters()]
                if expand_or_reduce == 'expand':
                    valid_possible_new_clusters = [i for i in range(max_clusters) if i not in valid_new_clusters]
                    cs = np.random.choice(valid_possible_new_clusters, size=np.random.choice(np.arange(1, len(valid_possible_new_clusters)+1)), replace=False)
                    valid_new_clusters.extend(cs)
                elif expand_or_reduce == 'reduce':
                    cs = np.random.choice(np.arange(len(valid_new_clusters)), size=np.random.choice(np.arange(1, len(valid_new_clusters))), replace=False)
                    #print(cs, valid_new_clusters)
                    for c in sorted(cs, reverse=True):
                        valid_new_clusters.pop(c)
                else:
                    pass # the valid clusters are the ones we already had
            else:
                valid_new_clusters = [0]
            flag = False
            for param in new_grounding.parameters():
                if param.requires_grad:
                    assert not th.any(param.isinf()) and not th.any(param.isnan())
                    param: th.Tensor
                    #mask = th.rand_like(param) < mutation_prob  # percentage of parameters changed.
                    normal = th.empty_like(param).normal_(0, mutation_scale*grounding_deviation*min(th.norm(param), 100.0))
                    #param.add_(th.where(mask, normal, th.zeros_like(param)))
                    param.add_(normal)
                    assert not th.any(param.isinf()) and not th.any(param.isnan())
            seen_clusters = set()
            agent_to_vs_cluster_assignments = {}
            for aid, cluster_vs_aid in reference_assignment.agent_to_vs_cluster_assignments.items():
                
                if random.random() > mutation_prob:
                    if int(cluster_vs_aid) not in valid_new_clusters:
                        cs = np.random.choice(valid_new_clusters)
                    else:
                        cs = cluster_vs_aid
                    assignment_vs_new[cs].append(aid)
                    agent_to_vs_cluster_assignments[aid] = cs    
                else:
                    cs = np.random.choice(valid_new_clusters)
                    assignment_vs_new[cs].append(aid)
                    agent_to_vs_cluster_assignments[aid] = cs

                if cs not in seen_clusters:
                    seen_clusters.add(cs)
                    flag = False
                    for param in value_system_per_cluster_c[cs].parameters():
                        if param.requires_grad:
                            param: th.Tensor
                            #mask = th.rand_like(param) < mutation_prob  # percentage of parameters changed.
                            normal = th.empty_like(param).normal_(0, mutation_scale*value_system_deviation*min(th.norm(param), 100.0))
                                    #param.add_(th.where(mask, normal, th.zeros_like(param)))
                            param.add_(normal)
                            assert not th.any(param.isinf()) and not th.any(param.isnan())
                            flag = True
            if not flag:
                raise ValueError("Should have changed value systems...")
            
            new_assignment = ClusterAssignment(weights_per_cluster=value_system_per_cluster_c, 
                                                     grounding=new_grounding,
                                                     agent_to_vs_cluster_assignments=agent_to_vs_cluster_assignments,
                                                     assignments_vs=assignment_vs_new,
                                                     aggregation_on_gr_scores=reference_assignment.aggregation_on_gr_scores)
            new_assignment.n_training_steps = int(reference_assignment.n_training_steps)
            new_assignment.explored = False
            new_assignment.protected = False
            new_assignment.optimizer_state = deepcopy(reference_assignment.optimizer_state)
                                                    
        return new_assignment


        
    