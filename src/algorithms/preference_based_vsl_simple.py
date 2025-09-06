from copy import deepcopy
from functools import cmp_to_key, partial
import math
import os
import pprint
import random
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Self, Set, Tuple, override
from morl_baselines.common.networks import get_grad_norm
from colorama import Style
import dill
import numpy as np
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from ordered_set import OrderedSet
import wandb
from defines import CHECKPOINTS, transform_weights_to_tuple
from src.algorithms.clustering_utils_simple import COLOR_PALETTE, assign_colors_colorama, generate_mutated_assignment
from src.algorithms.clustering_utils_simple import ClusterAssignment, ClusterAssignmentMemory, check_grounding_value_system_networks_consistency_with_optim, generate_random_assignment

from src.algorithms.preference_based_vsl_lib import BaseVSLClusterRewardLoss, VSLCustomLoss, VSLOptimizer, likelihood_x_is_target, probability_BT, probs_to_label
from src.dataset_processing.data import FixedLengthVSLPreferenceDataset, TrajectoryWithValueSystemRews, VSLPreferenceDataset
from src.policies.morl_custom_reward import CustomRewardReplayBuffer, EnvelopePBMORL, MORecordEpisodeStatisticsCR
from src.policies.mushroom_agent_mobaselines import MOBaselinesAgent
from src.reward_nets.vsl_reward_functions import ConvexAlignmentLayer, EnsembleRewardVectorModule, RewardVectorModule, VectorModule, LinearAlignmentLayer, TrainingModes, create_alignment_layer
import torch as th

from imitation.util.networks import RunningNorm
from imitation.util import util

from src.utils import most_recent_indices_to_ptr
from utils import visualize_pareto_front

from baraacuda.utils.miscellaneous import get_signature_params


from morl_baselines.common.utils import linearly_decaying_value
from morl_baselines.common.networks import polyak_update
from morl_baselines.common.weights import random_weights
from colorama import Fore, Style

def parse_or_invent_alignment_function_from_name(agname, rdim: int):
                    # Implement the parsing logic here
                    nl = eval(agname.split('_')[-1])
                    if isinstance(nl, int):
                        nl = transform_weights_to_tuple(np.full((rdim,), 1.0/float(rdim)))
                    return nl
class PVSL(object):
    def _clustered_pbmorl_reward_train_callback(self, base_dataset, 
                                      running_dataset, 
                                      reward_vector_function, 
                                      lexicographic_vs_first,
                                      initial_reward_learning_iterations,
                                      em_cycles_per_iteration, 
                                      m_steps_per_cycle, 
                                      batch_size_per_agent, 
                                      global_iter, 
                                      max_iter,
                                      initial_exploration_rate,
                                      mutation_prob, mutation_scale, 
                                      best_assignments_list: ClusterAssignmentMemory,initial_m_steps_per_cycle, 
                                      batch_size_in_running_dataset_for_cluster_assignment=0,
                                      continue_with_best_cluster=False ,
                                      disable_selection_after_initial_reward_iterations=False,
                                      use_running_dataset_in_assignment=False):
                assert batch_size_per_agent is not None
                assert reward_vector_function == self.train_reward_net
                condition_using_base = True
                condition_avoid_selection = False
                if running_dataset is not None:
                    condition_using_base = global_iter == 0 and (self.mobaselines_agent.agent_mobaselines.global_step <= self.mobaselines_agent.agent_mobaselines.learning_starts)
                    initial_reward_learning_iterations = initial_reward_learning_iterations if condition_using_base else 1
                
                    if condition_using_base:
                        print(f"{Fore.GREEN}USING BASE, {self.mobaselines_agent.agent_mobaselines.global_step}/{self.mobaselines_agent.agent_mobaselines.learning_starts}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}USING REPLAY, {self.mobaselines_agent.agent_mobaselines.global_step}/{self.mobaselines_agent.agent_mobaselines.learning_starts}{Style.RESET_ALL}")
                    train_set = base_dataset if condition_using_base else running_dataset
                else:
                    initial_reward_learning_iterations = 1
                    train_set = base_dataset
                val_set = base_dataset
                

                if isinstance(initial_reward_learning_iterations, str):
                    _, accuracy_needed = initial_reward_learning_iterations.split(':')
                    accuracy_needed = float(accuracy_needed)
                    initial_reward_learning_iterations = 10000
                else:
                    accuracy_needed = 1.0

                with tqdm(total=initial_reward_learning_iterations, desc="Initial Reward Learning Iterations",
                      dynamic_ncols=True, disable=False,
                      leave=False) as t_iterations:
                
                    for it in range(initial_reward_learning_iterations):
                        condition_avoid_selection = disable_selection_after_initial_reward_iterations and not condition_using_base
                        
                        if condition_avoid_selection:
                            if not hasattr(self, '_last_avoid_selection'):
                                self._last_avoid_selection = True
                            mutated = False
                            is_protected = False
                            if self._last_avoid_selection:
                                self._last_avoid_selection = False
                                original = best_assignments_list.get_best_assignment(lexicographic_vs_first=False)[0]
                                while len(best_assignments_list.memory) > 1:
                                    candidate = best_assignments_list.memory[-1]
                                    if candidate != original:
                                        best_assignments_list.memory.pop(-1)

                            else:
                                original = self.current_assignment
                            self.current_assignment = original
                            self.update_training_networks_from_assignment(self.value_system_per_cluster, self.train_reward_net, self.current_assignment, only_grounding=self.static_weights)
                            #print(self.current_assignment.value_systems)
                            #exit()
                        else:
                            #print("DOING THIS")
                            self.current_assignment, original, mutated, is_protected = self.selection(best_assignments_list, 
                                                                                                lexicographic_vs_first=lexicographic_vs_first, 
                                                                                                num_iterations=max_iter, 
                                                                                                iter_now=global_iter,
                                                                                                initial_exploration_rate=initial_exploration_rate, mutation_prob=mutation_prob, mutation_scale=mutation_scale, dataset=base_dataset)
                            #print(f"Selected assignment: {self.current_assignment} (mutated: {mutated}, protected: {is_protected}) from {len(best_assignments_list.memory)} candidates.")
                            #input()

                        if not is_protected and not mutated:
                            assert self.current_assignment == original, f"Mut? {mutated} Current assignment {self.current_assignment} is not equal to original {original}"
                        
                    
                        #train_dataset = dataset
                        #val_dataset = dataset
                        ##self.current_assignment.certify_cluster_assignment_consistency()
                        if self.static_weights:
                            print("Using static weights: (SHOULD BE)", self.get_value_systems())
                        if use_running_dataset_in_assignment:
                            running_dataset_in_assignment = None if condition_using_base else running_dataset if condition_avoid_selection else None
                        else:
                            running_dataset_in_assignment = None
                        """
                        assert not use_running_dataset_in_assignment and initial_reward_learning_iterations == 1
                        print("RUNNING DATASET IN ASSIGNMENT:", running_dataset_in_assignment, global_iter)
                        print(not condition_avoid_selection or not disable_selection_after_initial_reward_iterations)

                        input()"""
                        if not condition_avoid_selection or not disable_selection_after_initial_reward_iterations:
                            self.update_training_networks_from_assignment(self.value_system_per_cluster, self.train_reward_net, self.current_assignment, only_grounding=self.static_weights)
                        if self.static_weights:
                            self.loss.set_parameters(params_gr=self.optim.params_gr, params_vs=set(), optim_state=self.optim.get_state())
                            self.current_assignment.weights_per_cluster = self.value_system_per_cluster
                            assert np.allclose(np.array(self.current_assignment.value_systems), np.array(self.get_value_systems())), f"Static weights mismatch: {self.current_assignment.value_systems} vs {self.get_value_systems()}"
                            assert np.allclose(np.array(self.current_assignment.value_systems), self.mobaselines_agent.agent_mobaselines.sample_eval_weights(self.Lmax)), f"Static weights mismatch: {self.current_assignment.value_systems} vs {self.mobaselines_agent.agent_mobaselines.sample_eval_weights(self.Lmax)}"
                        self.set_mode('train')
                        ##self.current_assignment.certify_cluster_assignment_consistency()
                        for cycle in range(em_cycles_per_iteration):
                            steps_now = initial_m_steps_per_cycle if cycle == 0 else m_steps_per_cycle
                            #self.current_assignment.certify_cluster_assignment_consistency()
                            if ((not mutated) or cycle >= 1) or (condition_using_base and disable_selection_after_initial_reward_iterations) or condition_avoid_selection: #and (self.Lmax > 1 or global_iter == 0):
                                self.cluster_assignment(base_dataset, running_dataset=running_dataset_in_assignment, batch_size_in_running_dataset_for_cluster_assignment=batch_size_in_running_dataset_for_cluster_assignment, qualitative_cluster_assignment=False) # E-step
                            #self.current_assignment.certify_cluster_assignment_consistency()
                            train_losses = self.train_reward_models(train_set, global_step=self.global_step,
                                                    n_optim_steps=steps_now, batch_size_per_agent=batch_size_per_agent, transitions=False) # M-step
                            if self.static_weights:
                                assert np.allclose(np.array(self.current_assignment.value_systems), self.mobaselines_agent.agent_mobaselines.sample_eval_weights(self.Lmax)), f"Static weights mismatch: {self.current_assignment.value_systems} vs {self.get_value_systems()}"
                            #self.current_assignment.certify_cluster_assignment_consistency()
                            if cycle == em_cycles_per_iteration - 1:
                                if self.use_wandb:
                                    self.wandb_log_losses(global_iter, train_losses)
                            with th.no_grad():
                                self.current_assignment.n_training_steps += steps_now
                                self.current_assignment.optimizer_state = self.optim.get_state(copy=True)
                                original.optimizer_state = self.optim.get_state(copy=True)
                                self.global_step += steps_now
                                #self.current_assignment.certify_cluster_assignment_consistency()
                                
                        self.set_mode('eval')
                        with th.no_grad():
                            #self.current_assignment.weights_per_cluster = deepcopy(self.value_system_per_cluster)
                            ##self.current_assignment.grounding = deepcopy(self.train_reward_net)
                            self.current_assignment.recalculate_discordances(val_set, indifference_tolerance=self.loss.model_indifference_tolerance)
                            #self.current_assignment.certify_cluster_assignment_consistency()
                            if mutated or is_protected:
                                    best_assignments_list.insert_assignment(self.current_assignment)
                            else:
                                    best_assignments_list.notify_updated_assignment(self.current_assignment)
                            if self.debug_mode:
                                print(best_assignments_list)
                            if min(self.current_assignment.coherence()) >= accuracy_needed:
                                if self.current_assignment.representativity_vs() >= accuracy_needed:
                                    print(f"Early stopping at iteration {it} with coherence {self.current_assignment.coherence()} and representativity {self.current_assignment.representativity_vs()}")
                                    t_iterations.update(it+1)
                                    break

                        t_iterations.update(1)
                    
                if continue_with_best_cluster and not condition_avoid_selection:
                    self.current_assignment = best_assignments_list.get_best_assignment(lexicographic_vs_first=False)[0].copy()
                    self.current_assignment.recalculate_discordances(val_set, indifference_tolerance=self.loss.model_indifference_tolerance)
                    self.current_assignment.certify_cluster_assignment_consistency()
                    self.update_training_networks_from_assignment(self.value_system_per_cluster, self.train_reward_net, self.current_assignment, only_grounding=self.static_weights)
                    self.current_assignment.grounding = self.train_reward_net
                    self.current_assignment.weights_per_cluster = self.value_system_per_cluster
                self.set_mode('eval')
                return {'reward_net': self.train_reward_net, 'running_assignment': self.current_assignment, 'global_iter': global_iter+initial_reward_learning_iterations}
    def _pbmorl_reward_train_callback(self, base_dataset, 
                                      running_dataset, 
                                      reward_vector_function, m_steps_per_cycle, batch_size_reward_buffer, global_iter, best_assignments_list):
                condition = self.mobaselines_agent.agent_mobaselines.global_step < self.mobaselines_agent.agent_mobaselines.learning_starts
                if condition:
                    print(f"{Fore.GREEN}USING BASE, {self.mobaselines_agent.agent_mobaselines.global_step}/{self.mobaselines_agent.agent_mobaselines.learning_starts}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}USING REPLAY, {self.mobaselines_agent.agent_mobaselines.global_step}/{self.mobaselines_agent.agent_mobaselines.learning_starts}{Style.RESET_ALL}")
                train_set = base_dataset if condition else running_dataset
                val_set = base_dataset
                
                self.current_assignment.grounding = self.train_reward_net
                self.update_training_networks_from_assignment(self.value_system_per_cluster, self.train_reward_net, self.current_assignment, only_grounding=True)
                assert len(self.value_system_per_cluster) == self.current_assignment.Lmax, f"Value systems length mismatch: {len(self.value_system_per_cluster)} vs {self.current_assignment.Lmax}"
                
                
                assert isinstance(train_set, VSLPreferenceDataset), "Train set must be a VSLPreferenceDataset"
                assert isinstance(val_set, VSLPreferenceDataset), "Validation set must be a VSLPreferenceDataset"
                assert self.train_reward_net == reward_vector_function
                self.set_mode('train')
                # "sample minibatch", with their own weights...
                assert len(self.value_system_per_cluster) == self.current_assignment.Lmax, f"Value systems length mismatch: {len(self.value_system_per_cluster)} vs {self.current_assignment.Lmax}"
            
                #self.current_assignment.certify_cluster_assignment_consistency()
                assert all(p.requires_grad for p in self.optim.params_gr)
                #prev_params = deepcopy(self.optim.params_gr)
                check_grounding_value_system_networks_consistency_with_optim(self.train_reward_net, self.value_system_per_cluster, self.optim, only_grounding=True)
                train_losses = self.train_reward_models_no_clusters(train_set, 
                                                                    global_step=self.global_step,
                                                                    n_optim_steps=m_steps_per_cycle, 
                                                                    
                                                                    batch_size_reward_buffer=batch_size_reward_buffer) # M-step
                #new_params = self.optim.params_gr
                #ch = []
                #for p, np in zip(prev_params, new_params):
                #    ch.append(th.allclose(p, np))
                #print(ch)
                #assert not all(ch), "Optimizer parameters should change after training"
                if self.use_wandb:
                    self.wandb_log_losses(global_iter, train_losses)
                
                self.global_step += m_steps_per_cycle
                assert len(self.value_system_per_cluster) == self.current_assignment.Lmax, f"Value systems length mismatch: {len(self.value_system_per_cluster)} vs {self.current_assignment.Lmax}"
                
                if self.global_step % 10*m_steps_per_cycle == 0:
                    self.current_assignment = self.cluster_assignment(val_set, qualitative_cluster_assignment=False)
                    self.current_assignment.n_training_steps += m_steps_per_cycle
                    self.current_assignment.optimizer_state = self.optim.get_state()
                    self.current_assignment.recalculate_discordances(dataset=base_dataset, indifference_tolerance=self.loss.model_indifference_tolerance)
                    print("At iteration", global_iter, "cluster assignment:", self.current_assignment)
                    self.current_assignment.certify_cluster_assignment_consistency()
                
                best_assignments_list.insert_assignment(self.current_assignment.copy())
                return {'reward_net': self.train_reward_net}
    @staticmethod
    def load_from_state(best_assignment_list, historic, policy, global_step, config):
        last_assignment: ClusterAssignment = historic[-1]
        instance = PVSL(Lmax=last_assignment.Lmax,
                        mobaselines_agent=policy,
                        alignment_layer_class=config['alignment_layer_class'],
                        alignment_layer_kwargs=config['alignment_layer_kwargs'],
                        loss_class= config['loss_class'],
                        loss_kwargs= config['loss_kwargs'],
                        optim_class= config['optim_class'],
                        optim_kwargs= config['optim_kwargs'],
                        grounding_network=last_assignment.grounding,
                        reward_kwargs_for_config= config['reward_kwargs'],
                        n_rewards_for_ensemble= config['n_rewards_for_ensemble'],
                        debug_mode=config['debug_mode'],
                        use_wandb=config['use_wandb']
        )
        # Load the state from the provided parameters
        instance.historic = historic
        instance.mobaselines_agent = policy
        instance.global_step = global_step
        instance.current_assignment = last_assignment
        instance.device = instance.grounding_network.device
        instance.global_step = global_step

        instance.current_assignment: ClusterAssignment = historic[-1] # type: ignore
        instance.Lmax = instance.current_assignment.Lmax
        instance.update_training_networks_from_assignment(instance.value_system_per_cluster, instance.train_reward_net, instance.current_assignment, only_grounding=False)

        return instance

    def __init__(self, Lmax, mobaselines_agent: MOBaselinesAgent,  grounding_network: RewardVectorModule, alignment_layer_class=ConvexAlignmentLayer, alignment_layer_kwargs: Dict[str, Any] = {},
                  loss_class: BaseVSLClusterRewardLoss=VSLCustomLoss, n_rewards_for_ensemble=1, loss_kwargs: Dict[str, Any] = {}, reward_kwargs_for_config={}, optim_class=VSLOptimizer, optim_kwargs: Dict[str, Any] = {},
                   debug_mode=True,
                   use_wandb=False):
        self.use_wandb = use_wandb
        self.Lmax = Lmax
        self.mobaselines_agent = mobaselines_agent
        self.alignment_layer_class = alignment_layer_class
        self.alignment_layer_kwargs = alignment_layer_kwargs
        
        self.value_system_per_cluster = [create_alignment_layer(None, alignment_layer_class, alignment_layer_kwargs) for _ in range(Lmax)]
        self.cluster_colors_vs = dict()
        self.reward_kwargs_for_config = reward_kwargs_for_config

        self.loss: VSLCustomLoss = loss_class(**loss_kwargs)
        self.loss_kwargs= loss_kwargs
        self.loss_class = loss_class
        self.n_rewards_for_ensemble = n_rewards_for_ensemble
        
        if isinstance(grounding_network, RewardVectorModule):
            self.grounding_network = grounding_network
            assert isinstance(self.grounding_network, RewardVectorModule), "Grounding network must be an instance of RewardVectorModule"
        
            self.train_reward_net = EnsembleRewardVectorModule([self.grounding_network.copy_new() for _ in range(n_rewards_for_ensemble)])
        else:
            assert isinstance(grounding_network, EnsembleRewardVectorModule), "Grounding network must be an instance of EnsembleRewardVectorModule"
            self.train_reward_net = grounding_network
            grounding_network: EnsembleRewardVectorModule
            assert grounding_network.n_models == n_rewards_for_ensemble
            self.grounding_network = self.train_reward_net.rewards[0]  # Use the first reward vector as the grounding network referent
        self.optim: VSLOptimizer = optim_class( **optim_kwargs, params_gr=set(self.train_reward_net.parameters()), params_vs=set(p for v in self.value_system_per_cluster for p in v.parameters()), n_values=self.grounding_network.num_outputs)
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs


        self.debug_mode = debug_mode
        self.static_weights = False

    def update_assignment_from_trained_networks(self, reference_assignment: ClusterAssignment, trained_reward_net, weights, only_grounding: bool = False):
        if not only_grounding or reference_assignment.Lmax != len(weights):
                for c in range(len(reference_assignment.weights_per_cluster)):
                    reference_assignment.weights_per_cluster[c].load_state_dict(deepcopy(reference_assignment.weights_per_cluster[c].state_dict()))
        else:
            reference_assignment.weights_per_cluster =  weights
        if only_grounding:
            assert set(reference_assignment.value_systems) == set(self.get_value_systems()), f"Static weights mismatch: {reference_assignment.weights_per_cluster} vs {self.value_system_per_cluster}"
        assert len(weights) == reference_assignment.Lmax, f"Value systems length mismatch: {len(weights)} vs {reference_assignment.Lmax}"
        assert trained_reward_net.num_outputs == reference_assignment.grounding.num_outputs, f"Grounding network outputs mismatch: {trained_reward_net.num_outputs} vs {reference_assignment.grounding.num_outputs}"
        assert trained_reward_net.__class__.__name__ == reference_assignment.grounding.__class__.__name__, f"Grounding network class mismatch: {trained_reward_net.__class__.__name__} vs {reference_assignment.grounding.__class__.__name__}"
        if isinstance(reference_assignment.grounding, EnsembleRewardVectorModule):
            assert isinstance(reference_assignment.grounding, EnsembleRewardVectorModule)
            assert len(trained_reward_net.rewards) == len(reference_assignment.grounding.rewards), f"Grounding network models length mismatch: {len(trained_reward_net.rewards)} vs {len(reference_assignment.grounding.rewards)}"


        reference_assignment.grounding.load_state_dict(deepcopy(trained_reward_net.state_dict()))
        reference_assignment.grounding.requires_grad_(False)
        for w in reference_assignment.weights_per_cluster:
            w.requires_grad_(False)

    def update_training_networks_from_assignment(self, value_system_per_cluster: List[LinearAlignmentLayer], grounding: VectorModule, reference_assignment: ClusterAssignment, only_grounding: bool = False):

        with th.no_grad():
            if len(value_system_per_cluster) < reference_assignment.Lmax:
                value_system_per_cluster.extend([deepcopy(reference_assignment.weights_per_cluster[0]) for _ in range(reference_assignment.Lmax - len(value_system_per_cluster))])
               
            elif len(value_system_per_cluster) > reference_assignment.Lmax:
                n = len(value_system_per_cluster)
                for i in range(n - reference_assignment.Lmax):
                    value_system_per_cluster.pop(-1)
            
            if not only_grounding or reference_assignment.Lmax != len(self.value_system_per_cluster):
                for c in range(len(reference_assignment.weights_per_cluster)):
                    value_system_per_cluster[c].load_state_dict(deepcopy(reference_assignment.weights_per_cluster[c].state_dict()))
            else:
                value_system_per_cluster = self.value_system_per_cluster

            assert len(value_system_per_cluster) == reference_assignment.Lmax, f"Value systems length mismatch: {len(value_system_per_cluster)} vs {reference_assignment.Lmax}"
            assert grounding.num_outputs == reference_assignment.grounding.num_outputs, f"Grounding network outputs mismatch: {grounding.num_outputs} vs {reference_assignment.grounding.num_outputs}"
            assert grounding.__class__.__name__ == reference_assignment.grounding.__class__.__name__, f"Grounding network class mismatch: {grounding.__class__.__name__} vs {reference_assignment.grounding.__class__.__name__}"
            if isinstance(grounding, EnsembleRewardVectorModule):
                assert isinstance(reference_assignment.grounding, EnsembleRewardVectorModule)
                assert len(grounding.rewards) == len(reference_assignment.grounding.rewards), f"Grounding network models length mismatch: {len(grounding.rewards)} vs {len(reference_assignment.grounding.rewards)}"
                
           
            grounding.load_state_dict(deepcopy(reference_assignment.grounding.state_dict()))
            
           
            
            if isinstance(self.loss, VSLCustomLoss):
                assert isinstance(self.optim, VSLOptimizer)

                
                
                self.optim.params_gr = {param for param in grounding.parameters()}
                if only_grounding:
                    self.optim.params_vs = set()
                else:
                    self.optim.params_vs = {param for al in value_system_per_cluster for param in al.parameters()}

                if self.debug_mode:
                    should_be_wx = {param for param in grounding.parameters()}

                    assert should_be_wx.issubset(self.optim.params_gr), "Mismatch in wx parameters"
                    assert self.optim.params_gr == {param for param in grounding.parameters()}, "Mismatch in wx parameters"
                    
                    if not only_grounding:
                        should_be_wy = {param for al in value_system_per_cluster for param in al.parameters()}
                        assert should_be_wy.issubset(self.optim.params_vs), "Mismatch in wy parameters"
                    
                    
                        assert self.optim.params_vs == {param for network in value_system_per_cluster for param in network.parameters()}, "Mismatch in wy parameters"
                self.optim: VSLOptimizer = self.optim_class( **self.optim_kwargs, params_gr=self.optim.params_gr, params_vs=self.optim.params_vs, n_values=self.grounding_network.num_outputs)
                self.optim.set_state(reference_assignment.optimizer_state)
                self.loss.set_parameters(params_gr=self.optim.params_gr, params_vs=self.optim.params_vs, optim_state=self.optim.get_state())
            else:
                raise SystemError("Loss class must be VSLCustomLoss to use this method.")
            if self.debug_mode:
                
                # Ensure optimizer consistency
                check_grounding_value_system_networks_consistency_with_optim(grounding, value_system_per_cluster, self.optim, only_grounding)

    def get_value_systems(self) -> List[Tuple[float]]:
        """
        Get the value systems for each cluster.
        Returns:
            List[Tuple[float]]: List of value systems, each represented as a tuple of weights.
        """
        return [transform_weights_to_tuple(vs.get_alignment_layer()[0][0]) for vs in self.value_system_per_cluster]

    def init_config(self):
        return {
            "Lmax": self.Lmax,
            "alignment_layer_class": self.alignment_layer_class,
            "alignment_layer_kwargs": self.alignment_layer_kwargs,
            "loss_kwargs": self.loss_kwargs,
            "loss_class": self.loss_class,
            "optim_class": self.optim_class,
            "optim_kwargs": self.optim_kwargs,
            "use_wandb": self.use_wandb,
            "debug_mode": self.debug_mode,
            "reward_kwargs": self.reward_kwargs_for_config,
            "n_rewards_for_ensemble": self.n_rewards_for_ensemble

        }

    

    def train_algo(self, 
                   dataset,
        train_env,
        eval_env, 
        run_dir,
        resume_from=0,
        algo='pvsl', env_name='ffmo', experiment_name='testing_algo', tags=[], 
        
         **kwargs):
        
        tags += algo
        tags += env_name
        tags += experiment_name

        if algo == 'pc':
            tags += 'online_policy_learning' if kwargs['online_policy_update'] else 'offline_policy_learning'

        if self.use_wandb:
            
            wandb.init(project="PVSL", dir=run_dir, name=experiment_name, tags=tags, config={"algo": algo, "env_name": env_name, **kwargs, **self.init_config()}, reinit=True, )

        with th.no_grad():
            
            self.train_env = MORecordEpisodeStatisticsCR(train_env, gamma=1.0)
            self.eval_env = eval_env if eval_env is not None else train_env
            self.device = self.train_reward_net.device
            self.static_weights = kwargs.get('static_weights', False)

        
        try : 
            if algo == 'pc':
                ret = self.train_pc(experiment_name=experiment_name, dataset=dataset, resume_from=resume_from,**kwargs)
            elif algo == 'pbmorl':
                self.static_weights = True
                kwargs['static_weights'] = True
                ret = self.train_pbmorl(experiment_name=experiment_name, dataset=dataset, clustered_variant=False, resume_from=resume_from, **kwargs)
            elif algo == 'cpbmorl':
                ret = self.train_pbmorl(experiment_name=experiment_name, dataset=dataset, clustered_variant=True, resume_from=resume_from, **kwargs)
        except Exception as e:
            wandb.finish(exit_code=e.__cause__, quiet=False)
            raise e
        return ret


    def train_pbmorl(self, experiment_name: str, dataset: VSLPreferenceDataset, max_iter: int, H:int,
                 K: int, Ns: int, Nw: int,
                 max_reward_buffer_size: int, m_steps_per_cycle: int, policy_train_kwargs: Dict, 
                 discount_factor_preferences, clustered_variant: bool, use_running_dataset_in_assignment: bool=False, batch_size_in_running_dataset_for_cluster_assignment: int = None, initial_exploration_rate: float = None, initial_reward_learning_iterations: int = None,
                 mutation_prob: float = None, static_weights=False, initial_m_steps_per_cycle=None, mutation_scale: float = None, max_assignment_memory: int = 1,
                 batch_size_reward_buffer: int = None, batch_size_per_agent: int = None, lexicographic_vs_first: bool = None, em_cycles_per_iteration: int = None,
                 resume_from=0):
        assert not self.mobaselines_agent.is_single_objective
        if not clustered_variant:
            self.Lmax = dataset.n_agents
            print("NAGENTS", self.Lmax)
        self.static_weights = static_weights
        self.policy_train_kwargs = policy_train_kwargs
        self.K = K
        self.Ns = Ns
        self.Nw = Nw

        self.policy_train_kwargs = policy_train_kwargs
        policy_iterations = self.policy_train_kwargs['total_timesteps'] // max_iter #if online_policy_update else self.policy_train_kwargs['total_timesteps']
        self.policy_train_kwargs['total_timesteps'] = policy_iterations
        self.mobaselines_agent: MOBaselinesAgent = self.initialize_policy(self.train_env, eval_env=self.eval_env, algo='pbmorl')
        train_Agent: EnvelopePBMORL = self.mobaselines_agent.agent_mobaselines
        assert isinstance(train_Agent, EnvelopePBMORL)
        
        dataset.transition_mode(self.device)

        for global_iter in range(resume_from, max_iter):
            if global_iter == 0:
                self.policy_train_kwargs['reset_learning_starts'] = False
                self.policy_train_kwargs['reset_num_timesteps'] = True
                with th.no_grad():
                    self.global_step = 0
                    PVSL.reset_state(ename=experiment_name)
                    best_assignments_list = ClusterAssignmentMemory(
                        max_size=1 if self.Lmax == 1 else max_assignment_memory, n_values=dataset.n_values )
                    self.repopulate_assignment_list(dataset, best_assignments_list, evenly_spaced=True)
                    
                    self.current_assignment = best_assignments_list.get_best_assignment(lexicographic_vs_first=False)[0]
                    self.update_training_networks_from_assignment(self.value_system_per_cluster, self.train_reward_net, self.current_assignment, only_grounding=self.static_weights)
                    self.policy_train_kwargs['reference_assignment'] = self.current_assignment
                    assert set(self.current_assignment.agent_to_vs_cluster_assignments.keys()) == set(list(dataset.agent_data.keys()))
                
            elif resume_from == global_iter:
                self.policy_train_kwargs['reset_learning_starts'] = False
                self.policy_train_kwargs['reset_num_timesteps'] = False
            
                best_assignments_list, historic, self.mobaselines_agent, final_global_step, config = PVSL.load_state(
                ename=experiment_name, agent_name=self.mobaselines_agent.name, 
                ref_env=self.train_env, ref_eval_env=self.eval_env
                )
                self.current_assignment = historic[global_iter]
                self.global_step = math.floor(global_iter/len(historic))*final_global_step # approximated global step...
                self.update_training_networks_from_assignment(self.value_system_per_cluster, self.train_reward_net, self.current_assignment, only_grounding=self.static_weights)
                self.policy_train_kwargs['reference_assignment'] = self.current_assignment
                #self.Lmax = self.current_assignment.Lmax
            else:
                self.policy_train_kwargs['reset_learning_starts'] = False
                self.policy_train_kwargs['reset_num_timesteps'] = False
                self.update_training_networks_from_assignment(self.value_system_per_cluster, self.train_reward_net, self.current_assignment, only_grounding=self.static_weights)
                self.policy_train_kwargs['reference_assignment'] = self.current_assignment
            
            #self.Lmax = self.current_assignment.Lmax
            #self.update_training_networks_from_assignment(self.value_system_per_cluster, self.train_reward_net, self.current_assignment)
                
            self.current_assignment.grounding = self.train_reward_net
            self.current_assignment.weights_per_cluster = self.value_system_per_cluster
            if not clustered_variant:
                self.update_training_networks_from_assignment(self.value_system_per_cluster, self.train_reward_net, self.current_assignment, only_grounding=self.static_weights)
           
            #self.current_assignment.recalculate_discordances(dataset=dataset, indifference_tolerance=self.loss.model_indifference_tolerance)

            self.mobaselines_agent.agent_mobaselines = train_Agent
            self.mobaselines_agent.policy.set_agent_baselines(train_Agent)
            self.mobaselines_agent.set_reward_vector_function(self.train_reward_net)
            sp = get_signature_params(train_Agent.train, self.policy_train_kwargs)
            print("CHECK!!", self.current_assignment.Lmax, self.Lmax, len(self.value_system_per_cluster), len(best_assignments_list))
            #input()
            self.set_mode('eval')
            train_Agent.train(Ns=Ns, Nw=Nw, K=K, eval_env=self.eval_env,
                              gamma_preferences=discount_factor_preferences, 
                              dataset=dataset,
                              H=H,
                              max_reward_buffer_size=max_reward_buffer_size,
                              reward_train_callback=
                              partial(self._pbmorl_reward_train_callback, 
                                    global_iter=global_iter,
                                    
                                    m_steps_per_cycle=m_steps_per_cycle, 
                                    batch_size_reward_buffer=batch_size_reward_buffer,
                                    best_assignments_list=best_assignments_list) if not clustered_variant else partial(
                                        self._clustered_pbmorl_reward_train_callback, 
                                    global_iter=global_iter,
                                    batch_size_in_running_dataset_for_cluster_assignment=batch_size_in_running_dataset_for_cluster_assignment,
                                    use_running_dataset_in_assignment=use_running_dataset_in_assignment,
                                    initial_reward_learning_iterations=initial_reward_learning_iterations,
                                    initial_m_steps_per_cycle=initial_m_steps_per_cycle,
                                    initial_exploration_rate=initial_exploration_rate,
                                    em_cycles_per_iteration=em_cycles_per_iteration,
                                    mutation_prob=mutation_prob,
                                    mutation_scale=mutation_scale,
                                    batch_size_per_agent=batch_size_per_agent,
                                    lexicographic_vs_first=lexicographic_vs_first,
                                    max_iter=max_iter,
                                    m_steps_per_cycle=m_steps_per_cycle,
                                    disable_selection_after_initial_reward_iterations=True,
                                    best_assignments_list=best_assignments_list,continue_with_best_cluster=False)
                                    ,
                              **sp)
            self.mobaselines_agent.policy.set_agent_baselines(train_Agent)
            assert self.current_assignment.grounding == self.train_reward_net
            self.mobaselines_agent.set_reward_vector_function(self.train_reward_net)

            best = best_assignments_list.get_best_assignment(lexicographic_vs_first=False)[0]
            assert best.Lmax == self.Lmax == len(self.value_system_per_cluster) == self.current_assignment.Lmax
            if not clustered_variant:
                assert best.Lmax == dataset.n_agents
            PVSL.save_state(ename=experiment_name, best_assignments_list=best_assignments_list, historic_dict={global_iter: best}, policy_per_cluster=self.mobaselines_agent, global_step=self.global_step, save_config=self.init_config())
            
        _, historic, _, _, _ = PVSL.load_state(ename=experiment_name, agent_name=self.mobaselines_agent.name, ref_env=self.train_env, ref_eval_env=self.eval_env)
        
        return best_assignments_list, historic, self.mobaselines_agent



    def train_pc(self, experiment_name: str, dataset: VSLPreferenceDataset, max_iter: int, 
              em_cycles_per_iteration: int, initial_m_steps_per_cycle: int, m_steps_per_cycle: int, mutation_prob: float, mutation_scale: float, initial_exploration_rate: float, batch_size_per_agent: int, 
              max_assignment_memory: int, policy_train_kwargs: Dict, discount_factor_preferences, online_policy_update, lexicographic_vs_first,
              resume_from=0):
        
        self.policy_train_kwargs = policy_train_kwargs
        policy_iterations = self.policy_train_kwargs['total_timesteps'] // max_iter if online_policy_update else self.policy_train_kwargs['total_timesteps']
        self.policy_train_kwargs['total_timesteps'] = policy_iterations
        self.mobaselines_agent: MOBaselinesAgent = self.initialize_policy(self.train_env, eval_env=self.eval_env, algo='pc')

        dataset.transition_mode(self.device)
        global_iter = resume_from
        
        while global_iter < max_iter:
            """train_set, val_set = dataset.k_fold_split(1)[0]
            train_set.transition_mode(self.device)"""
            train_set , val_set = dataset, dataset
            self.set_mode('eval')
            with th.no_grad():
                if global_iter == 0:
                    self.global_step = 0
                    PVSL.reset_state(ename=experiment_name)
                    best_assignments_list = ClusterAssignmentMemory(
                        max_size=1 if self.Lmax == 1 else max_assignment_memory,n_values=dataset.n_values
                    )
                    self.repopulate_assignment_list(dataset, best_assignments_list)
                        #print(len(best_assignments_list), best_assignments_list.max_size)
                elif resume_from == global_iter:
                    best_assignments_list, historic, self.mobaselines_agent, final_global_step, config = PVSL.load_state(ename=experiment_name, agent_name=self.mobaselines_agent.name, ref_env=self.train_env, ref_eval_env=self.eval_env)
                    assert global_iter < len(historic), f"Historic does not contain the supplied resume_from ({resume_from}), maximum is {len(historic)}"
                    self.current_assignment: ClusterAssignment = historic[global_iter]
                    self.global_step = math.floor(global_iter/len(historic))*final_global_step # approximated global step...
                    self.update_training_networks_from_assignment(self.value_system_per_cluster, self.train_reward_net, self.current_assignment, only_grounding=self.static_weights)
                
            # Training logic for the algorithm
            ret = self._clustered_pbmorl_reward_train_callback(base_dataset=train_set, running_dataset=None,
                                                               batch_size_in_running_dataset_for_cluster_assignment=0,
                                                         initial_reward_learning_iterations=1, reward_vector_function=self.train_reward_net,
                                                         lexicographic_vs_first=lexicographic_vs_first,
                                                         em_cycles_per_iteration=em_cycles_per_iteration,
                                                         m_steps_per_cycle=m_steps_per_cycle,max_iter=max_iter,
                                                         initial_m_steps_per_cycle=initial_m_steps_per_cycle,
                                                         initial_exploration_rate=initial_exploration_rate,
                                                         use_running_dataset_in_assignment=False,
                                                         mutation_prob=mutation_prob, mutation_scale=mutation_scale,
                                                         batch_size_per_agent=batch_size_per_agent, global_iter=global_iter,
                                                         best_assignments_list=best_assignments_list)
            global_iter = ret['global_iter']
            """self.current_assignment, original, mutated, is_protected = self.selection(best_assignments_list, 
                                                                                        lexicographic_vs_first, 
                                                                                        global_iter, max_iter, 
                                                                                        initial_exploration_rate=initial_exploration_rate, mutation_prob=mutation_prob, mutation_scale=mutation_scale, dataset=dataset)

            if not is_protected and not mutated:
                assert self.current_assignment == original, f"Mut? {mutated} Current assignment {self.current_assignment} is not equal to original {original}"
            
            
            #train_dataset = dataset
            #val_dataset = dataset
            ##self.current_assignment.certify_cluster_assignment_consistency()
            self.update_training_networks_from_assignment(self.value_system_per_cluster, self.train_reward_net, self.current_assignment)
            ##self.current_assignment.certify_cluster_assignment_consistency()
            for cycle in range(em_cycles_per_iteration):
                steps_now = initial_m_steps_per_cycle if cycle == 0 else m_steps_per_cycle
                #self.current_assignment.certify_cluster_assignment_consistency()
                if (not mutated) or cycle >= 1: #and (self.Lmax > 1 or global_iter == 0):
                    self.cluster_assignment(dataset, qualitative_cluster_assignment) # E-step
                #self.current_assignment.certify_cluster_assignment_consistency()
                train_losses = self.train_reward_models(train_set, global_step=self.global_step,
                                        n_optim_steps=steps_now, batch_size_per_agent=batch_size_per_agent) # M-step
                #self.current_assignment.certify_cluster_assignment_consistency()
                if cycle == em_cycles_per_iteration - 1:
                    if self.use_wandb:
                        self.wandb_log_losses(global_iter, train_losses)
                with th.no_grad():
                    self.current_assignment.n_training_steps += steps_now
                    self.current_assignment.optimizer_state = self.optim.get_state(copy=True)
                    original.optimizer_state = self.optim.get_state(copy=True)
                    self.global_step += steps_now
                    #self.current_assignment.certify_cluster_assignment_consistency()
                    
            if online_policy_update:
                self.mobaselines_agent = self.update_policy(dataset, iterations=policy_iterations) # Policy update

            self.set_mode('eval')
            with th.no_grad():
                #self.current_assignment.grounding = self.train_reward_net
                #self.current_assignment.weights_per_cluster = self.value_system_per_cluster
                self.current_assignment.recalculate_discordances(val_set, indifference_tolerance=self.loss.model_indifference_tolerance)
                if mutated or is_protected:
                    best_assignments_list.insert_assignment(self.current_assignment)
                else:
                    best_assignments_list.notify_updated_assignment(self.current_assignment)
                if self.debug_mode:
                    print(best_assignments_list)"""
            best, _  = best_assignments_list.get_best_assignment(lexicographic_vs_first=False)
            best.certify_cluster_assignment_consistency()
            best.grounding.set_mode('test')
            self.wandb_log_cluster_metrics(best, global_iter, 'eval')
            self.wandb_log_cluster_metrics(self.current_assignment, global_iter, 'train')

                #wandb.log({"global/conciseness": best.get_average_discordance(val_set, indifference_tolerance=self.loss.model_indifference_tolerance)}, step=global_iter)
                
            PVSL.save_state(ename=experiment_name, best_assignments_list=None, historic_dict={global_iter: best}, policy_per_cluster=None, global_step=self.global_step, save_config=self.init_config())
                #PVSL.load_state(ename=experiment_name, agent_name=self.mobaselines_agent.name, ref_env=self.train_env, ref_eval_env=self.eval_env)
        PVSL.save_state(ename=experiment_name, best_assignments_list=best_assignments_list, historic_dict=None, policy_per_cluster=self.mobaselines_agent, global_step=self.global_step, save_config=self.init_config())
        if not online_policy_update:
            _, historic, _, _, _ = PVSL.load_state(ename=experiment_name, agent_name=self.mobaselines_agent.name, ref_env=self.train_env, ref_eval_env=self.eval_env)
            self.current_assignment = historic[-1]
            self.update_training_networks_from_assignment(self.value_system_per_cluster, self.train_reward_net, self.current_assignment, only_grounding=self.static_weights)
            self.mobaselines_agent = self.update_policy(dataset, iterations=policy_iterations)
        PVSL.save_state(ename=experiment_name, best_assignments_list=best_assignments_list, historic_dict=None, policy_per_cluster=self.mobaselines_agent, global_step=self.global_step, save_config=self.init_config())
        
        return best_assignments_list, historic, self.mobaselines_agent

    def repopulate_assignment_list(self, dataset, best_assignments_list, evenly_spaced=False):
        best_assignments_list.initializing = True
        while len(best_assignments_list) < best_assignments_list.max_size:
            ca = generate_random_assignment(dataset, l_max=self.Lmax, alignment_layer_class=self.alignment_layer_class, alignment_layer_kwargs=self.alignment_layer_kwargs, ref_grounding=self.train_reward_net, seed=np.random.randint(0, 10000), evenly_spaced=evenly_spaced)
            
            ca.recalculate_discordances(dataset, indifference_tolerance=self.loss.model_indifference_tolerance)
            ca.optimizer_state = self.optim.get_state(copy=True)
            self.update_training_networks_from_assignment(self.value_system_per_cluster, self.train_reward_net, ca, only_grounding=self.static_weights)
            best_assignments_list.insert_assignment(ca.copy())
            print("BA", best_assignments_list)
        best_assignments_list.initializing = False

    def wandb_log_cluster_metrics(self, assignment, global_iter, mode='train'):
        if self.use_wandb:
            wandb.log({f"{mode}/conciseness": assignment.conciseness_vs(), f"{mode}/representativeness": assignment.representativity_vs(), f"{mode}/coherence": float(np.mean(assignment.coherence()))}, step=global_iter)

    def wandb_log_losses(self, global_iter, train_losses, mode='train'):
        if self.use_wandb:
            wandb.log({f"{mode}/loss_vs": train_losses[0], f"{mode}/loss_gr": train_losses[1], f"{mode}/vs_norm": train_losses[2], f"{mode}/gr_norm": train_losses[3]}, step=global_iter)

    def reset_state(ename):
        with th.no_grad():
            folder = os.path.join(CHECKPOINTS, "train_results", ename)
            if os.path.exists(folder):
                for root, dirs, files in os.walk(folder, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
            
    def save_state(ename, best_assignments_list: ClusterAssignmentMemory, historic_dict, policy_per_cluster: MOBaselinesAgent, global_step: int, save_config=None):
        with th.no_grad():
            folder = os.path.join(CHECKPOINTS, "train_results", ename)
            os.makedirs(folder, exist_ok=True)
            if best_assignments_list is not None:
                with open(os.path.join(folder, "best_assignments_list.pkl"), "wb") as file:
                    dill.dump(best_assignments_list, file)
            # Save the historic assignments in different files
            if historic_dict is not None:
                for global_iter, assignment in historic_dict.items():
                    assignment: ClusterAssignment
                    assignment.save(os.path.join(folder, 'historic'), f"assignment_{global_iter}")
            # Save policy
            if policy_per_cluster is not None:
                policy_per_cluster.save(path=os.path.join(folder, "policy"), full_save=True)
            # Save global step
            with open(os.path.join(folder, "global_step.pkl"), "wb") as file:
                dill.dump(global_step, file)
            if save_config is not None:
                with open(os.path.join(folder, "config.pkl"), "wb") as file:
                    dill.dump(save_config, file)

    def load_state(ename, agent_name, ref_env, ref_eval_env):
        folder = os.path.join(CHECKPOINTS, "train_results", ename)
        print(f"\033[94mLOADING PVSL from {folder}\033[0m")
        with open(os.path.join(folder, "best_assignments_list.pkl"), "rb") as file:
            best_assignments_list = dill.load(file)
        historic = {}
        for filename in os.listdir(os.path.join(folder, 'historic')):
            if filename.endswith('.pkl'):
                global_iter = int(filename.split('_')[1].split('.')[0])
                with open(os.path.join(folder, 'historic', filename), "rb") as file:
                    historic[global_iter] = dill.load(file)
        historic_list = [v for k, v in sorted(historic.items(), key=lambda item: item[0], reverse=False)]
        #assert historic_list[0] == historic[0]
        policy = MOBaselinesAgent.load(env=ref_env, eval_env=ref_eval_env, path=os.path.join(folder, "policy"), name=agent_name)
        # load global step
        with open(os.path.join(folder, "global_step.pkl"), "rb") as file:
            global_step = dill.load(file)
        with open(os.path.join(folder, "config.pkl"), "rb") as file:
            save_config = dill.load(file)
        return best_assignments_list, historic_list, policy, global_step, save_config
    
    def initialize_policy(self, env, eval_env, algo='pc', **kwargs):
        """
        Initialize a policy for a given cluster.
        This method should be implemented to return a policy agent for the specified cluster.
        """
        self.mobaselines_agent.set_envs(env=env, eval_env=eval_env)
        if algo == 'pbmorl':
            """replay_buffer = CustomRewardReplayBuffer(obs_shape=self.mobaselines_agent.agent_mobaselines.observation_shape,
                action_dim=1,
                prioritized=self.mobaselines_agent.agent_mobaselines.per,
                estimated_horizon=self.mobaselines_agent.agent_mobaselines.replay_buffer.estimated_horizon,
                buffer_with_weights=self.mobaselines_agent.agent_mobaselines.replay_buffer.buffer_with_weights,
                rew_dim=self.grounding_network.num_outputs,
                max_size=self.mobaselines_agent.agent_mobaselines.replay_buffer.max_size,
                action_dtype=np.uint8,
                reward_vector_function=self.train_reward_net,
                relabel_buffer=True,
                maintain_original_reward=True
            )
            self.mobaselines_agent.agent_mobaselines.set_buffer(replay_buffer)"""
        self.mobaselines_agent.set_reward_vector_function(self.train_reward_net)
        
        return self.mobaselines_agent
        
    def exploration_rate(self, initial_exploration_rate , iter_now, num_iterations):
        return initial_exploration_rate * ((num_iterations-iter_now)/num_iterations) 
    
    def selection(self, best_assignments_list: ClusterAssignmentMemory, lexicographic_vs_first, iter_now, num_iterations, initial_exploration_rate, mutation_scale, mutation_prob, dataset) -> ClusterAssignment:
        if len(best_assignments_list) < best_assignments_list.max_size:
            best_assignments_list.initializing = True
            while len(best_assignments_list) < best_assignments_list.max_size:
                original, is_protected = best_assignments_list.get_random_weighted_assignment(consider_only_unexplored=False, lexicographic_vs_first=lexicographic_vs_first)
                mut = generate_mutated_assignment(original.copy(), mutation_scale, mutation_prob)
                
                mut.recalculate_discordances(dataset, indifference_tolerance=self.loss.model_indifference_tolerance)
                
                best_assignments_list.insert_assignment(mut)
            best_assignments_list.initializing = False

        with th.no_grad():
            original, is_protected = best_assignments_list.get_random_weighted_assignment(consider_only_unexplored=False, lexicographic_vs_first=lexicographic_vs_first)
            if random.random() > self.exploration_rate(initial_exploration_rate=initial_exploration_rate , iter_now=iter_now, num_iterations=num_iterations):  # Example: 1000 is the total number of iterations
                # Randomly select an assignment
                original.explored = True
                if is_protected:
                    new_assignment = original.copy()
                else:
                    new_assignment = original
                return new_assignment, original, False, is_protected
            else:
                # TODO Select two assignments, crossover them (TODO), then mutate
                mutation = generate_mutated_assignment(original.copy(), mutation_scale, mutation_prob)
                mutation.explored = False
                if initial_exploration_rate == 0.0:
                    raise ValueError("Initial exploration rate is 0.0, cannot mutate.")
                return mutation, original, True, False


    def get_cluster_colors(self, mode='colorama', **kwargs):
        self.cluster_colors_vs[mode] = COLOR_PALETTE.get(mode)(len(self.value_system_per_cluster), **kwargs)
        assert len(self.cluster_colors_vs[mode]) == self.Lmax, f"Cluster colors length mismatch: {len(self.cluster_colors_vs[mode])} vs {self.Lmax}"
        return self.cluster_colors_vs[mode]

    def cluster_assignment(self, dataset: VSLPreferenceDataset, running_dataset: FixedLengthVSLPreferenceDataset=None, batch_size_in_running_dataset_for_cluster_assignment = 0, qualitative_cluster_assignment=False):
        self.set_mode('eval')
        if self.debug_mode:
            print("Clustering assignments based on current value systems and grounding network...")
        
        cluster_colors_vs= self.get_cluster_colors('colorama')
        
        agent_to_vs_cluster_assignments = dict()
        assignment_vs = [[] for _ in range(self.Lmax)]
        with th.no_grad():
            fragments1 = dataset.fragments1
            fragments2 = dataset.fragments2
            if running_dataset is not None:
                frags_per_aid_running = dict()
                for aid in dataset.fidxs_per_agent.keys():
                    if aid in running_dataset.fidxs_per_agent:
                        size_data = len(running_dataset.fidxs_per_agent[aid])
                        if batch_size_in_running_dataset_for_cluster_assignment > size_data:
                            frags_per_aid_running[aid] = running_dataset.fidxs_per_agent[aid]
                        else:
                            frags_per_aid_running[aid] = np.random.choice(running_dataset.fidxs_per_agent[aid], size=batch_size_in_running_dataset_for_cluster_assignment, replace=False)

                

            grounding1 = self.train_reward_net.forward(fragments1)
            grounding2 = self.train_reward_net.forward(fragments2)
            for aid in dataset.agent_data.keys():
                
                fidxs = dataset.fidxs_per_agent[aid]
                if running_dataset is not None and aid in frags_per_aid_running:
                    fidxs_running = frags_per_aid_running[aid]
                    grounding1_running = self.train_reward_net.forward(running_dataset.fragments1[fidxs_running]) 
                    grounding2_running = self.train_reward_net.forward(running_dataset.fragments2[fidxs_running])
                #print("AGENT IDS", aid, fidxs, grounding1[fidxs][0:10])
                agent_preferences = dataset.preferences[fidxs]
                if (running_dataset is not None) and aid in frags_per_aid_running.keys():
                    agent_preferences = np.concatenate((agent_preferences, running_dataset.preferences[fidxs_running]))
                    assert agent_preferences.shape == (len(fidxs) + len(fidxs_running),), f"Preference shape mismatch: {agent_preferences.shape} vs {len(fidxs)} + {len(fidxs_running)}"
                aid_likelihood_per_vs_cluster = [1.0]*self.Lmax
                for c in range(self.Lmax):
                    vs = self.value_system_per_cluster[c]
                    # TODO this can be more efficient...
                    
                    probs = probability_BT(vs.forward(grounding1[fidxs]), vs.forward(grounding2[fidxs])).squeeze(-1)  # shape: (n_fragments, n_values)
                    if (running_dataset is not None) and aid in frags_per_aid_running.keys():
                        probs_running = probability_BT(vs.forward(grounding1_running), vs.forward(grounding2_running)).squeeze(-1)
                        probs = th.cat([probs, probs_running], dim=0)
                    gt_probs = agent_preferences
                    assert probs.shape == gt_probs.shape, f"Shape mismatch: {probs.shape} vs {gt_probs.shape}"

                    qualitative_mode = False
                    aid_likelihood_per_vs_cluster[c] = likelihood_x_is_target(probs.detach(), util.safe_to_tensor(gt_probs).detach(), mode='th', slope=0.3, adaptive_slope=False, qualitative_mode=qualitative_mode, indifference_tolerance=self.loss.model_indifference_tolerance)

                    #aid_likelihood_per_vs_cluster[c] = likelihood_x_is_target(probs.detach(), util.safe_to_tensor(gt_probs).detach(), mode='th', slope=0.3, adaptive_slope=False, qualitative_mode=True, indifference_tolerance=self.loss.model_indifference_tolerance)
                
                # Find clusters with maximum likelihood
                if qualitative_mode:
                    max_likelihood = max(aid_likelihood_per_vs_cluster)
                    best_clusters = [i for i, v in enumerate(aid_likelihood_per_vs_cluster) if v == max_likelihood]
                    # From those, select the one with most agents already assigned (break ties by smallest index)
                    counts = [len(assignment_vs[i]) for i in best_clusters]
                    max_count = max(counts)
                    candidate_clusters = [c for c, cnt in zip(best_clusters, counts) if cnt == max_count]
                    best_cluster = min(candidate_clusters)
                else:
                    best_cluster = np.argmax(aid_likelihood_per_vs_cluster)
                agent_to_vs_cluster_assignments[aid] = best_cluster
                assignment_vs[best_cluster].append(aid)

                if len(self.value_system_per_cluster) > 1 and self.debug_mode:
                    print("AID: ", aid, "to:", end='')
                    for i, value in enumerate(aid_likelihood_per_vs_cluster):
                        color = cluster_colors_vs[i]
                        print(f"{color}{value}{Style.RESET_ALL}", end=', ')
                    print(f"), {cluster_colors_vs[best_cluster]} {self.value_system_per_cluster[best_cluster].get_alignment_layer()[0]} - VS Cluster {best_cluster}: {aid_likelihood_per_vs_cluster[best_cluster]}, {Style.RESET_ALL}")

            asgin = deepcopy(assignment_vs)
            for cl, ags in enumerate(asgin):
                if len(ags) == 0:
                    continue
                vsc = self.value_system_per_cluster[cl].get_alignment_layer()[0].detach().numpy()
                for cl2, ags2 in enumerate(asgin): 
                    if len(ags2) == 0 or cl2 == cl:
                        continue
                    
                    vsc2 = self.value_system_per_cluster[cl2].get_alignment_layer()[0].detach().numpy()
                    if np.max(abs(vsc - vsc2)) <= 0.01:
                               
                        if len(ags) >= len(ags2): # we choose the most popular weights
                            assignment_vs[cl].extend(assignment_vs[cl2])
                            assignment_vs[cl2] = []
                            for ag in ags2:
                                agent_to_vs_cluster_assignments[ag] = cl
                            self.value_system_per_cluster[cl2].set_weights(random_weights(dataset.n_values, 1, dist='gaussian', seed=np.random.randint(0,1000)))
                        else:
                            assignment_vs[cl2].extend(assignment_vs[cl])
                            assignment_vs[cl] = []
                            for ag in ags:
                                agent_to_vs_cluster_assignments[ag] = cl2
                            self.value_system_per_cluster[cl].set_weights(random_weights(dataset.n_values, 1, dist='gaussian', seed=np.random.randint(0,1000)))

            self.current_assignment.agent_to_vs_cluster_assignments = agent_to_vs_cluster_assignments
            self.current_assignment.assignments_vs = assignment_vs
            self.current_assignment.grounding = self.train_reward_net
            self.current_assignment.weights_per_cluster = self.value_system_per_cluster
            self.current_assignment._inter_discordances_vs = None
            self.current_assignment._inter_discordances_vs_per_cluster_pair = None
            self.current_assignment._inter_discordances_vs_per_agent = None
            self.current_assignment._intra_discordances_gr_per_agent = None
        return self.current_assignment

    def set_mode(self, mode: str):
        if not self.static_weights:
            if hasattr(self, 'current_assignment'):
                self.current_assignment.grounding.set_mode(mode)
                for vs in self.current_assignment.weights_per_cluster:
                    vs.requires_grad_(False)
            self.train_reward_net.set_mode(mode)
            for vs in self.value_system_per_cluster:
                vs.requires_grad_(mode == 'train')
            
        else:
            if hasattr(self, 'current_assignment'):
                self.current_assignment.grounding.set_mode(mode)
                for vs in self.current_assignment.weights_per_cluster:
                    vs.requires_grad_(False)
            self.train_reward_net.set_mode(mode)
            for vs in self.value_system_per_cluster:
                vs.requires_grad_(False)
            
        check_grounding_value_system_networks_consistency_with_optim(self.train_reward_net, self.value_system_per_cluster, self.optim, only_grounding=self.static_weights)
        #check_grounding_value_system_networks_consistency_with_optim(self.current_assignment.grounding, self.current_assignment.weights_per_cluster, self.optim, only_grounding=True, copies=True)

    def train_reward_models_no_clusters(self, dataset, global_step, n_optim_steps, batch_size_reward_buffer=32):
        train_losses = []
        
        self.set_mode('train')
        for sum_ensemble in range(self.train_reward_net.n_models):
            self.train_reward_net.use_models(sum_ensemble)
            for i in range(n_optim_steps):
                optim_step = global_step + i
                with th.no_grad():
                    dataset: VSLPreferenceDataset
                    t = time.time()
                    dataset_b, fids_per_agent = dataset.select_batch(batch_size_reward_buffer, transitions=False, device=self.train_reward_net.device, per_agent=False)

                    fragments1, fragments2, preferences, preferences_with_grounding, agent_ids = dataset_b
                    rt = time.time()
                    #print(f"\033[33mBATCH SELECT back took {rt-t:.4f} seconds for {len(fragments1),} frags and {len(set(agent_ids))}agent\033[0m")

                #print("AGENT IDS. SHOULD GROUP THEM ACCORDING TO WEIGHTS, THEN...", agent_ids)
                t = time.time()
                unique_agent_ids = OrderedSet(agent_ids)
                cluster_fidxs = [[] for _ in range(len(unique_agent_ids))]
                agent_to_vs_cluster_assignments = dict()
                assignments_vs = [[] for _ in range(len(unique_agent_ids))]
                for iag, ag in enumerate(unique_agent_ids):
                    fidxsag = fids_per_agent.get(ag, [])
                    #print("AGENT", ag, "fidxs", fidxsag)
                    #assert fragments1[fidxsag][0].agent in ag, f"Fragment {fragments1[fidxsag][0].agent} does not match agent {ag} in fidxs_per_agent."
                    cluster_fidxs[int(iag)].extend(fidxsag)
                    agent_to_vs_cluster_assignments[ag] = int(iag)
                    assignments_vs[int(iag)] = [ag]
                
                #print(len(weights_per_cluster), len(cluster_fidxs), len(assignments_vs), len(agent_to_vs_cluster_assignments))
                weights_per_cluster = [create_alignment_layer(parse_or_invent_alignment_function_from_name(agname, self.train_reward_net.num_outputs), self.alignment_layer_class, self.alignment_layer_kwargs).requires_grad_(False) for agname in unique_agent_ids]
                rt = time.time()
                #print(f"\033[Create layers callback took {rt-t:.4f} seconds\033[0m")
                optimizer_state = self.optim.get_state()
                dummy_assignment = ClusterAssignment(weights_per_cluster, self.train_reward_net,agent_to_vs_cluster_assignments=agent_to_vs_cluster_assignments,assignments_vs=assignments_vs)
                dummy_assignment.optimizer_state = optimizer_state
                dummy_assignment.n_training_steps = optim_step
                self.update_training_networks_from_assignment(dummy_assignment.weights_per_cluster, self.train_reward_net, dummy_assignment, only_grounding=True)

                t =time.time()
                if __debug__:
                    prev_params_vs = deepcopy(dummy_assignment.weights_per_cluster)
                    prev_params_gr = deepcopy(OrderedSet(self.train_reward_net.rewards[sum_ensemble].parameters()))
                rt = time.time()
                #print(f"\033[33mPREV PARAMS. BATCH SELECT back took {rt-t:.4f} seconds\033[0m {self.loss.per_agent}")
                self.optim.zero_grad()
                assert th.is_grad_enabled()
                #print(len(fragments1[0]))
                t = time.time()
                output = self.loss.forward(fragments1=fragments1, fragments2=fragments2,
                                            reward_vector_module=self.train_reward_net,
                                            weights_per_cluster= dummy_assignment.weights_per_cluster,
                                            cluster_fidxs=cluster_fidxs, 
                                            fidxs_per_agent=fids_per_agent,
                                            preferences=util.safe_to_tensor(preferences, dtype=self.train_reward_net.desired_dtype).requires_grad_(False), 
                                            preferences_with_grounding=util.safe_to_tensor(preferences_with_grounding, dtype=self.train_reward_net.desired_dtype).requires_grad_(False))
                loss_vs, loss_gr, loss_gr_per_vi = output.loss
                rt = time.time()
                #print(f"\033[33m FORWARD back took {rt-t:.4f} seconds\033[0m")
                loss =  loss_vs  + loss_gr # + loss_policy??? TODO.
                renormalization = 1.0
                """if batch_size_reward_buffer != 'full':
                    if isinstance(batch_size_reward_buffer , float):
                        renormalization = batch_size_reward_buffer*len(dataset)
                    else:
                        renormalization = batch_size_reward_buffer/math.ceil(len(dataset))
                print("R", renormalization)"""

                self.loss.gradients(scalar_loss = loss, renormalization = renormalization)
                rt = time.time()
                #print(f"\033[33m GRADIENTS AND FORWARD back took {rt-t:.4f} seconds\033[0m")
                if __debug__:
                    next_params_vs = dummy_assignment.weights_per_cluster
                    next_params_gr = OrderedSet(self.train_reward_net.rewards[sum_ensemble].parameters())
                
                    for p, g in zip(next_params_vs, prev_params_vs):
                        assert all([gi.grad is None for gi in g.parameters()]), "Gradients must NOT be computed for THESE parameters."
                        assert all([pi.grad is None for pi in p.parameters()]), "Gradients must NOT be computed for THESE parameters."
                    for p, g in zip(next_params_gr, prev_params_gr):
                        assert p.grad is not None, "Gradients must be computed for THESE parameters."
                        assert g.grad is None, "Gradients must NOT be computed for THESE parameters."
                self.optim.step()
                
                if __debug__:
                    next_params_vs = dummy_assignment.weights_per_cluster
                    for p, g in zip(next_params_vs, prev_params_vs):
                        assert th.allclose(p.weight,g.weight)
                    for p, g in zip(next_params_gr, prev_params_gr):
                        assert th.allclose(p,g) == False, "Grounding network parameters did not change after optimization step."
                
                    #assert set(next_params_vs) == set(prev_params_vs) # "Value system parameters did not change after optimization step."
                gr_norm = get_grad_norm(self.train_reward_net.parameters())
                vs_norm = get_grad_norm([p for vs in self.value_system_per_cluster for p in vs.parameters()])
                
                self.optim.zero_grad()

                train_losses.append((loss_vs.detach().item(), loss_gr.detach().item(), vs_norm.detach().item(), gr_norm.detach().item()))

        # Perform sum over each array in each tuple inside train_losses:
        train_losses = [np.sum([x[i] for x in train_losses]) / self.train_reward_net.n_models for i in range(len(train_losses[0]))]
        self.update_assignment_from_trained_networks(self.current_assignment, trained_reward_net=self.train_reward_net, weights=self.value_system_per_cluster, only_grounding=self.static_weights)
        

        if self.debug_mode:
            print("Reward models trained successfully.")
            print("------------------")
            print("Training losses", train_losses)
            print("------------------")
            #input()

        self.train_reward_net.use_models('all')
        self.set_mode('eval')
        return train_losses
    
    def train_reward_models(self, dataset: VSLPreferenceDataset, global_step, n_optim_steps, batch_size_per_agent='full', transitions=True):
        #self.loss.set_parameters(params_gr=self.optim.params_gr, params_vs=self.optim.params_vs, optim_state=self.optim.get_state())
        print("TRmodels", [w.get_alignment_layer()[0] for w in self.value_system_per_cluster])
        print("Current assignment:", self.current_assignment.value_systems)
        
        self.set_mode('train')
        train_losses = []
        print("SIZE DATA", len(dataset), "NAGENTS", dataset.n_agents, "BATCH SIZE", batch_size_per_agent)

        self.update_training_networks_from_assignment(self.value_system_per_cluster, self.train_reward_net, self.current_assignment, only_grounding=self.static_weights)
            
        #splits = dataset.k_fold_split(self.train_reward_net.n_models, generate_val_dataset=True)
        for sum_ensemble in range(self.train_reward_net.n_models):
            self.train_reward_net.use_models(sum_ensemble)

            #self.update_training_networks_from_assignment(self.value_system_per_cluster, self.train_reward_net, self.current_assignment, only_grounding=self.static_weights)

            for i in range(n_optim_steps):
                optim_step = global_step + i
                with th.no_grad():
                    dataset_b, fids_per_agent = dataset.select_batch(batch_size_per_agent, transitions=transitions, device=self.train_reward_net.device)
                    
                    fragments1, fragments2, preferences, preferences_with_grounding, agent_ids = dataset_b
                    print("Sampled", len(fragments1), "fragments, agents:", len(OrderedSet(agent_ids)))

                    cluster_fidxs = [list() for _ in self.current_assignment.assignments_vs]
                for ag, fidxsag in fids_per_agent.items():
                    #print(self.current_assignment.agent_to_vs_cluster_assignments.keys())
                    #assert fragments1[fidxsag][0].agent in ag, f"Fragment {fragments1[fidxsag][0].agent} does not match agent {ag} in fidxs_per_agent."
                    cluster_fidxs[int(self.current_assignment.agent_to_vs_cluster_assignments[ag])].extend(fidxsag)
                self.optim.zero_grad()
                #print(len(fragments1[0]))
                check_grounding_value_system_networks_consistency_with_optim(self.train_reward_net, self.value_system_per_cluster, self.optim, only_grounding=self.static_weights, check_grads=True)
                
                output = self.loss.forward(fragments1=fragments1, fragments2=fragments2,
                                            reward_vector_module=self.train_reward_net,
                                            weights_per_cluster=self.value_system_per_cluster,
                                            cluster_fidxs=cluster_fidxs, 
                                            fidxs_per_agent=fids_per_agent,
                                            preferences=util.safe_to_tensor(preferences, dtype=self.train_reward_net.desired_dtype).requires_grad_(False), 
                                            preferences_with_grounding=util.safe_to_tensor(preferences_with_grounding, dtype=self.train_reward_net.desired_dtype).requires_grad_(False))
                loss_vs, loss_gr, loss_gr_per_vi = output.loss
                
                loss =  loss_vs  + loss_gr # + loss_policy??? TODO.
                renormalization = 1.0
                """if batch_size_per_agent != 'full':
                    if isinstance(batch_size_per_agent , float):
                        renormalization = batch_size_per_agent*math.ceil(len(dataset))
                    else:
                        assert isinstance(batch_size_per_agent, int), f"batch_size_per_agent must be an int or a float, got {type(batch_size_per_agent)}"
                        renormalization = batch_size_per_agent*dataset.n_agents/math.ceil(len(dataset))
"""
                self.loss.gradients(scalar_loss = loss_vs, renormalization = renormalization)
                
                self.optim.step()
                gr_norm = get_grad_norm(self.train_reward_net.parameters())
                vs_norm = get_grad_norm([p for vs in self.value_system_per_cluster for p in vs.parameters()])
                self.optim.zero_grad()

                train_losses.append((loss_vs.detach().item(), loss_gr.detach().item(), vs_norm.detach(), gr_norm.detach()))

        # Perform sum over each array in each tuple inside train_losses:
        train_losses = [np.sum([x[i] for x in train_losses]) for i in range(len(train_losses[0]))]
        
        self.update_assignment_from_trained_networks(self.current_assignment, trained_reward_net=self.train_reward_net, weights=self.value_system_per_cluster, only_grounding=self.static_weights)
        
        #input("PAUSED")

        if self.debug_mode:
            print("Reward models trained successfully.")
            print("------------------")
            print("Training losses", train_losses)
            print("------------------")
            #input()

        self.train_reward_net.use_models('all')
        self.set_mode('eval')
        return train_losses
        

    def update_policy(self, dataset: VSLPreferenceDataset, iterations=10):
        """
        Update the policies for each cluster based on the current value systems.
        """
        if self.debug_mode:
            print("Updating policies for each cluster...")
        self.set_mode('eval')
        self.mobaselines_agent.set_reward_vector_function(self.train_reward_net)
        self.mobaselines_agent.set_envs(self.train_env, self.eval_env)
        self.mobaselines_agent.train_kwargs = self.policy_train_kwargs

        if self.mobaselines_agent.is_single_objective:
            for c, agents in self.current_assignment.active_vs_clusters():
                
                self.mobaselines_agent.set_weights(self.value_system_per_cluster[c].get_alignment_layer()[0][0], override_existent=c)
                self.mobaselines_agent.fit(eval_env=self.eval_env, weight=self.value_system_per_cluster[c])
                
        else:
            self.mobaselines_agent.fit(eval_env=self.eval_env)
        
        if self.debug_mode:
            print("Policies updated successfully.")
        
        return self.mobaselines_agent
    

from imitation.data import rollout
from tqdm import tqdm
class EnvelopeClusteredPBMORL(EnvelopePBMORL):
    def __init__(self, env, buffer_with_weights, cluster_weights_in_envelope_prob, cluster_assignment=None, learning_rate=0.0003, estimated_horizon=10, initial_epsilon=0.01, final_epsilon=0.01, epsilon_decay_steps=None, tau=1, target_net_update_freq=200, buffer_size=..., net_arch=..., batch_size=256, learning_starts=100, gradient_updates=1, gamma=0.99, max_grad_norm=1, envelope=True, num_sample_w=4, per=False, per_alpha=0.6, initial_homotopy_lambda=0, final_homotopy_lambda=1, homotopy_decay_steps=None, project_name="MORL-Baselines", experiment_name="Envelope", wandb_entity=None, log=True, seed=None, device="auto", activation=None, group=None, masked=False, relabel_buffer=True, reward_vector_function=None):
        super().__init__(env, buffer_with_weights, True, learning_rate, estimated_horizon, initial_epsilon, final_epsilon, epsilon_decay_steps, tau, target_net_update_freq, buffer_size, net_arch, batch_size, learning_starts, gradient_updates, gamma, max_grad_norm, envelope, num_sample_w, per, per_alpha, initial_homotopy_lambda, final_homotopy_lambda, homotopy_decay_steps, project_name, experiment_name, wandb_entity, log, seed, device, activation, group, masked, relabel_buffer, reward_vector_function)
        self.running_assignment: ClusterAssignment = cluster_assignment
        self.cluster_weights_in_envelope_prob = cluster_weights_in_envelope_prob
    def sample_eval_weights(self, n):
        if self.running_assignment is None:
            return super().sample_eval_weights(n)
        else:
            if n < len(self.running_assignment.value_systems):
                return np.random.choice(self.running_assignment.value_systems, size=n, replace=False)
            elif n == len(self.running_assignment.value_systems):
                return self.running_assignment.value_systems
            else:
                weights_base = set(deepcopy(self.running_assignment.value_systems))
                eqweights = set([tuple(w) for w in super().sample_eval_weights(n=n)])
                weights_base.update(eqweights)
                return weights_base
                
    
    @override
    def reward_training(self, Ns, Nw, H, gamma_preferences, reward_train_callback, max_reward_buffer_size, qualitative_preferences=True):
        #print("RW1?? RUNNING ASSIGNMENT", self.running_assignment)
        #print(self.running_assignment.value_systems if self.running_assignment is not None else None)
        #input("Press Enter to continue...")
        return_dict = super().reward_training(Ns, Nw, H, gamma_preferences, reward_train_callback, max_reward_buffer_size, qualitative_preferences=qualitative_preferences)
        self.running_assignment = return_dict.get('running_assignment', None)
        #print("RW2?? RUNNING ASSIGNMENT", self.running_assignment)
        #print(self.running_assignment.value_systems if self.running_assignment is not None else None)
        #input("Press Enter to continue...")
        return return_dict
    
    def sample_traj_from_dataset(self,dataset: VSLPreferenceDataset, Ns, gamma):
        fragments = dataset.fragments_best
        idx = np.random.choice(np.arange(len(fragments)), size=Ns, replace=True if Ns > len(fragments) else False)
        
        frag = fragments[idx]
        
        
        v_rews_real = []
        v_rews = []
        drews_real = []
        drews = []
        lf_unique = len(frag[0])
        is_unique = True
        states1, acts1, next_states1, dones1 = [], [], [], []
        for f in frag:

            if is_unique:
                lf = len(f)
                if lf != lf_unique:
                    is_unique=False
            
            drews_real.append([None for _ in range(dataset.n_values)])
            if f.v_rews_real is not None:
                rews_real = f.v_rews_real
            else:
                rews_real = f.v_rews
                    
            v_rews.append(f.v_rews.T)
            v_rews_real.append(rews_real.T)
            drews.append([None for _ in range(dataset.n_values)])
            for vi in range(dataset.n_values):
                drews[-1][vi] = rollout.discounted_sum(f.v_rews[vi],  gamma=gamma)
                drews_real[-1][vi] = rollout.discounted_sum(rews_real[vi],  gamma=gamma)
            states1.append(f.obs[0:-1])
            acts1.append(f.acts)
            next_states1.append(f.obs[1:])
            dones1.append(f.dones)


        assert len(v_rews) == Ns
        assert len(v_rews_real) == Ns
        assert len(drews) == Ns
        assert len(drews_real) == Ns

        if is_unique:
            v_rews = np.asarray(v_rews)
            v_rews_real = np.asarray(v_rews_real)
            drews = np.asarray(drews)  
            drews_real = np.asarray(drews_real)
            states1 = np.array(states1)
            acts1 = np.array(acts1)
            next_states1 = np.array(next_states1)
            dones1 = np.array(dones1)
        else:
            v_rews = np.array(v_rews, dtype=list)
            v_rews_real = np.array(v_rews_real, dtype=list)
            drews = np.array(drews)
            drews_real = np.array(drews_real)
            states1 = np.array(states1, dtype=list)
            acts1 = np.array(acts1, dtype=list)
            next_states1 = np.array(next_states1, dtype=list)
            dones1 = np.array(dones1, dtype=list)

        assert v_rews.shape[0:1] == ( Ns, ), f"Rewards shape mismatch: {v_rews.shape} vs {(dataset.n_values, Ns)}"
        assert v_rews_real.shape[0:1] == ( Ns, ), f"Rewards shape mismatch: {v_rews_real.shape} vs {(dataset.n_values, Ns)}"
        assert drews.shape[0:2] == ( Ns, dataset.n_values), f"Rewards shape mismatch: {drews.shape} vs {(dataset.n_values, Ns)}"
        return (states1, acts1, next_states1, dones1), v_rews_real, v_rews, util.safe_to_tensor(drews_real, device=self.device), util.safe_to_tensor(drews, device=self.device)
    def train(self, total_timesteps, Ns, Nw, H, K, gamma_preferences, reward_train_callback, dataset = None, eval_env=None, ref_point=None, known_pareto_front=None, weight=None, total_episodes=None, reset_num_timesteps=True, eval_freq=10000, num_eval_weights_for_front=100, num_eval_episodes_for_front=5, num_eval_weights_for_eval=50, qualitative_preferences=True, reset_learning_starts=False, verbose=False, max_reward_buffer_size=None, reference_assignment: ClusterAssignment =None):
        self.running_assignment = reference_assignment
        for i, w in enumerate(reference_assignment.weights_per_cluster):
            print(f"Cluster {i}: {[p.requires_grad for p in w.parameters()]}")
            assert not any([p.requires_grad for p in w.parameters()]), "Weights in reference assignment must be frozen (requires_grad=False)."
            
        return super().train(total_timesteps, Ns, Nw, H, K, gamma_preferences, reward_train_callback, dataset, eval_env, ref_point, known_pareto_front, weight, total_episodes, reset_num_timesteps, eval_freq, num_eval_weights_for_front, num_eval_episodes_for_front, num_eval_weights_for_eval, qualitative_preferences, reset_learning_starts, verbose, max_reward_buffer_size)
    @override
    def overseer_new_preferences(self, Ns, Nw, H, gamma_preferences, max_reward_buffer_size, discordance_based_agent_selection=False, qualitative_preferences=True):
        
        with th.no_grad():
            assert self.running_assignment is not None, "Running assignment must be set before calling overseer_new_preferences."
            from_dataset_2 = False
            fragments1, fragments2, real_v_rew1, real_v_rew2, lear_v_rew1, lear_v_rew2, dsreal_v_rew1, dsreal_v_rew2, dslear_v_rew1, dslear_v_rew2  = self.replay_buffer.sample_trajs(ns=Ns, gamma=gamma_preferences,get_rewards_orig=True,device=self.device, to_tensor=True, H=H)
            if from_dataset_2:
                random_positions = np.random.choice(len(fragments1[0]), size=int(0.5*len(fragments1[0])), replace=False  )
                fragments2_, real_v_rew2_, lear_v_rew2_, dsreal_v_rew2_, dslear_v_rew2_ = self.sample_traj_from_dataset(self.base_dataset,Ns, gamma=gamma_preferences)
                for i in range(len(fragments2)):
                    fragments2[i][random_positions] = fragments2_[i][random_positions]
                real_v_rew2[random_positions] = real_v_rew2_[random_positions]
                lear_v_rew2[random_positions] = lear_v_rew2_[random_positions]
                dsreal_v_rew2[random_positions] = dsreal_v_rew2_[random_positions]
                dslear_v_rew2[random_positions] = dslear_v_rew2_[random_positions]

            assert real_v_rew1.shape == real_v_rew2.shape, f"Original rewards shape mismatch: {real_v_rew1.shape} vs {real_v_rew2.shape}"
            assert lear_v_rew1.shape == lear_v_rew2.shape, f"Rewards shape mismatch: {lear_v_rew1.shape} vs {lear_v_rew2.shape}"
            assert len(fragments1[0]) == len(fragments2[0]), f"Fragments shape mismatch: {len(fragments1)} vs {len(fragments2[0])}"
            assert len(real_v_rew1) == len(fragments1[0]), f"Original rewards and rewards length mismatch: {len(real_v_rew1)} vs {len(fragments1[0])}"
            assert len(real_v_rew2) == len(fragments2[0]), f"Original rewards and rewards length mismatch: {len(real_v_rew2)} vs {len(fragments2[0])}"
            assert dsreal_v_rew1.shape == (Ns, self.reward_dim), f"Rewards shape mismatch: {dsreal_v_rew1.shape} vs {(Ns, self.reward_dim)}"
            assert dsreal_v_rew2.shape == (Ns, self.reward_dim), f"Rewards shape mismatch: {dsreal_v_rew2.shape} vs {(Ns, self.reward_dim)}"
            assert dslear_v_rew1.shape == (Ns, self.reward_dim), f"Rewards shape mismatch: {dslear_v_rew1.shape} vs {(Ns, self.reward_dim)}"
            assert dslear_v_rew2.shape == (Ns, self.reward_dim), f"Rewards shape mismatch: {dslear_v_rew2.shape} vs {(Ns, self.reward_dim)}"

            #sampled_w = random_weights(self.reward_dim, Nw, dist="gaussian", rng=self.np_random)

            if max_reward_buffer_size < len(self.running_dataset) + Ns * Nw:
                    print("\033[95mPOPPPED",  len(self.running_dataset) - max_reward_buffer_size + Ns * Nw, "out of", len(self.running_dataset), "max", max_reward_buffer_size, "\033[0m")
                    self.running_dataset.pop(len(self.running_dataset) - max_reward_buffer_size + Ns * Nw)

            shuffled_assignments = np.random.permutation(np.asarray(self.running_assignment.assignments_vs, dtype=list))
            already_picked = set()
            counter = 0
            cluster_idx_pointer = 0
            while counter < Nw:
                candidates = shuffled_assignments[cluster_idx_pointer % self.running_assignment.Lmax]
                candidates = [c for c in candidates if c not in already_picked]
                cluster_idx_pointer+=1
                if len(candidates) == 0:
                    continue
                counter+=1
                if not discordance_based_agent_selection:
                    # Option 1: ask 6 (Nw) agents randomly from different clusters. Add that to the reward training dataset.
                    aname = np.random.choice(candidates)
                else:
                    # Option 1.1: focus on those that are worse represented.
                    max_discordance = float('-inf')
                    for an in candidates:
                        if self.running_assignment.intra_discordances_vs_per_agent[an] > max_discordance:
                            aname = an
                            max_discordance = self.running_assignment.intra_discordances_vs_per_agent[an]
                
                w = self.base_dataset.agent_data[aname]['value_system']
                wnp = w
                w = th.tensor(w).float().to(self.device)
                
                # dsreal_v_rew1 has shape [Ns, w.shape[0]]. apply dot over the last dimension
                assert dslear_v_rew1.shape == (Ns, self.reward_dim), f"Rewards shape mismatch: {dslear_v_rew1.shape} vs {(Ns, self.reward_dim)}"
                
                dsrw1 = (dsreal_v_rew1 * w).sum(dim=1)
                dsrw2 = (dsreal_v_rew2 * w).sum(dim=1)
                assert dsrw1.shape == dsrw2.shape == (Ns,), f"Rewards shape mismatch: {dsrw1.shape} vs {dsrw2.shape} vs {(Ns,)}"
                preferences_w = probs_to_label(probability_BT(dsrw1, dsrw2))
                
                preferences_with_grounding = th.zeros((len(preferences_w), self.reward_dim)).to(self.device)
                """if not th.all((dsrw1 < dsrw2) == (preferences_w < 0.5)):
                    print("Inconsistent preferences detected")
                    print("dsrw1", dsrw1)
                    print("dsrw2", dsrw2)
                    print("preferences_w", preferences_w)
                    for i in range(len(dsrw1)):
                        if (dsrw1[i] < dsrw2[i]) != (preferences_w[i] > 0.5):
                            print(f" Inconsistency at index {i}: dsrw1={dsrw1[i]}, dsrw2={dsrw2[i]}, preference={preferences_w[i]}")
                assert th.all((dsrw1 <= dsrw2) == (preferences_w <= 0.5)), "preferences_w should be 1 when dsrw1 > dsrw2"
                
                if not th.all((dsrw1 >= dsrw2) == (preferences_w >= 0.5)):
                    print("Inconsistent preferences detected")
                    print("dsrw1", dsrw1)
                    print("dsrw2", dsrw2)
                    print("preferences_w", preferences_w)
                    for i in range(len(dsrw1)):
                        if (dsrw1[i] >= dsrw2[i]) != (preferences_w[i] >= 0.5):
                            print(f" Inconsistency at index {i}: dsrw1={dsrw1[i]}, dsrw2={dsrw2[i]}, preference={preferences_w[i]}")
                assert th.all((dsrw1 >= dsrw2) == (preferences_w >= 0.5)), "preferences_w should be 1 when dsrw1 > dsrw2"
                    """
                    
                for vi in range(self.reward_dim):
                    preferences_with_grounding[:, vi] =  probs_to_label(probability_BT(dsreal_v_rew1[:,vi], dsreal_v_rew2[:,vi]))
                    #preferences_learned_w = probability_BT(lear_rew1, lear_rew2)
                fpairs_w = []
                #aname = '_synth_' + str(transform_weights_to_tuple(w))
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
                                                    rews_real=th.sum(real_v_rew1[i] * w, dim=1),n_vals=self.reward_dim, infos=[{'_envelope_weights': wnp}]*len(a1), 
                                                    agent=aname)
                    
                    f2 = TrajectoryWithValueSystemRews(obs=fobs2, 
                                                    acts=a2,
                                                    dones=d2, 
                                                    terminal=d2[-1],
                                                    v_rews=lear_v_rew2[i].T, 
                                                    rews=th.sum(util.safe_to_tensor(lear_v_rew2[i]) * w, dim=1),v_rews_real=real_v_rew2[i].T, 
                                                    rews_real=th.sum(util.safe_to_tensor(real_v_rew2[i]) * w, dim=1),n_vals=self.reward_dim, infos=[{'_envelope_weights': wnp}]*len(a2), 
                                                    agent=aname)
                    """if len(fobs1) == len(fobs2) and not from_dataset_2:
                        assert not (np.allclose(fobs2, fobs1) and np.allclose(a2, a1))
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
                self.running_dataset.push(np.asarray(fpairs_w, dtype=tuple), preferences=preferences_w, 
                                        preferences_with_grounding=preferences_with_grounding, 
                                        agent_data={aname: self.base_dataset.agent_data[aname]}, 
                                        agent_ids=[aname for _ in range(len(fpairs_w))])
                rt = time.time()
                print(f"\033[35mPush to running dataset took {rt-t:.4f} seconds\033[0m")
        
    def __sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)
    
    def sample_new_weights(self, n, random=False):
        if self.running_assignment is  None:
            weights = random_weights(dim=self.reward_dim, n=n, dist="gaussian", rng=self.np_random) 
        else:
            """if n > 1:
                    weights = np.random.choice(np.asarray(self.running_assignment.value_systems, dtype=tuple), size=min(n, len(self.running_assignment.value_systems)), replace=False)
                    if len(weights) < n:
                        weights.extend(random_weights(dim=self.reward_dim, n=n - len(weights), dist="gaussian", rng=self.np_random))
            else:
                
                idx = np.random.randint(0, len(self.running_assignment.value_systems))
                return self.running_assignment.value_systems[idx]"""
            weights =[] # 50% of times sample from running_assignment
            for _ in range(n):
                if self.cluster_weights_in_envelope_prob == 1.0:
                    # MODOPURE!!!!
                    idx = np.random.choice(len(self.running_assignment.value_systems_active))
                    w = self.running_assignment.value_systems_active[idx]
                elif np.random.rand() < self.cluster_weights_in_envelope_prob:
                    idx = np.random.randint(0, len(self.running_assignment.value_systems_active))
                    assert len(self.running_assignment.value_systems_active) == len([_ for asg in self.running_assignment.assignments_vs if len(asg) > 0 ])
                    
                    w = self.running_assignment.value_systems_active[idx]
                else:
                    w = random_weights(dim=self.reward_dim, n=1, dist="gaussian", rng=self.np_random)
                weights.append(w)
            assert len(weights) == n, f"Weights length mismatch: {len(weights)} vs {n}"
            if n==1:
                weights = weights[0]
        return weights
        
    """def update(self):
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

                weights = self.sample_new_weights(self.num_sample_w)
            
                sampled_w = (
                    th.tensor(weights)
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
                (
                        b_obs,
                        b_actions,
                        b_rewards,
                        b_next_obs,
                        b_dones,
                        w,
                    ) = self.__sample_batch_experiences()
                w= th.tensor(w).float().to(self.device)
                assert len(w) == len(b_obs)
                sampled_w = th.unique(w, dim=0)  # Just for logging purposes
                if sampled_w.ndim == 1:
                    sampled_w = sampled_w.unsqueeze(0)

            

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
            critic_loss = th.nn.functional.mse_loss(q_value, target_q)

            if self.homotopy_lambda > 0:
                wQ = th.einsum("br,br->b", q_value, w)
                wTQ = th.einsum("br,br->b", target_q, w)
                auxiliary_loss = th.nn.functional.mse_loss(wQ, wTQ)
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
                wandb.log({"metrics/mean_priority": np.mean(priority)})"""