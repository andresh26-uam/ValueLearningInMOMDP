from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
import enum
import itertools
import time
from typing import Any, Dict, Iterable, List, Sequence, Union, override
import imitation
import imitation.algorithms
import imitation.algorithms.base
from imitation.data.types import TrajectoryPair
import imitation.regularization
import imitation.regularization.regularizers

from imitation.util import util
from ordered_set import OrderedSet


#from src.algorithms.base_vsl_algorithm import BaseVSLAlgorithm
from src.dataset_processing.data import TrajectoryWithValueSystemRews, TrajectoryWithValueSystemRewsPair
from src.reward_nets.vsl_reward_functions import LinearAlignmentLayer, RewardVectorModule, TrainingModes
import torch as th

from imitation.algorithms.preference_comparisons import LossAndMetrics

from typing import (
    Optional,
    Sequence,
    Tuple,
)

from imitation.util import logger as imit_logger

import numpy as np
from imitation.data import rollout, types
from imitation.data.types import (
    TrajectoryPair,
    Transitions,
)
from imitation.algorithms import preference_comparisons

from src.utils import print_tensor_and_grad_fn


OPTION3 = False
class PrefLossClasses(enum.Enum):
    CROSS_ENTROPY_CLUSTER = 'cross_entropy_cluster'
    SOBA = 'soba'
    LAGRANGE = 'lagrange'



def probability_BT(x: th.Tensor, y: th.Tensor, threshold=50.0, with_logits=False) -> th.Tensor:
    #print("DIFF", th.max(x - y))
    assert isinstance(x, th.Tensor) and isinstance(y, th.Tensor), f"Expected th.Tensor, got {type(x)} and {type(y)}"
    returns_diff = x - y
    returns_diff = th.clip(returns_diff, -threshold, threshold)
    assert max(returns_diff) <= threshold and min(returns_diff) >= -threshold, f"Clipping failed: max {max(returns_diff)}, min {min(returns_diff)}, threshold {threshold}"
    ret = th.sigmoid(returns_diff)
    
    with th.no_grad():
        for i, p in enumerate(returns_diff):
            if th.allclose(p, th.tensor(0.0)):
            
                assert th.allclose(ret[i], th.tensor(0.5))
        assert not any(x.isnan()) and not any(x.isinf()), f"Input tensor x contains NaN or Inf values: {x}"
        assert not any(y.isnan()) and not any(y.isinf()), f"Input tensor y contains NaN or Inf values: {y}"
        assert not any(returns_diff.isnan()) and not any(returns_diff.isinf()), f"returns_diff is NaN or Inf. rews1: {x}, rews2: {y}, returns_diff: {returns_diff}"
        assert not any(ret.isnan()) and not any(ret.isinf()), f"Output prob ret contains NaN or Inf values: {ret}"
        
       # assert all(ret.detach()>=0.0)
        #assert all(ret.detach()<=1.0)
        #assert not any(returns_diff.isnan()), f"returns_diff is NaN. rews1: {x}, rews2: {y}, returns_diff: {returns_diff}"

    #assert th.isnan(y).sum() == 0, f"Input tensor y contains NaN values: {y}"
        
        assert th.allclose(ret, 1.0/(1.0 + th.exp(-returns_diff))), f"oh, {ret} vs {1.0/(1.0 + th.exp(-returns_diff))}" 
        assert isinstance(ret, th.Tensor), f"Expected th.Tensor, got {type(ret)}"
    if ret.requires_grad:
        assert ret.grad_fn is not None, "The returned tensor does not require gradients."
        assert ret.shape == returns_diff.shape, f"Output shape {ret.shape} does not match input shape {returns_diff.shape}"
    if with_logits:
        if returns_diff.requires_grad:
            assert returns_diff.grad_fn is not None, "The returned tensor does not require gradients."
        return ret, returns_diff
    return ret

def discordance(probs: th.Tensor = None, gt_probs: th.Tensor = None, indifference_tolerance=0.0, reduce='mean'):
    return 1.0 - calculate_accuracies(probs_vs=probs, gt_probs_vs=gt_probs, indifference_tolerance=indifference_tolerance, reduce=reduce, return_TAC=False)[0]
    
def calculate_accuracies(probs_vs: th.Tensor = None, probs_gr: th.Tensor = None, gt_probs_vs: th.Tensor = None, gt_probs_gr: th.Tensor = None, indifference_tolerance=0.0, reduce='mean', return_TAC=False):
    accuracy_vs = None
    accuracy_gr = None
    misclassified_vs = None
    misclassified_gr = None
    if probs_vs is not None:
        assert gt_probs_vs is not None
        if isinstance(probs_vs, th.Tensor):
            assert isinstance(gt_probs_vs , th.Tensor)
            detached_probs_vs = probs_vs.detach()
        else:
            assert isinstance(probs_vs , np.ndarray)
            assert isinstance(gt_probs_vs , np.ndarray)
            detached_probs_vs = probs_vs

        vs_predictions_positive = detached_probs_vs > 0.5
        vs_predictions_negative = detached_probs_vs < 0.5
        vs_predictions_indifferent = abs(
            detached_probs_vs - 0.5) <= indifference_tolerance

        if isinstance(gt_probs_vs, th.Tensor):
            gt_detached_probs_vs = gt_probs_vs.detach()
        else:
            gt_detached_probs_vs = gt_probs_vs

        
        gt_predictions_positive = gt_detached_probs_vs > 0.5
        gt_predictions_negative = gt_detached_probs_vs < 0.5
        gt_predictions_indifferent = gt_detached_probs_vs == 0.5
        
        
        wellclassified_positive = (
            gt_predictions_positive & vs_predictions_positive)
        wellclassified_negative = (
            gt_predictions_negative & vs_predictions_negative)
        wellclassified_indifferent = (
            gt_predictions_indifferent & vs_predictions_indifferent)
        
        # Combine all misclassified examples
        misclassified_vs = ~(wellclassified_positive | wellclassified_negative | wellclassified_indifferent)
        
        """if return_TAC:
            assert reduce is not None and reduce != "none", f"Reduction method must be specified as mean! {reduce}"
            P = th.sum(well_classified)
            Q = th.sum(misclassified_vs)
            X0 = th.sum(gt_predictions_indifferent & ~wellclassified_indifferent)
            Y0 = th.sum(vs_predictions_indifferent & ~wellclassified_indifferent)
            accuracy_vs =(P - Q) / (th.sqrt((P + Q + X0) * (P + Q + Y0)))
            assert accuracy_vs < 1.0 and accuracy_vs > -1.0, f"Accuracy must be in [-1, 1], got {accuracy_vs}"""
        if reduce == 'mean':
            accuracy_vs = (1.0 - sum(misclassified_vs)/len(probs_vs)) 
        else:
            accuracy_vs = 1.0 - misclassified_vs.float()

    if probs_gr is not None:
        assert gt_probs_gr is not None
        accuracy_gr = []
        misclassified_gr = []
        for j in range(probs_gr.shape[-1]):
            if isinstance(probs_gr, th.Tensor):
                detached_probs_vgrj = probs_gr[:, j].detach()
            else:
                detached_probs_vgrj = probs_gr[:, j]

            vgrj_predictions_positive = detached_probs_vgrj > 0.5
            vgrj_predictions_negative = detached_probs_vgrj < 0.5
            vgrj_predictions_indifferent = abs(
                detached_probs_vgrj - 0.5) <= indifference_tolerance
            
            
            if isinstance(gt_probs_gr, th.Tensor):
                gt_detached_probs_vgrj = gt_probs_gr[:, j].detach()
            else:
                gt_detached_probs_vgrj = gt_probs_gr[:, j]
            gt_predictions_positive = gt_detached_probs_vgrj > 0.5
            gt_predictions_negative = gt_detached_probs_vgrj < 0.5
            gt_predictions_indifferent = gt_detached_probs_vgrj == 0.5
            
            wellclassified_positive = (
                gt_predictions_positive & vgrj_predictions_positive)
            wellclassified_negative = (
                gt_predictions_negative & vgrj_predictions_negative)
            wellclassified_indifferent = (
                gt_predictions_indifferent & vgrj_predictions_indifferent)
            
            missclassified_vgrj = ~(wellclassified_positive | wellclassified_negative | wellclassified_indifferent)
            

            if reduce == 'mean':
                acc_gr_vi = (1.0 - sum(missclassified_vgrj)/len(detached_probs_vgrj))
            else:
                acc_gr_vi = 1.0 - missclassified_vgrj.float()

            accuracy_gr.append(acc_gr_vi)
            misclassified_gr.append(missclassified_vgrj)

    return accuracy_vs, accuracy_gr, misclassified_vs, misclassified_gr

def total_variation_distance(preferences1, preferences2):
    #return th.norm(preferences1-preferences2, p=p)
    return th.mean(th.square(preferences1 - preferences2))#/len(preferences1)

def jensen_shannon_pairwise_preferences(preferences1, preferences2, reduce = 'sum'):
    #return th.norm(preferences1-preferences2, p=p)
    m = (preferences1+preferences2)/2.0
    k1 = preferences1*(th.log(preferences1+1e-8) - th.log(m+1e-8)) + (1-preferences1)*(th.log(1-preferences1+1e-8) - th.log(1-m+1e-8))
    k2 = preferences2*(th.log(preferences2+1e-8) - th.log(m+1e-8)) + (1-preferences2)*(th.log(1-preferences2+1e-8) - th.log(1-m+1e-8))
    if reduce == 'sum':
        return th.sum((k1+k2)/2.0)
    else:
        return th.mean((k1+k2)/2.0)#/len(preferences1)

def likelihood_x_is_target(pred_probs, target_probs, mode='numpy', slope=1, adaptive_slope=True, qualitative_mode=False, indifference_tolerance=0.0):
    # SEE GEOGEBRA: https://www.geogebra.org/calculator/vdn3mj4k
    # to visualize the output probabilities used for the likelihood estimation that pred_probs would correspond to the modelled target_probs.
    # In general, you can use the slope to scale the probability differences, so the likelihood do not tend to 0 under bigger datasets.
    
    assert mode == 'numpy' or mode == 'th'
    minfun = th.min if mode == 'th' else np.minimum
    absfun = th.abs if mode == 'th' else np.abs
    productfun = th.prod if mode == 'th' else np.prod


    if qualitative_mode:
        probs = th.zeros_like(pred_probs) if mode == 'th' else np.zeros_like(pred_probs)
        probs [(pred_probs > 0.5) & (target_probs > 0.5)] = 1.0

        probs [(pred_probs < 0.5) & (target_probs < 0.5)] = 1.0
        
        probs[(absfun(pred_probs - target_probs) <= indifference_tolerance)] = 1.0
        return int(sum(probs))
    if adaptive_slope:
            # Here you can interpret the slope parameter as 1 - the minimum possible probability value that would be given to any input, i.e.
            # if slope = a, the minimum possible probability value would be 1 - a, for instance, if slope = 0.3, the minimum possible probability value would be 0.7
            # If slope is bigger than 1, the minimum possible probability value would be 0, in a segment of the input space bigger as the slope increases.
            probs = 1 - slope * \
                minfun(1/(target_probs+1e-8), 1/(1-(target_probs+1e-8))) * \
                absfun(target_probs - pred_probs)
    else:
            # Here the slope is the slope of the linear , and it will always give bigger or equal probabilities than the adaptive slope. It is then, more lax but "unfair" method
            probs = 1 - slope*absfun(target_probs - pred_probs)
    
    
    return productfun(probs)


class PreferenceModelClusteredVSL(preference_comparisons.PreferenceModel):
    """Class to convert two fragments' rewards into preference probability."""
    """
    Extension of https://imitation.readthedocs.io/en/latest/algorithms/preference_comparisons.html
    """

    def _slice_all_transitions_into_pairs(v, K):
        """This function returns len(v)/K slices of v of length K,the even slices first, the odd slices second, 

        Args:
            v (np.ndarray): Vector
            K (int): length desired of each slice

        Returns:
            Tuple[np.ndarray, np.ndarray]: Odd slices, even slices.
        """
        # Reshape the vector into N slices of length K
        reshaped_v = v.reshape((-1, K))
        # Calculate the slices we need: indexes K+1 to 2K+1, etc.
        odd_slices = reshaped_v[1::2]
        even_slices = reshaped_v[0::2]
        return even_slices, odd_slices

    def __init__(
        self,
        model: RewardVectorModule,
        algorithm,#: BaseVSLAlgorithm,
        noise_prob: float = 0.0,
        discount_factor: float = 1.0,
        threshold: float = 50,
    ) -> None:
        super().__init__(model, noise_prob, discount_factor, threshold)
        self.algorithm = algorithm

        self.dummy_models = []

    
    def forward(
        self,
        fragment_pairs_per_agent_id: Dict[str, Sequence[TrajectoryWithValueSystemRewsPair]],
        fragment_pairs_idxs_per_agent_id: Dict[str, np.ndarray],
        custom_model_per_agent_id: Union[RewardVectorModule,
                                         Dict[str, RewardVectorModule]] = None,

        only_for_alignment_function=None,
        only_grounding=False,
        return_rewards_per_agent=False,
        return_rewards_global = True,
        add_probs_per_agent=False
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor], Optional[th.Tensor]]:
        """Computes the preference probability of the first fragment for all pairs.

        """
        
        dtype = self.algorithm.reward_net.dtype
            
        if custom_model_per_agent_id is not None:
            prev_model = self.model

            if isinstance(custom_model_per_agent_id, dict):
                models_per_agent: Dict = custom_model_per_agent_id
            else:
                models_per_agent = {}

        total_number_of_fragment_pairs = sum(
            [len(fpair) for fpair in fragment_pairs_per_agent_id.values()])

        probs_vs = None
        probs_gr = None
        if not only_grounding:
            probs_vs = th.zeros(total_number_of_fragment_pairs, dtype=dtype)
        if only_for_alignment_function is None:
            probs_gr = th.zeros(
                (total_number_of_fragment_pairs, self.algorithm.env.n_values), dtype=dtype)

        probs_vs_per_aid = dict()
        probs_gr_per_aid = dict()

        if return_rewards_per_agent:
            rews_vs_per_aid = dict()
            rews_gr_per_aid = dict()
        rews_vs = None
        rews_gr = None

        counter_idx = 0

        #  TODO... Fragments per agent id... But remain in the order said by fragment_pairs
        n_values = 0
        idx_global = 0
        for aid, fragment_pairs_aid in fragment_pairs_per_agent_id.items():
            if n_values == 0:
                n_values = fragment_pairs_aid[0][0].value_rews.shape[0]

            model = models_per_agent.get(aid, custom_model_per_agent_id)
            n_fragments = len(fragment_pairs_aid)
            fragment_size = len(fragment_pairs_aid[0][0])
            
            if idx_global == 0:
                if return_rewards_global:
                    rews_vs = (th.empty(total_number_of_fragment_pairs, fragment_size, dtype=dtype), th.empty(total_number_of_fragment_pairs, fragment_size, dtype=dtype))
                    rews_gr = (th.empty(
                        (total_number_of_fragment_pairs, fragment_size, self.algorithm.env.n_values), dtype=dtype), th.empty(
                        (total_number_of_fragment_pairs, fragment_size, self.algorithm.env.n_values), dtype=dtype))
            

            probs_vs_per_aid[aid] = th.empty(n_fragments, dtype=dtype)
            if only_for_alignment_function is None:
                probs_gr_per_aid[aid] = th.zeros(
                    (n_fragments, n_values), dtype=dtype)
                if return_rewards_per_agent:
                    rews_vs_per_aid[aid] = None # tuple initialized later.
                    rews_gr_per_aid[aid] = (th.empty(
                    (n_fragments, fragment_size, n_values), dtype=dtype), th.empty(
                    (n_fragments, fragment_size, n_values), dtype=dtype))
            with th.no_grad():
                all_transitions_aid = rollout.flatten_trajectories(
                    [frag for fragment in fragment_pairs_aid for frag in fragment])
                
            all_rews_vs_aid, all_rews_gr_aid = self.rewards(
                all_transitions_aid, only_with_alignment=only_for_alignment_function is not None, alignment=only_for_alignment_function, custom_model=model, only_grounding=only_grounding)
            """if all_rews_gr_aid is not None: 
                print("RW??", all_rews_gr_aid[0:10])
                print("VS??", all_rews_vs_aid[0:10])
"""
            idx = 0

            if self.algorithm.allow_variable_horizon:
                raise ValueError("You should overhaul this... Like padding with 0 maybe?")
                for iad, (f1, f2) in enumerate(fragment_pairs_aid):

                    rews1_vsaid, rews1_graid = all_rews_vs_aid[idx:idx+len(
                        f1)], all_rews_gr_aid[:, idx:idx+len(f1)]
                    assert np.allclose(
                        f1.acts, all_transitions_aid[idx:idx+len(f1)].acts)

                    idx += len(f1)
                    rews2_vsaid, rews2_graid = all_rews_vs_aid[idx:idx+len(
                        f2)], all_rews_gr_aid[:, idx:idx+len(f2)]
                    assert np.allclose(
                        f2.acts, all_transitions_aid[idx:idx+len(f2)].acts)

                    idx += len(f2)
                    if not only_grounding:
                        probs_vs_per_aid[aid][iad] = self.probability(
                            rews1_vsaid, rews2_vsaid)
                    if only_for_alignment_function is None:
                        for j in range(rews1_graid.shape[0]):
                            probs_gr_per_aid[aid][iad, j] = self.probability(
                                rews1_graid[j], rews2_graid[j])
            else:
                
                if not only_grounding:
                    rews1_vsaid_all, rews2_vsaid_all = PreferenceModelClusteredVSL._slice_all_transitions_into_pairs(
                        all_rews_vs_aid, fragment_size)
                    if return_rewards_global:
                        
                        rews_vs[0][idx_global:idx_global+len(rews1_vsaid_all),:] = rews1_vsaid_all
                        rews_vs[1][idx_global:idx_global+len(rews2_vsaid_all),:] = rews2_vsaid_all
                        

                    probs_vs_per_aid[aid] = self.probability(
                        rews1_vsaid_all, rews2_vsaid_all)
                    if return_rewards_per_agent:
                        rews_vs_per_aid[aid] = (rews1_vsaid_all, rews2_vsaid_all)

                if only_for_alignment_function is None:
                    for j in range(n_values):
                        rews1_graid_all_j, rews2_graid_all_j = PreferenceModelClusteredVSL._slice_all_transitions_into_pairs(
                            all_rews_gr_aid[j], fragment_size)
                        probs_gr_per_aid[aid][:, j] = self.probability(
                            rews1_graid_all_j, rews2_graid_all_j)
                        
                        if return_rewards_global:
                            rews_gr[0][idx_global:idx_global+len(rews1_graid_all_j),:,j] = rews1_graid_all_j
                            rews_gr[1][idx_global:idx_global+len(rews2_graid_all_j),:,j] = rews2_graid_all_j
                        if return_rewards_per_agent:
                            rews_gr_per_aid[aid][0][:, :, j] = rews1_graid_all_j
                            rews_gr_per_aid[aid][1][:, :, j] = rews2_graid_all_j
                            assert rews_gr_per_aid[aid][0].shape == (len(fragment_pairs_aid), fragment_size, n_values)
                idx_global += len(rews1_vsaid_all)
            with th.no_grad():    
                if fragment_pairs_idxs_per_agent_id is not None:
                    fragment_idxs = fragment_pairs_idxs_per_agent_id[aid] #TODO optimize later to index rews_gr_per_aid
                else:
                    # just put them in order of appearance of each agent. 
                    fragment_idxs = np.array(list(range(n_fragments))) + counter_idx
            if not only_grounding:
                probs_vs[fragment_idxs] = probs_vs_per_aid[aid]
            if only_for_alignment_function is None:
                probs_gr[fragment_idxs] = probs_gr_per_aid[aid]
            counter_idx += n_fragments
        
        if custom_model_per_agent_id is not None:
            self.model = prev_model

        if return_rewards_per_agent is False:
            if add_probs_per_agent:
                return probs_vs, probs_gr, probs_vs_per_aid, probs_gr_per_aid
            else:
                return probs_vs, probs_gr,
        else:
            if add_probs_per_agent:
                return probs_vs, probs_gr, probs_vs_per_aid, probs_gr_per_aid, rews_vs_per_aid, rews_gr_per_aid, rews_vs, rews_gr
            else:
                return probs_vs, probs_gr, rews_vs_per_aid, rews_gr_per_aid, rews_vs, rews_gr

    def rewards(self, transitions: Transitions, only_with_alignment=False, only_grounding=False, real=False, alignment=None, grounding=None, custom_model=None, reward_mode=None) -> th.Tensor:
        """Computes the reward for all transitions.

        Args:
            transitions: batch of obs-act-obs-done for a fragment of a trajectory.

        Returns:
            The reward given by the network(s) for all the transitions.
            Shape - (num_transitions, ) for Single reward network and
            (num_transitions, num_networks) for ensemble of networks.
        """
        if reward_mode is None:
            reward_mode = self.algorithm.training_mode
        if custom_model is not None:
            prev_model = self.model
            self.model: AbstractVSLRewardFunction = custom_model

        state = None
        action = None
        next_state = None
       
        state = util.safe_to_tensor(
                transitions.obs, device=self.model.device, dtype=self.model.dtype)
        action = util.safe_to_tensor(
                transitions.acts, device=self.model.device, dtype=self.model.dtype)
        if self.model.use_next_state:
            next_state = util.safe_to_tensor(
            transitions.next_obs, device=self.model.device, dtype=self.model.dtype)
        
        info = transitions.infos[0] 


        grouped_transitions = {}
        if 'context' in info.keys():
            # group by context...
            # Initialize a dictionary to store grouped transitions
            # Iterate through the transitions and group them by context
            grouped_transitions = defaultdict(list)
            for iinf, info_ in enumerate(transitions.infos):
                grouped_transitions[info_['context']].append(iinf)
        
        else:
            grouped_transitions['no-context1'] = list(range(len(transitions)))

        
        rews = th.zeros((len(transitions.obs), ), device=self.model.device, dtype=self.model.dtype)
        rews_gr = th.zeros((self.algorithm.env.n_values, len(transitions.obs)), device=self.model.device, dtype=self.model.dtype)

        if self.model.mode == TrainingModes.VALUE_GROUNDING_LEARNING and alignment is None:
                # This is a dummy alignment function when we only want the grounding preferences. We set it as tuple to be sure no chages are made here.
                alignment = tuple(self.model.get_learned_align_function())

        #indexes_so_far = []
        for context, indexes in grouped_transitions.items():
            info = transitions.infos[indexes[0]]

            """for i in indexes:
                assert i not in indexes_so_far
            indexes_so_far.extend(indexes)"""

            #self.algorithm.env.contextualize(context) TODO?
            #th.testing.assert_close(rews[indexes], th.zeros((len(deepcopy(states_i)),), device=self.model.device, dtype=self.model.dtype))
            #th.testing.assert_close(rews_gr[:,indexes], th.zeros((self.algorithm.env.n_values, len(deepcopy(states_i))), device=self.model.device, dtype=self.model.dtype))

            # done = transitions.dones
            states_i = state[indexes] if state is not None else None
            action_i = action[indexes] if action is not None else None
            next_states_i = next_state[indexes] if next_state is not None else None

            if only_with_alignment:
                rews[indexes], np_rewards = self.algorithm.calculate_rewards(alignment if reward_mode == TrainingModes.VALUE_GROUNDING_LEARNING else None,
                                                                            custom_model=self.model,
                                                                            grounding=None,
                                                                            obs_mat=states_i,
                                                                            action_mat=action_i,
                                                                            next_state_obs_mat=next_states_i,
                                                                            obs_action_mat=None,  # TODO?
                                                                            reward_mode=reward_mode,
                                                                            recover_previous_config_after_calculation=False,
                                                                            use_probabilistic_reward=False, requires_grad=True, forward_groundings=False,
                                                                            info=info)

                if real:
                    _, rews[indexes] = self.algorithm.calculate_rewards(alignment,
                                                                        custom_model=self.model,
                                                                        grounding=grounding,
                                                                        obs_mat=states_i,
                                                                        action_mat=action_i,
                                                                        next_state_obs_mat=next_states_i,
                                                                        obs_action_mat=None,
                                                                        reward_mode=TrainingModes.EVAL,
                                                                        recover_previous_config_after_calculation=True,
                                                                        use_probabilistic_reward=False, requires_grad=False, info=info)
                    
                
                    

                
            else:
                rews[indexes], np, rews_gr[:, indexes], np2 = self.algorithm.calculate_rewards(alignment if reward_mode == TrainingModes.VALUE_GROUNDING_LEARNING else None,
                                                                            custom_model=self.model,
                                                                            grounding=None if reward_mode == TrainingModes.VALUE_GROUNDING_LEARNING else grounding,
                                                                            obs_mat=states_i,
                                                                            action_mat=action_i,
                                                                            next_state_obs_mat=next_states_i,
                                                                            obs_action_mat=None,  # TODO?
                                                                            reward_mode = reward_mode,
                                                                            recover_previous_config_after_calculation=False,
                                                                            use_probabilistic_reward=False, requires_grad=True, info=info, forward_groundings=True)

                

        
        if custom_model is not None:
            self.model = prev_model
            
        if only_with_alignment:
            return rews, None
    
        #assert len(state) == len(action)
        if not only_grounding:
            assert rews.shape == (len(transitions.obs),)
        assert rews_gr.shape == (
            self.algorithm.env.n_values, len(transitions.obs))
        
        
        return rews, rews_gr

    def probability(self, rews1: th.Tensor, rews2: th.Tensor) -> th.Tensor:
        """Changed from imitation the ability to compare fragments of different lengths
        """
        # First, we compute the difference of the returns of
        # the two fragments. We have a special case for a discount
        # factor of 1 to avoid unnecessary computation (especially
        # since this is the default setting). The else part is for calculating probability of batches of pairs of trajectories.
        if len(rews1.shape) <= 1:
            if self.discount_factor == 1:
                if len(rews1) == len(rews2):
                    returns_diff = (rews2 - rews1).sum(axis=0)
                else:
                    returns_diff = rews2.sum(axis=0) - rews1.sum(axis=0)
            else:
                device = rews1.device
                assert device == rews2.device
                l1, l2 = len(rews1), len(rews2)
                discounts = self.discount_factor ** th.arange(
                    max(l1, l2), device=device)
                if self.ensemble_model is not None:
                    discounts = discounts.reshape(-1, 1)
                if len(rews1) == len(rews2):
                    returns_diff = (discounts * (rews2 - rews1)).sum(axis=0)
                else:
                    returns_diff = (
                        discounts[0:l2] * rews2).sum(axis=0) - (discounts[0:l1] * rews1).sum(axis=0)
        else:
            # Batched calculation. Need to have same length trajectories
            assert rews1.shape == rews2.shape
            if self.discount_factor < 1.0:
                device = rews1.device
                assert device == rews2.device

                if not hasattr(self, 'cached_discounts') or len(self.cached_discounts) < rews1.shape[1]:
                    self.cached_discounts = self.discount_factor ** np.arange(
                        rews1.shape[1])

                returns_diff = (self.cached_discounts *
                                (rews2 - rews1)).sum(axis=1)
            else:
                returns_diff = (rews2 - rews1).sum(axis=1)

            assert returns_diff.shape == (rews1.shape[0],)

        # Clip to avoid overflows (which in particular may occur
        # in the backwards pass even if they do not in the forward pass).
        returns_diff = th.clip(returns_diff, -self.threshold, self.threshold)
        # We take the softmax of the returns. model_probability
        # is the first dimension of that softmax, representing the
        # probability that fragment 1 is preferred.
        assert not any(returns_diff.isnan()), f"returns_diff is NaN. rews1: {rews1}, rews2: {rews2}, returns_diff: {returns_diff}"
        probability = 1.0 / (1.0 + returns_diff.exp())
        if self.noise_prob > 0:
            probability = self.noise_prob * 0.5 + \
                (1 - self.noise_prob) * probability

        return probability


class BasicRewardTrainerVSL(preference_comparisons.BasicRewardTrainer):
    """Train a basic reward model."""

    def __init__(
        self,
        preference_model: preference_comparisons.PreferenceModel,
        n_values: int,
        loss: preference_comparisons.RewardLoss,
        rng: np.random.Generator,
        batch_size: int = 32,
        minibatch_size: Optional[int] = None,
        epochs: int = 1,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        regularizer_factory: Optional[imitation.regularization.regularizers.RegularizerFactory] = None,
        optim_cls=th.optim.AdamW,
        optim_kwargs=dict(lr=1e-3, weight_decay=0.0)
    ) -> None:
        lr = optim_kwargs['lr']
        
        super().__init__(preference_model, loss, rng, batch_size,
                         minibatch_size, epochs, lr, custom_logger, regularizer_factory)
        # Maybe neeed... WEIGHT DECAY MUST BE 0, otherwise not affected networks may decay without reason!!!!!!!
        # optim_kwargs['weight_decay'] = 0.0
        self.optim_kwargs = optim_kwargs
        self.optim_cls = optim_cls
        self.n_values= n_values

        # Default. This should be overriden.
        if not issubclass(self.optim_cls, VSLOptimizer):
            self.reset_optimizer_with_params(
                parameters=self._preference_model.parameters())
        else:
            self.reset_optimizer_with_params(
                parameters={'params_gr': OrderedSet(self._preference_model.parameters()), 'params_vs': OrderedSet(self._preference_model.parameters())})
            

        self.regularizer = (
            regularizer_factory(optimizer=self.optim, logger=self.logger)
            if regularizer_factory is not None
            else None
        )

    def reset_optimizer_with_params(self, parameters):
        # Maybe needed... WEIGHT DECAY MUST BE 0, otherwise not affected networks may decay without reason!!!!!!!
        #assert self.optim_kwargs['weight_decay'] == 0.0, f"Weight decay must be 0, otherwise not affected networks may decay without reason. Current value: {self.optim_kwargs['weight_decay']}"

        if isinstance(parameters, dict):
            self.optim = self.optim_cls(params_gr=OrderedSet(parameters['params_gr']), params_vs=OrderedSet(parameters['params_vs']), n_values=self.n_values, **self.optim_kwargs)
        else:
            self.optim = self.optim_cls(parameters, **self.optim_kwargs)


def nth_derivative(f, wrt, n):
    all_grads = []
    for i in range(n):

        grads = th.autograd.grad(f, wrt, create_graph=i <n-1, retain_graph=True)
        all_grads.append(grads)
        f = th.sum(th.cat(tuple(g.flatten() for g in grads)))

    return all_grads
def derivative21(f, wrt1, wrt2, dw2 = None, retain_last = True):
    if dw2 is None:
        grads = th.autograd.grad(f, wrt2, create_graph=True, retain_graph=True, allow_unused=True, materialize_grads=True)
        if grads is None:
            return None, None
    else:
        grads = dw2
    f = th.sum(th.cat(tuple(g.flatten() for g in grads)))
    grads2 = th.autograd.grad(f, wrt1, retain_graph=True, create_graph=True, allow_unused=True, materialize_grads=True)
    
    return grads, grads2

def gradients_soba_eff(wx, wy, vt, goal1_x, goal2_xy):
    
    grads_G_wrt_wx = th.autograd.grad(goal1_x, wx, create_graph=True, retain_graph=True, allow_unused=True, materialize_grads=True)
    #f = th.sum(th.cat(tuple(g.flatten() for g in grads_G_wrt_wx)))
    
    #grads_G_wrt_wx2 = th.autograd.grad(f, wx, create_graph=False, retain_graph=True, allow_unused=True, materialize_grads=True)

    #grads_F_wrt_wx = th.autograd.grad(goal2_xy, wx, retain_graph=True, create_graph=True,allow_unused=True, materialize_grads=True)
    grads_F_wrt_wy = th.autograd.grad(goal2_xy, wy, retain_graph=False, allow_unused=True, materialize_grads=True)

    Dtheta = grads_G_wrt_wx
    Dlambda = grads_F_wrt_wy
    
    """for fwx, qwxwx, vt_i in zip(grads_F_wrt_wx,grads_G_wrt_wx2, vt):
        Dv.append(fwx + qwxwx*vt_i)"""
#grad_F_wrt_wy = torch.autograd.grad(goal2_xy, wy, retain_graph=False, create_graph=False)[0]

    #torch.testing.assert_close(grad_G_wrt_wywx, torch.zeros_like(grad_G_wrt_wywx))
    #torch.testing.assert_close(grad_G_wrt_wy, torch.zeros_like(grad_G_wrt_wy))
    return Dtheta,None, Dlambda

def gradients_soba(wx, wy, vt, goal1_x, goal2_xy):
    
    grad_G_wrt_wx, grad_G_wrt_wx2 = nth_derivative(goal1_x, wx, 2)
    grad_G_wrt_wy, grad_G_wrt_wywx  = derivative21(goal1_x, wx, wy, retain_last = False)
    
    grad_F_wrt_wx = th.autograd.grad(goal2_xy, wx, retain_graph=True, create_graph=True,allow_unused=True, materialize_grads=True)
    grad_F_wrt_wy = th.autograd.grad(goal2_xy, wy, retain_graph=False, create_graph=True, allow_unused=True, materialize_grads=True)

    
    if grad_G_wrt_wywx is None:
        Dlambda = grad_F_wrt_wy
    else:
        Dlambda = []
        cum_sum = sum(th.sum(gryx.mul(vt_i)) for gryx, vt_i in zip(grad_G_wrt_wywx, vt))
        for p in grad_F_wrt_wy:
            Dlambda.append(cum_sum + p) 

    Dv = []
    cum_sum2 = sum(th.sum(grx2.mul(vt_i)) for grx2, vt_i in zip(grad_G_wrt_wx2, vt))
    for p in grad_F_wrt_wx:
        Dv.append(cum_sum2 + p) 

    Dtheta = grad_G_wrt_wx
    
    return Dtheta,Dv, Dlambda


class BaseVSLClusterRewardLoss(preference_comparisons.RewardLoss):
    
    def gradients(self, scalar_loss: th.Tensor, renormalization: float) -> None:
        scalar_loss *= renormalization
        return scalar_loss.backward()
    
    def set_parameters(self, *params) -> None:
        pass

    def __init__(self, model_indifference_tolerance, gr_apply_on_misclassified_pairs_only=False, 
                         vs_apply_on_misclassified_pairs_only=False, confident_penalty=0.0,
                           label_smoothing = 0.0, cluster_similarity_penalty=0.00,
                           conciseness_penalty_reduction='min',
                           missclassification_min_weighting=0.3,per_agent=True) -> None:
        """Create cross entropy reward loss."""
        super().__init__()
        # This is the tolerance in the probability model when in the ground truth two trajectories are deemed equivalent (i.e. if two trajectories are equivalent, the ground truth target is 0.5. The model should output something in between (0.5 - indifference, 0.5 + indifference) to consider it has done a correct preference prediction.)
        self.model_indifference_tolerance = model_indifference_tolerance
        self.gr_apply_on_misclassified_only = gr_apply_on_misclassified_pairs_only
        self.vs_apply_on_misclassified_only = vs_apply_on_misclassified_pairs_only
        self.confident_penalty = confident_penalty
        self.cluster_similarity_penalty = cluster_similarity_penalty
        self.conciseness_penalty_reduction =  conciseness_penalty_reduction
        self.label_smoothing = label_smoothing
        self.missclassification_min_weighting = missclassification_min_weighting
        self.per_agent = per_agent


    def loss_func(self, probs, target_probs, misclassified_pairs=None, apply_on_misclassified_only=False, logits=None):
        if misclassified_pairs is not None and apply_on_misclassified_only:
            probs_l = probs[misclassified_pairs == True]
            target_probs_l = target_probs[misclassified_pairs == True]
            logits_l = logits[misclassified_pairs == True] if logits is not None else None
        else:
            probs_l = probs
            logits_l = logits

            target_probs_l = target_probs
        if len(probs_l) == 0:
            raise ValueError("No probabilities to compute loss on. Check if the misclassified pairs are correct or if the probabilities are empty.")
            return th.tensor(0.0, device=probs.device, dtype=probs.dtype)

        if self.label_smoothing > 0.0:
            with th.no_grad():
                #IMPORTANT LABEL SMOOTHING: label_smoothing aas specified in https://arxiv.org/pdf/1512.00567
                target_probs_l = target_probs_l * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        weights = None
        #diffs = th.abs(logits_l - target_probs_l)
        #assert not any(diffs.isnan()), f"probs_l: {probs_l}, target_probs_l: {target_probs_l}, diffs: {diffs}, logits_l: {logits_l}"
        #assert not any(diffs.isinf()), f"probs_l: {probs_l}, target_probs_l: {target_probs_l}, diffs: {diffs}, logits_l: {logits_l}"

        if self.missclassification_min_weighting < 1.0 and not apply_on_misclassified_only:
            weights = th.full_like(probs_l, 1.0, requires_grad=False)
            """min_w = self.missclassification_min_weighting
            max_w = 1.0/min_w
            with th.no_grad():
                weights = th.full_like(probs_l, 1.0, requires_grad=False)
                # weights add up to len(probs_l) Have the weights is for misclassified and half for classified.
                mx = sum(misclassified_pairs)
                n = len(probs_l)
                miss_classified_proportion = (mx/n)
                w = 1.0 if min_w ==1.0 else (1 - miss_classified_proportion) / (miss_classified_proportion + 1/(max_w-min_w)) + min_w
                weights[misclassified_pairs] = w
                weights[~misclassified_pairs] = 1.0 if min_w ==1.0 else (miss_classified_proportion) / (1.0-miss_classified_proportion + 1/(max_w-min_w)) + min_w
            """
            weights[~misclassified_pairs] = self.missclassification_min_weighting
            closs =  th.nn.functional.binary_cross_entropy_with_logits(logits_l, target_probs_l, reduction='none', weight=weights)#/sum(weights)
        else:
            closs =  th.nn.functional.binary_cross_entropy_with_logits(logits_l, target_probs_l, reduction='none', weight=weights)
            # if apply_on_misclassified_only else len(probs_l)
        #assert closs.shape == probs_l.shape, f"closs shape {closs.shape} != probs_l shape {probs_l.shape}"
        
        #closs2 =  (-th.log(probs_l) * target_probs_l - th.log(1.0 - probs_l) * (1.0 - target_probs_l)) * weights
        if self.confident_penalty > 0.0:
            p =self.confident_penalty*th.multiply(probs_l, th.log(th.max(probs_l, th.full_like(probs_l, 1e-5))))
            assert not any(p.isnan()), f"probs_l: {probs_l}, target_probs_l: {target_probs_l}"
            closs = th.mean(closs + p)
        else:
            closs = th.mean(closs)
        return closs
        
    def simple_forward(self, reward_vector_module: RewardVectorModule, weights_per_cluster: Dict[int, LinearAlignmentLayer], fragments1, fragments2, cluster_fidxs):

        grounding1 = reward_vector_module.forward(fragments1)
        grounding2 = reward_vector_module.forward(fragments2)

        probs_grounding = th.empty_like(grounding1)
        probs_value_system = th.empty_like(grounding1[:, 0])
        logits_grounding = th.empty_like(grounding1)
        logits_value_system = th.empty_like(grounding1[:, 0])
        for vi in range(reward_vector_module.num_outputs):
            pbt, logits = probability_BT(grounding1[:, vi], grounding2[:, vi], with_logits=True)
            assert pbt.shape == probs_grounding[:, vi].shape
            probs_grounding[:, vi] = pbt
            logits_grounding[:, vi] = logits

        for cluster, fidxs in enumerate(cluster_fidxs):
            if len(fidxs) == 0:
                continue
            
            w = weights_per_cluster[cluster]
            probs, logits = probability_BT(w.forward(grounding1[fidxs]), w.forward(grounding2[fidxs]), with_logits=True)
            assert probs.shape[0] == probs_value_system[fidxs].shape[0], (probs.shape, probs_value_system[fidxs].shape)
            probs_value_system[fidxs] = probs.squeeze(-1)
            logits_value_system[fidxs] = logits.squeeze(-1)

        return grounding1, grounding2, probs_grounding.requires_grad_(), probs_value_system.requires_grad_(), logits_grounding.requires_grad_(), logits_value_system.requires_grad_()
    def forward(
        self,
        fragments1: Union[Tuple[th.Tensor], List[TrajectoryWithValueSystemRews]],
        fragments2: Union[Tuple[th.Tensor], List[TrajectoryWithValueSystemRews]],
        reward_vector_module: RewardVectorModule,
        weights_per_cluster: Dict[int, LinearAlignmentLayer],
        cluster_fidxs: List[List[int]],
        preferences: np.ndarray, preferences_with_grounding: np.ndarray,
        fidxs_per_agent: Dict[str, List[int]] = None,

    ) -> preference_comparisons.LossAndMetrics:
        
        grounding1, grounding2, probs_gr, probs_vs, logits_grounding, logits_value_system = self.simple_forward(reward_vector_module=reward_vector_module,
                                                                                          weights_per_cluster=weights_per_cluster,
                                                                                          fragments1=fragments1, fragments2=fragments2, cluster_fidxs=cluster_fidxs)

        """
        This might make sense too... Summin over all cluster fragments directly
        fragments_idxs_per_cluster = [[] for _ in range(len(value_system_network_per_cluster))]
        for aid, cidx in agent_to_vs_cluster_assignments.items(): 
            fragments_idxs_per_cluster[cidx].extend(fragment_idxs_per_aid[aid])
        print(f"Fragments per cluster: {fragments_idxs_per_cluster[0]}")
        exit(0)
        """
        with th.no_grad():
            assert probs_gr.shape == preferences_with_grounding.shape
            assert probs_vs.shape == preferences.shape
            accuracy_vs, accuracy_gr, missclassified_vs, missclassified_gr = calculate_accuracies(
                probs_vs, probs_gr, preferences, preferences_with_grounding, indifference_tolerance=self.model_indifference_tolerance, return_TAC=False)
            #accuracy_vs_per_agent, accuracy_gr_per_agent, missclassified_vs_per_agent, missclassified_gr_per_agent = {}, {}, {}, {}
            """for aid in preferences_per_agent_id.keys():
            accuracy_vs_per_agent[aid], accuracy_gr_per_agent[aid], missclassified_vs_per_agent[aid], missclassified_gr_per_agent[aid] = calculate_accuracies(
                probs_vs_per_agent[aid], probs_gr_per_agent[aid], preferences_per_agent_id[aid], preferences_per_agent_id_with_grounding[aid], indifference_tolerance=self.model_indifference_tolerance)
            accuracy_gr_per_agent[aid] = np.array(accuracy_gr_per_agent[aid])
            accuracy_vs_per_agent[aid] = float(accuracy_vs_per_agent[aid])"""

            self.last_accuracy_gr = np.array(accuracy_gr)

            metrics = {}
            metrics["global_accvs"] = accuracy_vs
            metrics["global_accgr"] = np.array(accuracy_gr)

            #metrics["accvs"] = accuracy_vs_per_agent
            #metrics["accgr"] = accuracy_gr_per_agent
            metrics['loss_per_vi'] = {}
            # misclassified_pairs = predictions != ground_truth

            metrics = {key: value.detach().cpu() if isinstance(
                value, th.Tensor) else value for key, value in metrics.items()}

        # LOSS VALUE SYSTEM.
        loss_vs = th.tensor(
                0.0, device=probs_vs.device, dtype=probs_vs.dtype)
        
        # This is the global approach. Does not optimize for the true representativity (works only if all agents rank same number of trajectories!!!) 
        #
        n_agents = len(fidxs_per_agent)
        if self.per_agent:
            for ag, fidxs_ag in fidxs_per_agent.items():
                if len(fidxs_ag) == 0:
                    raise ValueError(f"Agent {ag} has no fragments to compute loss on. Check if the agent has fragments in the dataset or if the fidxs_per_agent is correct.")
                
                loss_vs += self.loss_func(probs_vs[fidxs_ag], preferences[fidxs_ag], misclassified_pairs=missclassified_vs[fidxs_ag], apply_on_misclassified_only=self.vs_apply_on_misclassified_only, logits=logits_value_system[fidxs_ag])
            loss_vs/=n_agents
        else:
            loss_vs = self.loss_func(probs_vs, preferences, missclassified_vs, apply_on_misclassified_only=self.vs_apply_on_misclassified_only, logits=logits_value_system)
        if loss_vs > 1e+35:
            print(f"Large loss vs: {loss_vs}, probs_vs: {probs_vs}, preferences: {preferences}, missclassified_vs: {missclassified_vs}, logits_value_system: {logits_value_system}")
            raise ValueError("Loss vs is too large. Stopping training.")
                
        conc_penalty = 0.0
        if self.cluster_similarity_penalty > 0.0:
            conc_penalty = self.conciseness_penalty(grounding1=grounding1, grounding2=grounding2, cluster_fidxs=cluster_fidxs, 
                                                    value_system_network_per_cluster=weights_per_cluster, preferences=preferences, conc_penalty_reduction=self.conciseness_penalty_reduction)
            
            print(f"Conciseness penalty: {self.cluster_similarity_penalty, conc_penalty}, reduction: {self.conciseness_penalty_reduction}")
            loss_vs = loss_vs - self.cluster_similarity_penalty*conc_penalty
        #loss_vs*=0

        assert not loss_vs.isnan(), f"Loss is NaN: {loss_vs}"

        metrics['loss_vs'] = loss_vs
        # LOSS GROUNDING.
        # start_time = time.time()

        loss_gr = th.tensor(0.0, device=probs_vs.device,
                            dtype=probs_vs.dtype)


        missclassified_gr = np.array(missclassified_gr)
        if self.per_agent:
            for ag, fidxs_ag in fidxs_per_agent.items():
                probs_ag_v = probs_gr[fidxs_ag]
                preferences_with_grounding_ag = preferences_with_grounding[fidxs_ag]
                missclassified_gr_ag = missclassified_gr[:, fidxs_ag]
                logits_ag = logits_grounding[fidxs_ag]
                for vi in range(probs_gr.shape[1]):
                    if len(fidxs_ag) == 0:
                        raise ValueError(f"Agent {ag} has no fragments to compute loss on. Check if the agent has fragments in the dataset or if the fidxs_per_agent is correct.")

                    loss_vi_ag = self.loss_func(probs_ag_v[:, vi], preferences_with_grounding_ag[:, vi], misclassified_pairs=missclassified_gr_ag[vi], apply_on_misclassified_only=self.gr_apply_on_misclassified_only, logits=logits_ag[:, vi])
                    if vi not in metrics['loss_per_vi']:
                        metrics['loss_per_vi'][vi] = loss_vi_ag/n_agents
                    else:
                        metrics['loss_per_vi'][vi] += loss_vi_ag/n_agents
            for vi in range(probs_gr.shape[1]):
                loss_gr += metrics['loss_per_vi'][vi]
                    

        else:

            for vi in range(probs_gr.shape[1]):
                # TODO: This is the modified loss...
                """
                # This idea is CONFIDENT PENALTY: https://openreview.net/pdf?id=HyhbYrGYe ICLR 2017
                # TODO: Idea for future work:
                # from pairwise comparisons, minimize divergence from a prior that consists on the expected class probability, 
                # being it estimated online.
                # Under the assumption of convergence, we should find a single only possible function that is equivalent to the original
                # preferences (up to some operation, probably multiplication by a constant).
                
                """
                # loss_gr += th.mean(th.nn.functional.binary_cross_entropy(prgvi, preferences_with_grounding[:, vi], reduce=False) -th.multiply(prgvi, self.beta*th.log(prgvi)))
                # TODO: IMPORTANT: This method works because we are assuming SAME NUMBER OF EXAMPLES PER AGENT AND 1 CLUSTER.
                # TODO: IMPORTANT NEED TO DO AS IN LOSS VS FOR COHERENCE IN THE GENERAL CASE.
                nl = self.loss_func(probs_gr[:, vi], preferences_with_grounding[:, vi], misclassified_pairs=missclassified_gr[vi], apply_on_misclassified_only=self.gr_apply_on_misclassified_only, logits=logits_grounding[:, vi])
                metrics['loss_per_vi'][vi] = nl
                loss_gr = nl + loss_gr
                #print("LOSS GR", vi, nl, loss_gr, probs_gr[0:5])
                #print(list(reward_vector_module.parameters()))
                assert not loss_gr.isnan(), probs_gr[0:10]
                assert loss_gr.grad_fn is not None, f"Loss grounding gradient function is None. Loss grounding: {loss_gr}, probs_gr: {probs_gr[0:5]}, preferences_with_grounding: {preferences_with_grounding[0:5]}, logits_grounding: {logits_grounding[0:5]}"
                
        
        metrics['loss_gr'] = loss_gr
        # end_time = time.time()
        # print(f"Execution time for loss grounding: {end_time - start_time} seconds")
        return preference_comparisons.LossAndMetrics(
            loss=(loss_vs, loss_gr, metrics['loss_per_vi']),
            metrics=metrics,
        )
    
    def conciseness_penalty(self, grounding1: th.Tensor, grounding2: th.Tensor,
                            cluster_fidxs: List[List[int]],
                            value_system_network_per_cluster: List[LinearAlignmentLayer], 
                            preferences: th.Tensor,conc_penalty_reduction='min'):
        
        device = grounding1.device
        dtype = grounding1.dtype
        if len(value_system_network_per_cluster) <= 1:
            return th.tensor(0.0, device=device, dtype=dtype)
        
        agents_per_cluster = cluster_fidxs

        result = []
        for (c1, vs1), (c2, vs2) in itertools.combinations(list(enumerate(value_system_network_per_cluster)), 2):
            ac1 = agents_per_cluster[c1]
            ac2 = agents_per_cluster[c2]
            if len(ac1) == 0 or len(ac2) == 0:
                continue
            cfidxs1 = cluster_fidxs[c1]
            cfidxs2 = cluster_fidxs[c2]

            rews_vs1_f1 = vs1.forward(grounding1).squeeze(-1)
            rews_vs1_f2 = vs1.forward(grounding2).squeeze(-1)
            assert rews_vs1_f1.shape == rews_vs1_f2.shape == (len(grounding1),) 

            rews_vs2_f1 = vs2.forward(grounding1).squeeze(-1)
            rews_vs2_f2 = vs2.forward(grounding2).squeeze(-1)
            model1_in2 = probability_BT(rews_vs1_f1, rews_vs1_f2)
            model2_in1= probability_BT(rews_vs2_f1, rews_vs2_f2)
            assert model1_in2.shape == model2_in1.shape == (len(grounding1),)
            
            if OPTION3:
                jsd12 = jensen_shannon_pairwise_preferences(preferences[cfidxs1], model2_in1[cfidxs1])
                jsd21 = jensen_shannon_pairwise_preferences(preferences[cfidxs2], model1_in2[cfidxs2])

                result.append((jsd12 + jsd21)*1.0/(len(ac1) + len(ac2)))
            else:
                result.append(jensen_shannon_pairwise_preferences(model1_in2, model2_in1))
            # 1/(N1 + N2) * (sum(JSD(agentes1,2) + JSD(agentes2,1)))
        if len(result) == 0:
            return th.tensor(0.0, device=device, dtype=dtype)
        if conc_penalty_reduction == 'min':
            conc_penalty_reduction = th.min
        elif conc_penalty_reduction == 'mean':
            conc_penalty_reduction = th.mean
        elif conc_penalty_reduction == 'none':
            conc_penalty_reduction = lambda x, dim: x
        elif conc_penalty_reduction == 'sum':
            conc_penalty_reduction = th.sum
        else:
            raise ValueError(f"Unknown conc_penalty_reduction: {conc_penalty_reduction}. Must be 'min' or 'mean'.")

        conc_penalty = conc_penalty_reduction(th.tensor(result))
        return conc_penalty

    
    
class VSLOptimizer(th.optim.Optimizer):
    def __init__(self, params_gr, params_vs,n_values, lr=0.001, lr_grounding=None, lr_value_system=None,sub_optimizer_class=th.optim.Adam, **optimizer_kwargs):
        lr_grounding = lr if lr_grounding is None else lr_grounding
        lr_value_system = lr if lr_value_system is None else lr_value_system
        self.lr_grounding = lr_grounding
        self.lr_value_system = lr_value_system
        defaults = dict(lr_grounding=lr_grounding, lr_value_system=lr_value_system, lr_vt=lr_value_system)
        
        self.optimizer_kwargs = optimizer_kwargs
        self.n_values = n_values

        params_gr = OrderedSet(params_gr)
        params_vs = OrderedSet(params_vs)
        
        self.params_gr = params_gr
        self.params_vs = params_vs
        self.sub_optimizer_class = sub_optimizer_class
        self.optimx = sub_optimizer_class(params_gr, lr=lr_grounding, **self.optimizer_kwargs)
        if params_vs and len(params_vs) > 0:
            self.optimy = sub_optimizer_class(params_vs, lr=lr_value_system, **self.optimizer_kwargs)
        else:
            self.optimy = None
        
        super(VSLOptimizer, self).__init__([*params_gr, *params_vs], defaults)
       
    def get_state(self,copy=False):
        return {}
    def set_state(self, state):
        return
    
    @abstractmethod
    def zero_grad(self, set_to_none = True):
        super().zero_grad(set_to_none)
        self.optimx.zero_grad(set_to_none)
        if self.optimy is not None:
            self.optimy.zero_grad(set_to_none)
        return None
    
    @abstractmethod
    def step(self, closure=None):
        self.optimx.step()
        if self.optimy is not None:
            self.optimy.step()
        return None

class VSLCustomLoss(BaseVSLClusterRewardLoss):
    # TODO: Use efficient version when it is known the grounding does not depend on VS

    """def __init__(self, model_indifference_tolerance, gr_apply_on_misclassified_pairs_only=False, vs_apply_on_misclassified_pairs_only=False, confident_penalty=0, cluster_similarity_penalty=0.00,
                 label_smoothing=0.0, assume_number_of_examples_per_agent_is_equal=False, **kwargs):
        super().__init__(model_indifference_tolerance=model_indifference_tolerance, 
                         gr_apply_on_misclassified_pairs_only=gr_apply_on_misclassified_pairs_only, 
                         vs_apply_on_misclassified_pairs_only=vs_apply_on_misclassified_pairs_only, 
                         confident_penalty=confident_penalty, cluster_similarity_penalty=cluster_similarity_penalty,
                         label_smoothing=label_smoothing, **kwargs)
        
"""
    def __init__(self, model_indifference_tolerance, gr_apply_on_misclassified_pairs_only=False, vs_apply_on_misclassified_pairs_only=False, confident_penalty=0, label_smoothing=0, cluster_similarity_penalty=0, conciseness_penalty_reduction='min', missclassification_min_weighting=0.3, per_agent=True):
        super().__init__(model_indifference_tolerance=model_indifference_tolerance, 
                         gr_apply_on_misclassified_pairs_only=gr_apply_on_misclassified_pairs_only, 
                         vs_apply_on_misclassified_pairs_only=vs_apply_on_misclassified_pairs_only, 
                         confident_penalty=confident_penalty, 
                         label_smoothing=label_smoothing, 
                         cluster_similarity_penalty=cluster_similarity_penalty,
                         conciseness_penalty_reduction=conciseness_penalty_reduction, missclassification_min_weighting=missclassification_min_weighting, per_agent=per_agent)
    @abstractmethod
    def gradients(self, scalar_loss: th.Tensor, renormalization: float) -> None:
        pass

    def set_parameters(self, params_gr, params_vs, optim_state={}):
        self.params_gr = params_gr
        self.params_vs = params_vs
    
    def parameters(self, recurse = True):
        return self.params_gr + self.params_vs

    def forward(self, fragments1, fragments2, reward_vector_module, weights_per_cluster, cluster_fidxs, preferences, preferences_with_grounding, fidxs_per_agent):
        lossMetrics = super().forward(fragments1, fragments2, reward_vector_module, weights_per_cluster, cluster_fidxs, preferences, preferences_with_grounding, fidxs_per_agent)
        self.gr_loss_per_vi = lossMetrics.loss[2]
        self.gr_loss = lossMetrics.loss[1]
        self.vs_loss = lossMetrics.loss[0]
        return lossMetrics
    
    def _____forward_old(self, preferences, preferences_with_grounding, preference_model, reward_model_per_agent_id, fragment_idxs_per_aid, 
                fragment_pairs_per_agent_id = None, preferences_per_agent_id = None, 
                preferences_per_agent_id_with_grounding = None, value_system_network_per_cluster = None, 
                grounding_per_value_per_cluster = None,agent_to_vs_cluster_assignments=None) -> LossAndMetrics:
        lossMetrics = super().forward(preferences, preferences_with_grounding, preference_model, reward_model_per_agent_id, fragment_idxs_per_aid, fragment_pairs_per_agent_id, preferences_per_agent_id, preferences_per_agent_id_with_grounding, 
                                      value_system_network_per_cluster, grounding_per_value_per_cluster,agent_to_vs_cluster_assignments)
        self.gr_loss_per_vi = lossMetrics.loss[2]
        self.gr_loss = lossMetrics.loss[1]
        self.vs_loss = lossMetrics.loss[0]
        return lossMetrics
    


        #th.nn.utils.clip_grad_norm_(self.params_gr, self.max_grad_norm_gr)
        #th.nn.utils.clip_grad_norm_(self.params_vs, self.max_grad_norm_vs)
class ConstrainedOptimizer(VSLOptimizer):
    def __init__(self, params_gr, params_vs,n_values, max_grad_norm, lr=0.001, lr_grounding=None, 
                 lr_value_system=None, lr_lambda=None, initial_lambda=1.0,lambda_decay=1e-9,
                 sub_optimizer_class=th.optim.Adam, **optimizer_kwargs):
        super(ConstrainedOptimizer, self).__init__(params_gr=params_gr, params_vs=params_vs, n_values=n_values, lr=lr, lr_grounding=lr_grounding, lr_value_system=lr_value_system, sub_optimizer_class=sub_optimizer_class, **optimizer_kwargs)
        self.lr_lambda = lr_lambda if lr_lambda is not None else lr_value_system / 10.0
        self.initial_lambda = initial_lambda
        self.lambda_decay = lambda_decay
        self.max_grad_norm = max_grad_norm
        self.set_state({'time': 0, 'lambdas': [th.tensor(initial_lambda, requires_grad=True) for _ in range(n_values)]})
        

    def get_state(self, copy=False):
        return {'time': self.time, 'lambdas': [l.clone().detach() if copy else l for l in self.lambdas]}

    def set_state(self, state):
        if state is not None:
            self.time = state['time']
            self.lambdas = state['lambdas']
        else:
            self.time = 0
            self.lambdas = [th.tensor(self.initial_lambda, requires_grad=True)  for vi in range(self.n_values)]
        print("LAGRANGE MULTIPLIERS AT SET STATE:", self.lambdas)
        if self.lr_lambda > 0:
            self.optim_lambdas = th.optim.Adam(self.lambdas, lr=self.lr_lambda, betas=(0.5,0.9))

    """@override
    def set_parameters(self, params_gr, params_vs, optim_state={}):
        self.params_gr = params_gr
        self.params_vs = params_vs
        self.optimx = self.sub_optimizer_class(params_gr, lr=self.lr_grounding, **self.optimizer_kwargs)
        self.optimy = self.sub_optimizer_class(params_vs, lr=self.lr_value_system, **self.optimizer_kwargs)"""
    def zero_grad(self, set_to_none = True):
        super().zero_grad(set_to_none)
        if self.lr_lambda > 0:
            self.optim_lambdas.zero_grad(set_to_none)

    def step(self, closure=None):
        self.time += 1
        th.nn.utils.clip_grad_norm_(self.params_gr, self.max_grad_norm)
        th.nn.utils.clip_grad_norm_(self.params_vs, self.max_grad_norm)
        self.optimx.step()
        if self.optimy is not None:
            self.optimy.step()
        if self.lr_lambda > 0:
            self.optim_lambdas.step()

        if self.lambda_decay > 0:
            with th.no_grad():
                for vi in range(len(self.lambdas)):
                    if self.lambdas[vi] > self.initial_lambda:
                        decay = (self.lambdas[vi].detach()*self.lambda_decay)
                        self.lambdas[vi].data = th.clamp(self.lambdas[vi].data - decay, min=self.initial_lambda)
            #print("LAMBDA a", self.lambdas) 
        return None


class ConstrainedLoss(VSLCustomLoss):
    def __init__(self, model_indifference_tolerance, gr_apply_on_misclassified_pairs_only=False, 
                         vs_apply_on_misclassified_pairs_only=False, 
                         label_smoothing=0.0, lambda_decay=1e-9, confident_penalty=0, 
                         cluster_similarity_penalty=0.00,conciseness_penalty_reduction=0.0, missclassification_min_weighting=0.3, per_agent=True):
        super().__init__(model_indifference_tolerance=model_indifference_tolerance,
                        gr_apply_on_misclassified_pairs_only=gr_apply_on_misclassified_pairs_only, 
                         vs_apply_on_misclassified_pairs_only=vs_apply_on_misclassified_pairs_only, 
                         confident_penalty=confident_penalty, 
                         cluster_similarity_penalty=cluster_similarity_penalty, 
                         label_smoothing=label_smoothing,conciseness_penalty_reduction=conciseness_penalty_reduction, 
                         per_agent=per_agent,missclassification_min_weighting=missclassification_min_weighting)

        self.lagrange_multipliers = None
        self.best_gr_losses = None
        self.best_accuracies = None

        self.lambda_decay = lambda_decay
    def set_parameters(self, params_gr, params_vs, optim_state):
        super().set_parameters(params_gr, params_vs)
        self.lagrange_multipliers = optim_state['lambdas']
        for lm in self.lagrange_multipliers:
            lm.requires_grad = True
    
    def gradients(self, scalar_loss: th.Tensor, renormalization: float) -> None:
        
        if self.best_accuracies is None:
                #self.best_gr_losses = [float('inf')]*len(self.gr_loss_per_vi)
                self.best_accuracies = [0.0]*len(self.last_accuracy_gr)
        for vi in range(len(self.lagrange_multipliers)):
            self.lagrange_multipliers[vi].requires_grad_(False)
                    #self.best_gr_losses[vi] = min(gr_loss_vi, self.best_gr_losses[vi])
        #print("REAL_THINGS GRADIENTED", self.vs_loss, self.gr_loss_per_vi, self.lagrange_multipliers, self.last_accuracy_gr, self.best_accuracies)
        #input()
        lmbig0 = sum(self.lagrange_multipliers) > 0
        if lmbig0:
            real_loss = self.vs_loss*renormalization + sum(self.lagrange_multipliers[vi] * self.gr_loss_per_vi[vi] * renormalization for vi in range(len(self.gr_loss_per_vi)))
        else:
            real_loss = self.vs_loss*renormalization 
        #l2_norm = sum(p.pow(2).sum() for p in self.params_gr)
        #real_loss = real_loss + 0.001 * l2_norm # L2 regularization on the grounding parameters
        # TODO: this is not the best way to do it, but it is the only one that works with the current implementation.
        b = real_loss.backward()
        with th.no_grad():
            for vi in range(len(self.lagrange_multipliers)):
                self.lagrange_multipliers[vi].requires_grad_(True)
            print("LAMBDA b", self.lagrange_multipliers,  self.last_accuracy_gr, self.best_accuracies)  

            farthest_l = 0
            max_diff = float('-inf')
            for vi,l in enumerate(self.lagrange_multipliers):
                l.grad = None
                diff = abs(self.last_accuracy_gr[vi] - self.best_accuracies[vi])
                if self.last_accuracy_gr[vi] >= self.best_accuracies[vi]:
                    self.best_accuracies[vi] =  0.05*float(self.last_accuracy_gr[vi]) + 0.95*self.best_accuracies[vi]
                if diff > max_diff:
                    max_diff = diff
                    farthest_l = vi
            self.lagrange_multipliers[farthest_l].grad = -th.clamp(th.abs(self.gr_loss_per_vi[farthest_l] * renormalization), min=0.0).detach()
                    
            

        return b
class SobaOptimizer(VSLOptimizer):
    def __init__(self, params_gr, params_vs, n_values, lr=0.001, lr_grounding=None, lr_value_system=None, use_lr_decay = True, max_grad_norm_gr=1000.0, max_grad_norm_vs=1000.0, **optimizer_kwargs):

        
        super(SobaOptimizer, self).__init__(params_gr=params_gr,params_vs=params_vs,lr=lr, n_values=n_values, lr_grounding=lr_grounding, lr_value_system=lr_value_system,
                                         sub_optimizer_class=th.optim.Adam, **optimizer_kwargs)
        
        self.vt = [th.zeros_like(p) for p in params_gr]
        self.use_lr_decay = use_lr_decay
        self.lr_value_system = lr_value_system
        self.lr_grounding = lr_grounding
        self.max_grad_norm_gr = max_grad_norm_gr
        self.max_grad_norm_vs = max_grad_norm_vs
        self.time = 0
        self.set_state({'time': self.time, 'vt': self.vt})

    def zero_grad(self, set_to_none = True):

        self.optimx.zero_grad(set_to_none)
        if self.optimy is not None:
            self.optimy.zero_grad(set_to_none)
        self.optimv.zero_grad(set_to_none)

    def get_state(self,copy=False):
        
        return {'time': self.time, 'vt': self.vt}
    
    def set_state(self, state):
        if state is not None:
            self.time = state['time']
            self.vt = state['vt']
            self.optimv = th.optim.SGD(state['vt'], lr=self.lr_value_system)

            if self.use_lr_decay:
                self.optimx_scheduler = th.optim.lr_scheduler.LambdaLR(lr_lambda=lambda epoch: 1/np.sqrt(self.time+1), optimizer=self.optimx)
                self.optimy_scheduler = th.optim.lr_scheduler.LambdaLR(lr_lambda=lambda epoch: 1/np.sqrt(self.time+1), optimizer=self.optimy)
                self.optimv_scheduler = th.optim.lr_scheduler.LambdaLR(lr_lambda=lambda epoch: 1/np.sqrt(self.time+1), optimizer=self.optimv)
        if state is None:
            self.set_state({'time': 0, 'vt': [th.zeros_like(p) for p in self.params_gr]}) 
    def step(self, closure=None):
        self.optimx.step()
        self.optimy.step()
        
        self.optimv.step()
        self.time+=1
        if self.use_lr_decay:
            self.optimx_scheduler.step()
            self.optimy_scheduler.step()
            self.optimv_scheduler.step() 
        return None



class SobaLoss(VSLCustomLoss):
    def set_parameters(self, params_gr, params_vs, vt, **kwargs):
        super().set_parameters(params_gr, params_vs)
        self.vt = vt

    
    def parameters(self, recurse = True):
        return self.params_gr + self.params_vs + self.vt
    
    
    
    def gradients(self, scalar_loss: th.Tensor, renormalization: float, fast=False) -> None:
        """Dtheta, Dv, Dlambda = gradients_soba(
            self.wx, self.wy, self.vt, self.gr_loss*renormalization, self.vs_loss*renormalization
        )"""
        if fast:
            # This is when the derivative of gr_loss does not depend on self.params_vs. This makes convergence an
            Dtheta, Dv, Dlambda = gradients_soba_eff(
                self.params_gr, self.params_vs, self.vt, self.gr_loss*renormalization, self.vs_loss*renormalization
            )
        else:
            Dtheta, Dv, Dlambda = gradients_soba(
                self.params_gr, self.params_vs, self.vt, self.gr_loss*renormalization, self.vs_loss*renormalization
            )
            assert len(self.vt) == len(Dv)
            
            with th.no_grad():
                for p,pref in zip(self.vt, Dv):
                    p.grad = pref
        assert len(Dtheta) == len(self.params_gr)
        assert len(Dlambda) == len(self.params_vs)
        with th.no_grad():
            for p,pref in zip(self.params_gr, Dtheta):
                p.grad = pref
        
            for p,pref in zip(self.params_vs, Dlambda):
                p.grad = pref
            
        th.nn.utils.clip_grad_norm_(self.params_gr, self.max_grad_norm_gr)
        th.nn.utils.clip_grad_norm_(self.params_vs, self.max_grad_norm_vs)


def probs_to_label(probs, indifference_tol = 0.05):
    # probs: 1D tensor of shape (N,)
    labels = th.where(
        probs < 0.5 - indifference_tol,
        th.zeros_like(probs),
        th.where(
            probs > 0.5 + indifference_tol,
            th.ones_like(probs),
            th.full_like(probs, 0.5),
        )
    )
    if __debug__:
        if probs[0] > 0.5 + indifference_tol:
            assert labels[0] == 1, f"Expected label 0, but got {labels[0]} for probability {probs[0]}"

        if probs[0] < 0.5 - indifference_tol:
            assert labels[0] == 0, f"Expected label 1, but got {labels[0]} for probability {probs[0]}"
        if abs(probs[0] - 0.5) <= indifference_tol:
            assert labels[0] == 0.5, f"Expected label 0.5, but got {labels[0]} for probability {probs[0]}"

    return  labels