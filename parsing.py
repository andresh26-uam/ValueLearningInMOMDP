

from copy import deepcopy
import torch


from sb3_contrib.ppo_mask.policies import MlpPolicy as MASKEDMlpPolicy
from stable_baselines3.ppo import MlpPolicy
import argparse
from typing import Dict, Tuple, Union
import numpy as np
import torch

from gymnasium.wrappers import NormalizeObservation
from baraacuda.utils.normalizers import ExpWeighted_MeanStd_Normalizer
from defines import transform_weights_to_tuple
from envs.firefighters_env_mo import FeatureSelectionFFEnv
from envs.multivalued_car_env import MVFS, MultiValuedCarEnv
from src.algorithms.preference_based_vsl_lib import BaseVSLClusterRewardLoss, ConstrainedLoss, ConstrainedOptimizer, PrefLossClasses, SobaLoss, SobaOptimizer, VSLOptimizer
from src.algorithms.preference_based_vsl_simple import EnvelopeClusteredPBMORL
from src.dataset_processing.utils import DEFAULT_SEED
from src.feature_extractors import BaseRewardFeatureExtractor, ObservationMatrixRewardFeatureExtractor, ObservationWrapperFeatureExtractor
from src.policies.morl_custom_reward import ROLLOUT_BUFFER_CLASSES, EnvelopeCustomReward, EnvelopePBMORL, RolloutBufferCustomReward
from src.policies.vsl_policies import LearnerValueSystemLearningPolicy, MaskedPolicySimple, VAlignedDictSpaceActionPolicy, ValueSystemLearningPolicyCustomLearner
from baraacuda.utils.wrappers import LL_Terminations, RewardAccumulator, AccumulatedRewardsStateAugmentor, TransformVectorReward, OneHotObservationWrapper, NormalizeRewardWrapper

from mushroom_rl.utils.parameters import LinearParameter, Parameter
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym


from morl_baselines.common.experiments import (
    ALGOS, ENVS_WITH_KNOWN_PARETO_FRONT
)

from mo_gymnasium.wrappers.wrappers import MORecordEpisodeStatistics, MONormalizeReward, MOClipReward

from mo_gymnasium.wrappers.vector import MOSyncVectorEnv
from stable_baselines3.common.vec_env import DummyVecEnv

from src.reward_nets.vsl_reward_functions import parse_layer_name
from src.utils import sample_example_profiles
#from use_cases.roadworld_env_use_case.network_env import FeaturePreprocess, FeatureSelection

WRAPPERS = {
    "NormalizeObservation": NormalizeObservation,
    "FlattenObservation": FlattenObservation,
    "TransformVectorReward": TransformVectorReward,
    "RewardAccumulator": RewardAccumulator,
    "AccumulatedRewardsStateAugmentor": AccumulatedRewardsStateAugmentor,
    "LL_Terminations": LL_Terminations,
    "OneHotObservationWrapper": OneHotObservationWrapper,
    "NormalizeRewardWrapper": NormalizeRewardWrapper,
    "FilterObservation": gym.wrappers.FilterObservation,
    "MORecordEpisodeStatistics": MORecordEpisodeStatistics,
    "MONormalizeReward": MONormalizeReward,
    "MOClipReward": MOClipReward
}
NORMALIZERS = {
    "ExpWeighted_MeanStd_Normalizer": ExpWeighted_MeanStd_Normalizer
}
OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "RMSprop": torch.optim.RMSprop,
    "SGD": torch.optim.SGD,
}
ACTIVATIONS = {
    "LeakyReLU": torch.nn.LeakyReLU,
    "ReLU": torch.nn.ReLU,
    "Tanh": torch.nn.Tanh,
    "Sigmoid": torch.nn.Sigmoid,
    "ELU": torch.nn.ELU,
}


class DummyVecEnv2(DummyVecEnv, gym.Env):
    def __init__(self, env_fns):
        super().__init__(env_fns)

    @property
    def spec(self):
        return self.envs[0].spec

    def step(self, actions):
        n, r, d, i = super().step(actions)
        return n, r, d, d, i

    def reset(self, seed=None, options=None):
        super().seed(seed)
        super().set_options(options)
        for env_idx in range(self.num_envs):
            maybe_options = {
                "options": self._options[env_idx]} if self._options[env_idx] else {}
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(
                seed=self._seeds[env_idx], **maybe_options)
            self._save_obs(env_idx, obs)
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf(), self.reset_infos

    def has_wrapper_attr(self, name):
        return self.envs[0].has_wrapper_attr(name)

    def get_wrapper_attr(self, name):
        return self.envs[0].get_wrapper_attr(name)

    def set_wrapper_attr(self, name, value, *, force=True):
        for e in self.envs:
            e.set_wrapper_attr(name, value, force=force)


VECTORIZERS = {
    "MOSyncVectorEnv": MOSyncVectorEnv,
    "DummyVecEnv": DummyVecEnv2
}


def parse_epsilon_params_from_json(eps_conf: dict) -> dict:
    """
    Parses epsilon parameters from a JSON file.
    The JSON should contain keys: epsilon_threshold, epsilon_init, epsilon_decay_steps, epsilon_train, epsilon_eval.
    Example:
    epsilon_config = {
        "epsilon_threshold": 0.2,
        "epsilon_init": 0.1,
        "epsilon_decay_steps": 100000,
        "epsilon_train": 1.0,
        "epsilon_eval": 0.0
    }
    """
    epsilon_threshold = eps_conf.get("epsilon_threshold", 0.2)
    epsilon_init = Parameter(value=eps_conf.get("epsilon_init", 0.1))
    epsilon_decay_steps = eps_conf.get("epsilon_decay_steps", 100000)
    epsilon_train = LinearParameter(
        value=eps_conf.get("epsilon_train", 1.0),
        threshold_value=epsilon_threshold,
        n=epsilon_decay_steps
    )
    epsilon_eval = Parameter(value=eps_conf.get("epsilon_eval", 0.0))
    return dict(
        epsilon_threshold=epsilon_threshold,
        epsilon_init=epsilon_init,
        epsilon_decay_steps=epsilon_decay_steps,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval
    )


def parse_vectorizer_from_json(vectorizer_conf: dict):
    if vectorizer_conf is None:
        return None, {}
    return VECTORIZERS.get(vectorizer_conf.get('type', "no"), None), vectorizer_conf.get('kwargs', {})


def parse_wrappers_from_json(wrapper_json_list, discount_official=1.0):
    """
    Parses a list of wrapper specifications (as dicts) and returns a list of tuples (wrapper_class, kwargs).
    Supports referencing a normalizer instance if needed.
    """
    transform_reward_idxs = None
    wrappers = []
    normalizer = None
    num_envs = 0
    for w in wrapper_json_list:
        wtype = w["type"]
        wkwargs = w.get("kwargs", {})
        if "gamma" in wkwargs and wkwargs["gamma"] is None:
            wkwargs["gamma"] = discount_official

        # Special handling for normalizer
        if wtype == "TransformVectorReward" and "transform_reward_idxs" in wkwargs:
            transform_reward_idxs = wkwargs["transform_reward_idxs"]
            if isinstance(transform_reward_idxs, list):
                transform_reward_idxs = np.array(
                    transform_reward_idxs, dtype=np.int32)
            wkwargs.pop("transform_reward_idxs", None)
            if "transformer" not in wkwargs:
                wkwargs["transformer"] = lambda vec: vec[transform_reward_idxs]
            else:
                wkwargs["transformer"] = eval(
                    wkwargs["transformer"]) if "transformer" in wkwargs else lambda x: x

            wkwargs["output_vec_len"] = len(transform_reward_idxs)
        elif wtype == "AccumulatedRewardsStateAugmentor" and "normalizer" in wkwargs:
            norm_spec = wkwargs["normalizer"]
            assert transform_reward_idxs is not None, "transform_reward_idxs must be provided for AccumulatedRewardsStateAugmentor. Use TransformVectorReward wrapper before this one."
            if isinstance(norm_spec, dict):
                norm_cls = NORMALIZERS[norm_spec["type"]]
                norm_kwargs = norm_spec.get("kwargs", {})
                norm_kwargs['input_vec_len'] = len(transform_reward_idxs)
                wkwargs["normalizer"] = norm_cls(**norm_kwargs)
                normalizer = wkwargs["normalizer"]
                normalizer.track_stats = False

        elif wtype not in WRAPPERS:
            raise ValueError(f"Unknown wrapper type: {wtype}")
        wrappers.append((WRAPPERS[wtype], wkwargs))
    return wrappers, normalizer

MORL_ALGOS = deepcopy(ALGOS)
MORL_ALGOS.update({
    "EnvelopeCustomReward": EnvelopeCustomReward,
    "EnvelopePBMORL": EnvelopePBMORL,
    "EnvelopeClusteredPBMORL": EnvelopeClusteredPBMORL
})
def parse_policy_approximator(ref_class, env_name: str, society_data: Dict, environment_data: Dict, ref_policy_kwargs: Dict, train_environment: gym.Env, train_environment_mush=None, parser_args=None, learner_or_expert: str = 'expert'):

    is_single_objective = False
    normalizer = None
    ret_class = ref_class
    alg_config = environment_data['algorithm_config'][parser_args.algorithm]
    ref_policy_kwargs['wandb_entity'] = str(parser_args.wandb_entity) if hasattr(
        parser_args, 'wandb_entity') and parser_args.wandb_entity is not None else None

    if ref_class not in MORL_ALGOS:
        is_single_objective = True

    if 'train_kwargs' in ref_policy_kwargs:
        if "known_pareto_front" in ref_policy_kwargs['train_kwargs'] and ref_policy_kwargs['train_kwargs']['known_pareto_front'] is not None:
            ref_policy_kwargs['train_kwargs']['known_pareto_front'] = eval(
                ref_policy_kwargs['train_kwargs']['known_pareto_front'])

        if env_name in ENVS_WITH_KNOWN_PARETO_FRONT:
            ref_policy_kwargs['train_kwargs']['known_pareto_front'] = train_environment.unwrapped.pareto_front(
                gamma=alg_config['discount_factor'])

    seed = parser_args.seed if parser_args is not None else DEFAULT_SEED
    ref_policy_kwargs.update(parse_learner_policy_kwargs(
        ref_class, ref_policy_kwargs, ref_env=train_environment, gamma=alg_config['discount_factor'], seed=seed))
    ret_class = parse_learner_policy_class(ref_class)

    train_kwargs = ref_policy_kwargs.get('train_kwargs', {})
    n_epochs = train_kwargs.get('n_epochs', 30)
    n_train_steps_per_epoch = train_kwargs.get(
        'n_train_steps_per_epoch', train_kwargs.get('total_timesteps', 6000))

    n_train_steps_per_fit = train_kwargs.get('n_train_steps_per_fit', 1)
    n_eval_episodes_per_epoch = train_kwargs.get(
        'n_eval_episodes_per_epoch', 100)
    epochs_per_save = train_kwargs.get('epochs_per_save', min(10, n_epochs))
    seeds = list(range(n_eval_episodes_per_epoch))
    n_eval_episodes_per_epoch = None
    initial_states = train_kwargs.get('initial_states', None)
    get_action_info = train_kwargs.get('get_action_info', False)
    train_kwargs['n_epochs'] = n_epochs
    train_kwargs['n_train_steps_per_epoch'] = n_train_steps_per_epoch
    train_kwargs['n_train_steps_per_fit'] = n_train_steps_per_fit
    train_kwargs['n_eval_episodes_per_epoch'] = n_eval_episodes_per_epoch
    train_kwargs['epochs_per_save'] = epochs_per_save
    train_kwargs['seeds'] = seeds
    train_kwargs['initial_states'] = initial_states
    train_kwargs['get_action_info'] = get_action_info

    train_kwargs['epsilon_config'] = ref_policy_kwargs.get('epsilon_config', {
        "epsilon_threshold": 0.0,
        "epsilon_init": 0.0,
        "epsilon_decay_steps": 1000,
        "epsilon_train": 0.0,
        "epsilon_eval": 0.0
    })

    ref_policy_kwargs['train_kwargs'] = train_kwargs

    return ret_class, ref_policy_kwargs, is_single_objective


def parse_learner_policy_class(learner_policy_class):
    if learner_policy_class == 'CustomPolicy':
        return ValueSystemLearningPolicyCustomLearner
    elif learner_policy_class == 'VAlignedDictSpaceActionPolicy':
        return VAlignedDictSpaceActionPolicy
    elif learner_policy_class == 'LearnerValueSystemLearningPolicy' or learner_policy_class == 'ppo_learner':
        return LearnerValueSystemLearningPolicy
    elif learner_policy_class in MORL_ALGOS:
        print("Found MORL algorithm:", learner_policy_class)

        return MORL_ALGOS[learner_policy_class]
    else:
        raise NotImplementedError(
            f"Unknown learner policy class {learner_policy_class}")


def parse_learner_kwargs(learner_kwargs):
    
    if learner_kwargs.get('rollout_buffer_class', None) is not None:
        learner_kwargs['rollout_buffer_class'] = ROLLOUT_BUFFER_CLASSES[learner_kwargs['rollout_buffer_class']]
    return learner_kwargs   
def parse_learner_policy_kwargs(learner_policy_class, learner_policy_kwargs, ref_env: gym.Env, gamma: float, seed):

    if learner_policy_class in MORL_ALGOS.keys():
        learner_policy_kwargs['seed'] = seed
        learner_policy_kwargs['log'] = True
        learner_policy_kwargs['gamma'] = gamma

        if learner_policy_class == 'pgmorl':
            learner_policy_kwargs['origin'] = np.array(
                learner_policy_kwargs['train_kwargs']['ref_point'], dtype=np.float32)
            learner_policy_kwargs['env_id'] = ref_env.spec.id
        else:
            learner_policy_kwargs['train_kwargs']['ref_point'] = np.asarray(
                learner_policy_kwargs['train_kwargs']['ref_point'], dtype=np.float32)
        return learner_policy_kwargs

    elif learner_policy_class == 'VAlignedDictSpaceActionPolicy':
        return {'policy_per_va_dict': {}, 'state_encoder': None, 'expose_state': learner_policy_kwargs.get('expose_state', True), 'use_checkpoints': learner_policy_kwargs.get('use_checkpoints', False)}
    elif learner_policy_class == 'LearnerValueSystemLearningPolicy' or learner_policy_class == 'ppo_learner':
        return {'learner_class': parse_learner_class(learner_policy_kwargs['learner_policy_class']),
                'learner_kwargs': parse_learner_kwargs(learner_policy_kwargs['learner_policy_kwargs']),
                'observation_space': ref_env.observation_space, 'action_space': ref_env.action_space,
                'policy_class': parse_policy_class(learner_policy_kwargs.get('policy_class', 'MlpPolicy')),
                'policy_kwargs': learner_policy_kwargs.get('policy_kwargs', {}),
                'stochastic_eval': learner_policy_kwargs.get('stochastic_eval', False)
                }
    elif learner_policy_class == 'CustomPolicy':
        if 'MultiValued' in ref_env.spec.id:
            learner = lambda environment, alignment_function, grounding_function, reward, discount, stochastic, **kwargs: MultiValuedCarEnv.compute_policy(environment, reward=grounding_function,
                                                                                                    discount=discount,
                                                                                                    weights=alignment_function,**kwargs)
            eap_agg_ = dict(policy_per_va_dict={},learner_method=learner, **learner_policy_kwargs.get('learner_policy_kwargs', {}))
            eap_class = ValueSystemLearningPolicyCustomLearner
            return eap_agg_
        else:
            raise NotImplementedError(
                f"Custom policy class {learner_policy_kwargs['learner_policy_class']} not implemented for environment {ref_env.spec.id}")
    elif learner_policy_class == 'EnvelopeCustomReward' or learner_policy_class == 'EnvelopePBMORL' or learner_policy_class == 'EnvelopeClusteredPBMORL':
        return learner_policy_kwargs # TODO maybe more?
    else:
        raise NotImplementedError(
            f"Unknown learner class {learner_policy_class} for environment {ref_env.spec.id}")


def parse_device(device: str):
    if isinstance(device, torch.DeviceObjType):
        return device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if (
        device == "auto" or device == "cuda") else 'cpu'
    return device


def parse_learner_class(learner_class):
    if learner_class == 'PPO':
        from stable_baselines3 import ppo
        return ppo.PPO
    elif learner_class == 'MaskablePPO':
        from sb3_contrib import ppo_mask
        return ppo_mask.MaskablePPO
    else:
        raise ValueError(f"Unknown learner class: {learner_class}")


def parse_policy_class(policy_class):
    if policy_class == "MASKEDMlpPolicy":
        return MASKEDMlpPolicy
    elif policy_class == "MlpPolicy":
        return MlpPolicy
    elif policy_class == "MaskedPolicySimple":
        return MaskedPolicySimple
    else:
        return None


def parse_dtype_numpy(choice):
    ndtype = np.float32
    if choice == 'float16':
        ndtype = np.float16
    if choice == 'float32':
        ndtype = np.float32
    if choice == 'float64':
        ndtype = np.float64
    return ndtype


def parse_dtype_torch(choice):
    ndtype = torch.float32
    if choice == 'float16':
        ndtype = torch.float16
    if choice == 'float32':
        ndtype = torch.float32
    if choice == 'float64':
        ndtype = torch.float64
    return ndtype



def parse_feature_extractors(environment, env_name,  environment_data, dtype=torch.float32, device: Union[str, torch.device] = 'auto'):
    # Dummy implementation, replace with actual logic
    reward_net_features_extractor_kwargs = dict(observation_space=environment.observation_space,action_space=environment.action_space,device=device, dtype=dtype)
    EXTRACTORS = {
        "BaseRewardFeatureExtractor": BaseRewardFeatureExtractor,
        "ObservationMatrixRewardFeatureExtractor": ObservationMatrixRewardFeatureExtractor,
        "ObservationWrapperFeatureExtractor": ObservationWrapperFeatureExtractor,
    }
    if environment_data['reward_feature_extractor'] not in EXTRACTORS:
        raise ValueError(
            f"Unknown reward feature extractor {environment_data['reward_feature_extractor']}")
    
    reward_net_features_extractor_class = EXTRACTORS[environment_data['reward_feature_extractor']]
    
    if environment_data['reward_feature_extractor'] == "ObservationMatrixRewardFeatureExtractor":
        reward_net_features_extractor_kwargs.update(dict(
            observation_matrix=environment.get_wrapper_attr('observation_matrix'),
        ))
    elif environment_data['reward_feature_extractor'] == "OneHotRewardFeatureExtractor":
        reward_net_features_extractor_kwargs.update(dict(
            n_categories=environment.action_space.n))
    elif environment_data['reward_feature_extractor'] == "ObservationWrapperFeatureExtractor":
        if env_name == 'dst':
            reward_net_features_extractor_kwargs.update(dict(
                method=lambda obs: environment.get_wrapper_attr('get_reward')(obs),
            ))
            """reward_net_features_extractor_kwargs.update(dict(
                method=lambda obs: environment.get_wrapper_attr('get_int_obs')(obs),
            ))"""
    return reward_net_features_extractor_class, reward_net_features_extractor_kwargs

def parse_optimizer_data(environment_data, alg_config):
    opt_kwargs = environment_data['default_optimizer_kwargs']
    opt_kwargs = opt_kwargs if alg_config['optimizer_kwargs'] == "default" else alg_config['optimizer_kwargs']

    opt_class = environment_data['default_optimizer_class']
    opt_class = opt_class if alg_config['optimizer_class'] == "default" else alg_config['optimizer_class']

    if opt_class == 'VSLOptimizer':
        opt_class = VSLOptimizer
    elif opt_class == 'Soba':
        opt_class = SobaOptimizer
    elif opt_class == 'lagrange':
        opt_class = ConstrainedOptimizer
    else:
        raise ValueError(f"Unknown optimizer class {opt_class}")
    return opt_kwargs, opt_class


from mushroom_rl.environments import MO_Gymnasium as MO_GymMushroom

def make_env(env_name, horizon, gamma, wrappers, seed, extra_kwargs, vectorizer=None, vectorizer_kwargs=None):

    def thunk():
        wrappers_real = []
        for w in wrappers:
            wrappers_real.append(w)
        wrappers_real.append((MORecordEpisodeStatistics, {'gamma': gamma}))
        environment_mush: MO_GymMushroom = MO_GymMushroom(name=env_name,
                                                          horizon=horizon,
                                                          gamma=gamma,
                                                          wrappers=wrappers,
                                                          seed_env=seed,
                                                          **extra_kwargs)

        environment_mush.env.action_space.seed(seed)
        environment_mush.env.observation_space.seed(seed)

        return environment_mush.env, environment_mush

    def vec_env():
        if vectorizer is not None:
            num_envs = vectorizer_kwargs.pop('num_envs', 1)
            env = vectorizer([lambda: thunk()[0]] *
                             num_envs, **vectorizer_kwargs)
            environment_mush = thunk()[1]
            environment_mush.env = env
        else:
            env, environment_mush = thunk()
        return env, environment_mush

    return vec_env


def parse_expert_policy_parameters(parser_args, environment_data: Dict, society_data: Dict, train_environment: gym.Env, train_environment_mush, expert_policy_kwargs: Dict, expert_policy_class: str):
    epclass, epkwargs, is_single_objective = parse_policy_approximator(
        ref_class=expert_policy_class,
        learner_or_expert='expert',
        env_name=environment_data['name'],
        society_data=society_data, environment_data=environment_data,
        train_environment_mush=train_environment_mush,
        ref_policy_kwargs=expert_policy_kwargs, train_environment=train_environment, parser_args=parser_args)

    epkwargs_no_train_kwargs = deepcopy(epkwargs)
    train_kwargs = epkwargs_no_train_kwargs.pop('train_kwargs', None)

    if hasattr(parser_args, 'retrain'):
        if 'reset_num_timesteps' in train_kwargs:
            train_kwargs['reset_num_timesteps'] = (
                not parser_args.refine) or parser_args.retrain
        if 'reset_learning_starts' in train_kwargs:
            train_kwargs['reset_learning_starts'] = parser_args.retrain
    return epclass, epkwargs_no_train_kwargs, train_kwargs, is_single_objective

def parse_learning_policy_parameters(parser_args, environment_data: Dict, society_data: Dict, train_environment: gym.Env, train_environment_mush, learning_policy_kwargs: Dict, learning_policy_class: str):
    epclass, epkwargs, is_single_objective = parse_policy_approximator(
        ref_class=learning_policy_class,
        learner_or_expert='learner',
        env_name=environment_data['name'],
        society_data=society_data, environment_data=environment_data,
        train_environment_mush=train_environment_mush,
        ref_policy_kwargs=learning_policy_kwargs, train_environment=train_environment, parser_args=parser_args)

    epkwargs_no_train_kwargs = deepcopy(epkwargs)
    train_kwargs = epkwargs_no_train_kwargs.pop('train_kwargs', None)
    return epclass, epkwargs_no_train_kwargs, train_kwargs, is_single_objective

def parse_society_data(parser_args, society_config_env):
    society_data = society_config_env[parser_args.society_name]

    if society_data['agents'][0] == 'sample':
        agent_profiles = sample_example_profiles(
            society_data['agents'][1], n_values=society_data['n_values'])

        society_data['agents'] = []
        for i, profile in enumerate(agent_profiles):
            society_data['agents'].append({
                'name': f"a_{profile}",
                'value_system': transform_weights_to_tuple(profile),
                "n_agents": society_data['default_n_agents'],
                "data": society_data['default_data'],
            })
    # print(agent_profiles)

    d: dict = society_data['agents']
    society_data['agents'] = sorted(d, key=lambda x: x['name'])
    return society_data

def parse_loss_class(loss_class: str, loss_kwargs: dict):

    if loss_class == PrefLossClasses.CROSS_ENTROPY_CLUSTER.value:
        loss_class = BaseVSLClusterRewardLoss
    elif loss_class == PrefLossClasses.SOBA.value:
        # TODO: modified versions here...
        loss_class = SobaLoss
    elif loss_class == PrefLossClasses.LAGRANGE.value:
        # TODO: modified versions here...
        loss_class = ConstrainedLoss
    else:
        raise ValueError(
            "Unsupported for clustering VSL or unrecognized loss_class: ", loss_class)
        
    return loss_class, loss_kwargs

def create_environments(make_env, parser_args, environment_data, alg_config, expert_policy_kwargs):

    wrappers_train, normalizer = parse_wrappers_from_json(
        alg_config['default_wrappers'], alg_config['discount_factor'])
    wrappers_eval, normalizer = parse_wrappers_from_json(
        alg_config['default_wrappers'], alg_config['discount_factor'])
    if 'eval_extra_wrappers' in expert_policy_kwargs:
        wrappers_extra_eval, normalizer = parse_wrappers_from_json(
            expert_policy_kwargs['eval_extra_wrappers'], alg_config['discount_factor'])
        wrappers_eval.extend(wrappers_extra_eval)
    if 'train_extra_wrappers' in expert_policy_kwargs:
        wrappers_extra_train, normalizer_extra = parse_wrappers_from_json(
            expert_policy_kwargs['train_extra_wrappers'], alg_config['discount_factor'])
        wrappers_train.extend(wrappers_extra_train)
    vectorizer, vectorizer_kwargs = parse_vectorizer_from_json(
        expert_policy_kwargs.get('vectorizer', None))

    extra_kwargs = {}
    if parser_args.environment == 'ffmo':
        extra_kwargs = {
            'feature_selection': FeatureSelectionFFEnv(environment_data['feature_selection']),
            'initial_state_distribution': environment_data['initial_state_distribution'],
            # "horizon": environment_data['horizon']
        }
    elif parser_args.environment == 'rw':
        raise NotImplementedError("Old RW environment not supported anymore.")
        """extra_kwargs = {
            'feature_selection': FeatureSelection(environment_data['feature_selection']) if isinstance(environment_data['feature_selection'], str) else tuple([FeatureSelection(fs) for fs in environment_data['feature_selection']]),
            'feature_preprocessing': FeaturePreprocess(environment_data['feature_preprocessing']),
            'masked': True,
            'destination_method': environment_data['destination_method']
            # "horizon": environment_data['horizon']
        }"""
    elif parser_args.environment == 'mvc':
        extra_kwargs = {
            'feature_selection': MVFS(environment_data['feature_selection']),
            # "horizon": environment_data['horizon'],
            # if isinstance(environment_data['feature_selection'], str) else tuple([MVFS(fs) for fs in environment_data['feature_selection']]),
            'mo_version': True
        }
    elif parser_args.environment == 'livroom':
        extra_kwargs = {
            "width": environment_data['width'],
            "height": environment_data['height'],
            "start_location": environment_data['start_location'],
            "goal_location": environment_data['goal_location'],
            "rubbish_location": environment_data['rubbish_location'],
            "table_location": environment_data['table_location'],
            "cat_location": environment_data['cat_location'],
            "obstacle_locations": environment_data['obstacle_locations'],
            "time_penalty": environment_data['time_penalty'],
            "goal_reward": environment_data['goal_reward'],
            "displacement_penalty": environment_data['displacement_penalty'],
            "cat_penalty": environment_data['cat_penalty'],
        }
    eval_creator = make_env(environment_data['name'], environment_data['horizon'],
                            alg_config['discount_factor'], wrappers_eval, parser_args.seed, extra_kwargs, vectorizer=None, vectorizer_kwargs={})
    train_creator = make_env(environment_data['name'], environment_data['horizon'],
                             alg_config['discount_factor'], wrappers_train, parser_args.seed, extra_kwargs, vectorizer=vectorizer, vectorizer_kwargs=vectorizer_kwargs)

    return eval_creator, train_creator


def parse_args()-> Tuple[argparse.ArgumentParser, argparse._ArgumentGroup, argparse._ArgumentGroup, argparse._ArgumentGroup]:
    # IMPORTANT: Default Args are specified depending on the eval_environment in config.json

    parser = argparse.ArgumentParser(
        description="This script will generate a total of n_agents * trajectory_pairs of trajectories, and a chain of comparisons between them, per agent type, for the society selected. See the societies.json and algorithm_config.json files")

    general_group = parser.add_argument_group('General Parameters')
    general_group.add_argument(
        '-dname', '--dataset_name', type=str, default='', required=True, help='Dataset name')
    

    
    general_group.add_argument('-dev', '--device', type=parse_device, default='auto', choices=['auto', 'cuda', 'cpu'],
                               help='Device. Use auto to select the best available device')
    
    general_group.add_argument('-expol', '--exp_policy', default=None,
                               help="policy to be used to create the datasets")
    

    general_group.add_argument(
        '-wbe', "--wandb-entity", type=str, help="Wandb entity to use", required=False)

    general_group.add_argument('-cf', '--config_file', type=str, default='algorithm_config_pc.json',
                               help='Path to JSON general configuration file (overrides other defaults here, but not the command line arguments)')
    general_group.add_argument('-sf', '--society_file', type=str, default='societies.json',
                               help='Path to JSON society configuration file (overrides other defaults here, but not the command line arguments)')
    general_group.add_argument('-sname', '--society_name', type=str, default='default',
                               help='Society name in the society config file (overrides other defaults here, but not the command line arguments)')

    general_group.add_argument('-e', '--environment', type=str, required=True, choices=[
                               'rw', 'ffmo', 'vrw', 'mvc', 'livroom', 'moral', 'dst', 'mine'], help='environment (roadworld - rw, firefighters - ff, variable dest roadworld - vrw, multivalued car - mvc)')

    general_group.add_argument('-df', '--discount_factor', type=float, default=1.0,
                               help='Discount factor. For some environments, it will be neglected as they need a specific discount factor.')

    general_group.add_argument(
        '-s', '--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    general_group.add_argument('-varhz', '--end_trajs_when_ended', action='store_true', default=False,
                               help="Allow trajectories to end when the environment says an episode is done or horizon is reached, whatever happens first, instead of forcing all trajectories to have the length of the horizon")
    general_group.add_argument('-tsize', '--test_size', type=float,
                               default=0.0, help='Ratio_of_test_versus_train_preferences')
    alg_group = parser.add_argument_group('Algorithm-specific Parameters')

    general_group.add_argument('-a', '--algorithm', type=str, choices=[
                               'pc', 'pbmorl', 'cpbmorl'], default='pc', help='Algorithm to use')

    

    env_group = parser.add_argument_group('environment-specific Parameters')

    env_group.add_argument('-reps', '--reward_epsilon', default=0.0, type=float,
                           help='Distance between the cummulative rewards of each pair of trajectories for them to be considered as equal in the comparisons')

    return parser, general_group, env_group, alg_group


def parse_args_train():
    parser, general_group, env_group, alg_group = parse_args()

    general_group.add_argument('-ename', '--experiment_name', type=str,
                                default='test_experiment', required=True, help='Experiment name')

    general_group.add_argument('-sp', '--split_ratio', type=float, default=0.5,
                                help='Test split ratio. Not used unless the data is not already splitted.')

    

    pc_group = alg_group.add_argument_group(
        'Preference Comparisons Parameters')
    pc_group.add_argument('-dfp', '--discount_factor_preferences', type=float,
                          default=1.0, help='Discount factor for preference comparisons')
    
    alg_group.add_argument('-L', '--L_clusters', type=int, required=True,
                           help="Number of clusters per value (overriging configuration file)")
    
    general_group.add_argument('-dtype', '--dtype', type=parse_dtype_torch, default=torch.float32, choices=[torch.float16, torch.float32, torch.float64],
                               help="Reward data to be saved in this numpy format")
    general_group.add_argument('-wb', '--use_wandb', action='store_true', default=False,
                               help="Use Weights and Biases for experiment tracking")
    general_group.add_argument('-rs', '--resume_from', type=int, default=False,
                               help="Resume training from a specific checkpoint")
    general_group.add_argument('-pol', '--policy', default=None,
                               help="policy to train agents")
    return parser.parse_args()


def parse_args_generate_dataset():
    
    parser, general_group, env_group, alg_group = parse_args()


    general_group.add_argument(
        '-pareto', '--remain_with_pareto_optimal_agents', action='store_true', default=False,
        help="Generate only Pareto front agents for the selected society")
    
    general_group.add_argument('-gentr', '--gen_trajs', action='store_true', default=False,
                               help="Generate new trajs for the selected society")

    general_group.add_argument('-genpf', '--gen_preferences', action='store_true', default=False,
                               help="Generate new preferences among the generated trajectories")
    
    env_group.add_argument('-rt', '--retrain', action='store_true',
                           default=False, help='Retrain experts')
    env_group.add_argument('-rf', '--refine', action='store_true',
                           default=False, help='Refine experts')
    env_group.add_argument('-appr', '--is_tabular', action='store_true',
                           default=False, help='Approximate expert')
    
    general_group.add_argument('-dtype', '--dtype', type=parse_dtype_numpy, default=np.float32, choices=[np.float16, np.float32, np.float64],
                               help="Reward data to be saved in this numpy format")
    return parser.parse_args()


def parse_enames_list(learning_curve_from):
    """
    Parses the learning curve from the given string.
    """
    if not learning_curve_from:
        return []
    return [name.strip() for name in learning_curve_from.split(',') if name.strip()]

def parse_args_evaluate():
    # IMPORTANT: Default Args are specified depending on the environment in config.json


    parser, general_group, env_group, alg_group = parse_args()

    general_group.add_argument('-sp', '--split_ratio', type=float, default=0.5,
                                help='Test split ratio. Not used unless the data is not already splitted.')

    

    general_group = parser.add_argument_group('General Parameters')
    general_group.add_argument('-ename', '--experiment_name', type=str,
                                default='test_experiment', required=True, help='Experiment name')
    general_group.add_argument('-seeds', '--seeds', type=eval, default=None,
                               help='Number of seeds for the experiment.')
    general_group.add_argument('-Ltries', '--Ltries', type=eval, default=None,
                               help='Number of L tries for the experiment.')

    # subfig_multiplier
    general_group.add_argument('-subfm', '--subfig_multiplier', type=float, default=6.0,
                               help='Scales subfigs inside the plots.')
    general_group.add_argument('-pfont', '--plot_fontsize', type=int, default=15,
                               help='Font size in plots.')
    general_group.add_argument('-seps', '--sampling_epsilon', type=float, default=0.1,
                               help='Exploration rate for sampling.')
    general_group.add_argument('-strajs', '--sampling_trajs_per_agent', type=int, default=200,
                               help='Number of trajes to sample with expert/learned policies.')
    general_group.add_argument('-dicfrom', '--dunn_index_curve_from', type=parse_enames_list,
                               default=None, help="Generate the learning curve for the specified experiments")
    general_group.add_argument('-lrcfrom', '--learning_curve_from', type=parse_enames_list,
                               default=None, help="Generate the learning curve for the specified experiments")
    general_group.add_argument(
        '-scf', '--show_only_config', action='store_true', default=False, required=False, help='Only show the training configuration used.')
    general_group.add_argument('-pol', '--policy', default=None,
                               help="policy training agents")
    
    general_group.add_argument('-dtype', '--dtype', type=parse_dtype_torch, default=torch.float32, choices=[torch.float16, torch.float32, torch.float64],
                               help="Reward data to be saved in this numpy format")
    
    return parser.parse_args()