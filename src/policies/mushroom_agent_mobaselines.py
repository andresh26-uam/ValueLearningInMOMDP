from collections import OrderedDict
from copy import deepcopy
import random
import shutil
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from zipfile import ZipFile
import mushroom_rl
from mushroom_rl.core import Agent, MDPInfo, Serializable
from mushroom_rl.policy import EpsGreedy, Policy
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.multi_policy.pcn.pcn import PCN
from morl_baselines.multi_policy.envelope.envelope import Envelope
from mushroom_rl.utils.parameters import Parameter
import tempfile

from morl_baselines.common.pareto import filter_pareto_dominated, get_non_pareto_dominated_inds
from morl_baselines.common.weights import equally_spaced_weights, random_weights

import os
from mushroom_rl.utils.parameters import to_parameter
import numpy as np
import dill
import pickle
from pathlib import Path

from defines import CHECKPOINTS, transform_weights_to_tuple
from src.dataset_processing.data import TrajectoryWithValueSystemRews
from src.policies.vsl_policies import LearnerValueSystemLearningPolicy, ValueSystemLearningPolicy

import gymnasium as gym

from src.reward_nets.vsl_reward_functions import RewardVectorModule


def save_zip(self, zip_file, full_save, folder=''):
        """
        Serialize and save the agent to the given path on disk.

        Args:
            zip_file (ZipFile): ZipFile where te object needs to be saved;
            full_save (bool): flag to specify the amount of data to save for
                MushroomRL data structures;
            folder (string, ''): subfolder to be used by the save method.
        """
        primitive_dictionary = dict()

        for att, method in self._save_attributes.items():

            if not method.endswith('!') or full_save:
                method = method[:-1] if method.endswith('!') else method
                attribute = getattr(self, att) if hasattr(self, att) else None

                if attribute is not None:
                    if method == 'primitive':
                        primitive_dictionary[att] = attribute
                    elif method == 'none':
                        pass
                    elif hasattr(self, '_save_{}'.format(method)):
                        save_method = getattr(self, '_save_{}'.format(method))
                        file_name = "{}.{}".format(att, method)
                        save_method(zip_file, file_name, attribute,
                                    full_save=full_save, folder=folder)
                    else:
                        raise NotImplementedError(
                            "Method _save_{} is not implemented for class '{}'".
                                format(method, self.__class__.__name__)
                        )

        config_data = dict(
            type=type(self),
            save_attributes=self._save_attributes,
            primitive_dictionary=primitive_dictionary
        )

        self._save_pickle(zip_file, 'config', config_data, folder=folder)

def compare_values(p1, p2):
    if isinstance(p1, Parameter) and isinstance(p2, Parameter):
        print("Comparing Parameters")
        return (p1._max_value == p2._max_value) and (p1._min_value == p2._min_value) and (p1._initial_value == p2._initial_value) and np.allclose(p1._n_updates.table, p2._n_updates.table)
    elif isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
        print("Comparing NP")
        return np.allclose(p1, p2)
    elif p1 is None and p2 is None:
        print("Comparing p1p2")
        return True
    elif isinstance(p1, th.Tensor) and isinstance(p2, th.Tensor):
        print("Comparing tensor")
        return th.allclose(p1, p2)
    elif isinstance(p1, list) and isinstance(p2, list):
        print("Comparing list")
        if len(p1) != len(p2):
            print("list")
            return False
        for a, b in zip(p1, p2):
            v = compare_values(a, b)
            if not v:
                print("list elements")
                return False
        return True
    elif isinstance(p1, dict) and isinstance(p2, dict):
        print("Comparing Dict")
        for key, val in p1.items():
            if key not in p2:
                print("keys")
                return False
            v = compare_values(val, p2[key])
            print("Key:", key, "Value:", v)
            if not v:
                print("dict elements")
                return False
        return True
    else:
        print("Not comparable types", type(p1), type(p2))
        return all(p1 == p2)
class ProxyPolicy(Policy):
    
    def parameters(self):
        base = {'weights': self.weights, 'epsilon': self.epsilon}
        if isinstance(self.agent_mobaselines, Envelope):
            base['params'] = list(self.agent_mobaselines.q_net.parameters())
        elif isinstance(self.agent_mobaselines, PCN):
            base['params'] = list(self.agent_mobaselines.model.parameters())
        elif isinstance(self.agent_mobaselines, LearnerValueSystemLearningPolicy):
            base['params'] = dict({k: list(v.policy.parameters()) for k,v in self.agent_mobaselines.learner_per_align_func.items()})
        else:
            raise NotImplementedError(f"Unsupported agent_morl type: {type(self.agent_mobaselines)}")
        return base
    def compare_parameters(self, other):
        """
        Compare the parameters of this policy with another policy.
        :param other: The other policy to compare with.
        :return: True if the parameters are equal, False otherwise.
        """
        
        if not isinstance(other, ProxyPolicy):
            print("Not a ProxyPolicy")
            return True # Compare abstractt policies?????
            return False
        p1 = self.parameters() 
        p2 = other.parameters()
        return compare_values(p1, p2)

    def __init__(self, agent_morl: MOAgent, weight_default=None, epsilon=0.1):
        super().__init__()
        
        self.set_agent_baselines(agent_morl)
        if weight_default is not None:
            self.set_weight(weight_default)
        else:
            self.set_weight(np.ones(agent_morl.reward_dim)/agent_morl.reward_dim)
        
        self.set_epsilon(epsilon)
        self._add_save_attr(**{
            'weights': 'numpy',
            'epsilon': 'primitive',

        })

    
    def set_agent_baselines(self, agent_mobaselines: MOAgent):
        self.agent_mobaselines = agent_mobaselines
        if hasattr(self, 'weights'):
            self.set_weight(self.weights)

    def set_weight(self, weight):
        self.weights = weight
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = to_parameter(epsilon)
    def get_epsilon(self):
        return self.epsilon
    
    def draw_action(self, state, info=None, get_action_info=False):
        if random.random() < self.epsilon.get_value():
            if info is not None:
                mask = info.get('action_masks', None)
                if mask is not None:
                    a = int(np.random.choice(np.where(mask > 0)[0]))
                else:
                    a = self.agent_mobaselines.action_space.sample()
            else:
                a = self.agent_mobaselines.action_space.sample()
        else:
            a = self.agent_mobaselines.eval(state, w=self.weights)
                #print("Action drawn:", a, "with weights:", self.weights)
                
        return [a]
        raise NotImplementedError("NOT A MOPOLICY...")

from baraacuda.agents.mo_agents import MO_DQN
import baraacuda
from baraacuda.utils.miscellaneous import get_signature_params
import torch as th
import gymnasium as gym
from mushroom_rl.core.environment import Environment as MushroomEnvironment
from mushroom_rl.algorithms.actor_critic import DDPG

from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
MUSHROOM_ALGOS = {
    'MO_DQN': MO_DQN,
    'DDPG': DDPG,
}
SINGLE_OBJECTIVE_ALGOS = {
    'MO_DQN': MO_DQN,
    'DDPG': DDPG,
}

class WeightsToAlgos(object):

    def compare_parameters(self, other):
        """
        Compare the parameters of this mapping with another mapping.
        :param other: The other mapping to compare with.
        :return: True if the parameters are equal, False otherwise.
        """
        if not isinstance(other, WeightsToAlgos):
            return False
        if len(self.weights_to_index) != len(other.weights_to_index):
            return False
        for k, v in self.weights_to_index.items():
            if k not in other.weights_to_index or other.weights_to_index[k] != v:
                print("Keys or values do not match:", k, v, other.weights_to_index.get(k, None))
                return False
        for algo1, algo2 in zip(self.algorithm_dict.values(), other.algorithm_dict.values()):
            if algo1.__class__ != algo2.__class__:
                print("Algorithms do not match:", algo1.__class__, algo2.__class__)
                return False 
        print("Parameters match")   
        return True
    def __init__(self, weights_to_algos: Dict[str, Agent] ):
        self.weights_to_index = dict()
        self.algorithm_dict = OrderedDict()
        counter = 0
        self.size_weights = None

        for k, v in weights_to_algos.items():
            if not isinstance(v, Agent):
                raise ValueError(f"Expected MOAgent for key {k}, got {type(v)}")
            self.weights_to_index[transform_weights_to_tuple(k, self.size_weights)] = counter
            self.algorithm_dict[counter] = v
            self.size_weights = len(k)
            counter += 1
        
    
    
    def get_algo(self, weight_or_index: Union[int, tuple]) -> Agent:
        """
        Get the algorithm associated with the given weight.
        :param weight: The weight to look up.
        :return: The algorithm associated with the weight.
        """
        if isinstance(weight_or_index, int):
            return self.algorithm_dict[weight_or_index]
        weights_real = transform_weights_to_tuple(weight_or_index, self.size_weights)
        return self.algorithm_dict[self.weights_to_index[weights_real]]

    def contains(self, weight_or_index: Iterable) -> bool:
        """
        Check if the mapping contains the given weight or index.
        :param weight_or_index: The weight or index to check.
        :return: True if the mapping contains the weight or index, False otherwise.
        """
        if isinstance(weight_or_index, int):

            a = weight_or_index in self.algorithm_dict.keys()
            if a :
                assert self.weights_to_index.values().count(weight_or_index) == 1, "Inconsistent state in WeightsToAlgos"
        else:
            weights_real = transform_weights_to_tuple(weight_or_index, self.size_weights)
            a = weights_real in self.weights_to_index.keys()
            if a: 
                assert self.algorithm_dict.get(self.weights_to_index[weights_real], None) is not None, "Inconsistent state in WeightsToAlgos"
        return a
    def add_algo(self, weight: Iterable, algo: Agent, warn_override=False, override_existent=None):
        """
        Add a new algorithm to the mapping.
        :param weight: The weight to associate with the algorithm.
        :param algo: The algorithm to add.
        """
        weights_real = transform_weights_to_tuple(weight, self.size_weights)
        if self.contains(weights_real):
            if warn_override:
                raise Warning(f"Warning: Overriding existing algorithm for weight {weight}")
            self.algorithm_dict[self.weights_to_index[weights_real]] = algo

        else:
            if override_existent is not None:
                assert isinstance(override_existent, int), "override_existent must be an integer index"
                if override_existent in self.weights_to_index.values():
                    self.weights_to_index = {k: v for k, v in self.weights_to_index.items() if int(v) != int(override_existent)}
                self.weights_to_index[weights_real] = override_existent
                self.algorithm_dict[override_existent] = algo
                # pop the items in self.weights_to_index that have as value override_existent:
            else:
                self.weights_to_index[weights_real] = len(self.algorithm_dict)
                self.algorithm_dict[len(self.algorithm_dict)] = algo

        assert len(self.weights_to_index) == len(self.algorithm_dict)
from stable_baselines3.common.policies import BaseModel

class MOBaselinesAgent(Agent, Serializable):
    """
    Base class for multi-objective baselines agents.
    This class extends the Agent class from mushroom_rl.core.
    """

    def __init__(self, env: gym.Env, eval_env: gym.Env, agent_class: type[MOAgent], agent_kwargs: Dict, mdp_info:  MDPInfo, is_single_objective: bool, weights_to_sample: Optional[List[Tuple]]=None,
                 features=None,train_kwargs={},  name='MOBaselinesAgent'):
        assert env is not None, "Environment must be provided."
        assert eval_env is not None, "Evaluation environment must be provided."
        assert not isinstance(env, MushroomEnvironment), "Invalid environment type."
        super().__init__(mdp_info, None, features)
        self.name = name
        self.agent_class = agent_class

        self.agent_kwargs = agent_kwargs
        self.is_single_objective = is_single_objective 
        if weights_to_sample is None or len(weights_to_sample) == 0:
                weights_to_sample = [np.ones((env.get_wrapper_attr('reward_dim'),)) / float(env.get_wrapper_attr('reward_dim'))]
            
        sparams = get_signature_params(self.agent_class, self.agent_kwargs)
        if self.is_single_objective:
            self.weights_to_algos = WeightsToAlgos({})
                #raise ValueError("weights_to_sample must be provided for single-objective agents")
            for w in weights_to_sample:
                sparams['weights'] = np.array(w, dtype=np.float32)
                agent_mobaselines = self.agent_class(env=env, **sparams)
                self.weights_to_algos.add_algo(w, agent_mobaselines, warn_override=True)
            #agent_mobaselines = self.weights_to_algos.get_algo(weights_to_sample[0])
            #self._policy = ProxyPolicy(self.agent_mobaselines, weight_default=None, epsilon=0.0)
                #self.agent_kwargs['weights'] = np.ones((env.get_wrapper_attr('reward_dim'),)) / float(env.get_wrapper_attr('reward_dim'))
        else:
            agent_mobaselines = self.agent_class(env=env, **sparams)

        self.agent_mobaselines = agent_mobaselines
        self.current_weights = weights_to_sample[0] if weights_to_sample is not None else np.ones((env.get_wrapper_attr('reward_dim'),)) / float(env.get_wrapper_attr('reward_dim'))
        self.policy = ProxyPolicy(self.agent_mobaselines, weight_default=weights_to_sample[0], epsilon=0.0)

        self.train_kwargs = train_kwargs    
        self.set_envs(env, eval_env)

        """if 'pcn' in self.name:
            self._pcn_desired_return = np.asarray(self.agent_morl.desired_return)
            self._add_save_attr(**{
                '_pcn_desired_return': 'pickle'
            })"""
        self._add_save_attr(**{
                'policy': 'mushroom',
                'agent_class': 'pickle',
                'agent_kwargs': 'pickle',
                'is_single_objective': 'primitive',
                'current_weights': 'numpy',
                'train_kwargs': 'pickle',
                'features': 'pickle' if features is not None else 'none',
            })
        self._add_save_attr(**{
            'name': 'pickle',
            'mdp_info': 'pickle',
        })
        
    
    def get_weights(self):
        return self.current_weights
    
    def save_agent(self, agent_mobaselines: MOAgent, base_dir='', identification='main'):
        path=os.path.join(base_dir, f'{identification}')
        if isinstance(agent_mobaselines, ValueSystemLearningPolicy):
            path = path + '_st'
            
            agent_mobaselines.save(path=path)
        else:
            path = path + '_mo'
            assert isinstance(agent_mobaselines, MOAgent), "Agent must be a MOAgent."
            agent_mobaselines.save(save_dir=path, filename=f'file', save_replay_buffer=False)
        #test_agent_load = self.load_agent(base_dir, identification=identification, ref_env=self.env, weights=self.get_weights())
        #print(vars(test_agent_load).keys())
        #print(vars(agent_mobaselines).keys())
        #assert set(list(vars(agent_mobaselines).keys()))== set(list(vars(test_agent_load).keys())), f"Agent {identification} not saved correctly, loaded agent does not match original agent."

    def load_agent(self, base_dir: str, identification='main', ref_env: gym.Env = None, weights=None):
        path_t = os.path.join(base_dir, f'{identification}')
        if os.path.exists(path_t + '_st'):
            assert issubclass(self.agent_class, ValueSystemLearningPolicy), "real type: {}".format(self.agent_class.__name__)  # type: ignore
            
            agent_morl= self.agent_class.load(ref_env=ref_env, path=path_t + '_st')
            
        else:
            assert os.path.exists(path_t + '_mo'), f"Path {path_t} does not exist."
            path_t = path_t + '_mo'
            args = deepcopy(self.agent_kwargs)
            args.update({'weights': weights})
            const_kwargs = get_signature_params(self.agent_class, self.agent_kwargs)
            agent_morl = self.agent_class(env=ref_env, **const_kwargs)
            path_t = os.path.join(path_t, f'file')
            if os.path.exists(path_t):
                agent_morl.load(path=path_t)
            else:
                if 'envelope' in self.name.lower():
                    path_t1 = path_t+'.tar' # Envelope...
                    agent_morl.load(path=path_t1)
                else:
                    raise FileNotFoundError(f"Could not load, unrecognized agent: {self.name}, at {path_t}")
        return agent_morl
    
    def save(self, path, full_save=False):
        if 'checkpoints' not in path:
            path = os.path.join(CHECKPOINTS, path)
        #remove anything inside the path
        
            #os.makedirs(path)
        if os.path.exists(path):
            # REMOVE THE FILE OR DIR WITH THAT NAME
            if os.path.isfile(path):
                    os.remove(path)
            elif os.path.isdir(path):
                    shutil.rmtree(path)
        
        return super().save(path, full_save)
    def save_zip(self, zip_file: ZipFile, full_save, folder=''):
        

        ff = os.path.join(os.path.dirname(zip_file.filename), f'{self.name}_agent')
        os.makedirs(ff, exist_ok=True)

        self.save_agent(self.agent_mobaselines, base_dir=ff, identification='main')
        if self.is_single_objective:
            # save the weights to algos
            for windex, algo in self.weights_to_algos.algorithm_dict.items():
                print("SAVED AGENT, ", windex)
                self.save_agent(algo, base_dir=ff, identification=f'a_{windex}')
            with open(os.path.join(ff, f'weights_to_index.pkl'), 'wb') as f:
                dill.dump(self.weights_to_algos.weights_to_index, f)
        # save this with dill
        with open(os.path.join(ff, f'agent_class.pkl'), 'wb') as f:
            print("SAVING TO", os.path.join(ff, f'agent_class.pkl'))
            dill.dump(self.agent_class, f)
        
        super().save_zip(zip_file, full_save, folder)
        
    
    def load(env, eval_env, path, name=None):
        """
        Load the agent from a given path.
        :param path: Path to the saved agent.
        """
        if 'checkpoints' not in path:
            path = os.path.join(CHECKPOINTS, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Loading path not found: {path}")
        self = super(MOBaselinesAgent, MOBaselinesAgent).load(path)
        self.name = name if name is not None else self.name
        if self.name == 'LearnerValueSystemLearningPolicy':
            self.name = 'ppo_learner'
        print(self.name, "Loading from path:", path)
        # Call the superclass load method to load the agent's configuration
        with open(os.path.join(os.path.dirname(path), f'{self.name}_agent', 'agent_class.pkl'), 'rb') as f:
            self.agent_class = dill.load(f)
        
        ff = os.path.join(os.path.dirname(path), f'{self.name}_agent')
        agent_morl_main = self.load_agent(base_dir=ff, identification='main', ref_env=env)

        
        if self.is_single_objective:
                self.weights_to_algos = WeightsToAlgos({})
                with open(os.path.join(os.path.dirname(path), f'{self.name}_agent', 'weights_to_index.pkl'), 'rb') as f:
                    self.weights_to_algos.weights_to_index = dill.load(f)
                    self.weights_to_algos.algorithm_dict = dict()
                for weight, index_ in self.weights_to_algos.weights_to_index.items():
                    agent_morl = self.load_agent( base_dir=ff, identification=f'a_{index_}', ref_env=env, weights=weight)
                    self.weights_to_algos.algorithm_dict[index_] = agent_morl
        
        self.agent_mobaselines = agent_morl_main
        self.set_envs(env, eval_env)
        self.policy.set_agent_baselines(agent_morl_main)
        self.policy.set_weight(self.get_weights())
    
        return self

    

    def set_weights(self, weights, override_existent: int = None):
        """
        Set the weights for the agent's policy.
        :param weights: Weights to set for the policy.
        """
        self.current_weights = weights
        
        if isinstance(weights, list) or isinstance(weights, tuple):
            weights = np.array(weights, dtype=np.float32)
        if self.is_single_objective:
            uparams = get_signature_params(self.agent_class, self.agent_kwargs)
            if not self.weights_to_algos.contains(weights):
                    self.weights_to_algos.add_algo(weights, self.agent_class(env=self.env, **uparams), warn_override=True, override_existent=override_existent)
            self.agent_mobaselines = self.weights_to_algos.get_algo(weights)
            self.agent_mobaselines = self.weights_to_algos.get_algo(weights)
            self.agent_mobaselines.set_weight(weights)
            self.policy.set_agent_baselines(self.agent_mobaselines)

        self.policy.set_weight(weights)
        """if isinstance(self.agent_morl, PCN):
            print("WEGIHTS", weights, self._pcn_desired_return)
            
            self.agent_morl.set_desired_return_and_horizon(weights*self._pcn_desired_return, 
            self.agent_morl.desired_horizon)"""
    

    def set_envs(self, env=None, eval_env=None):
        """
        Set the evaluation environment for the agent.
        :param eval_env: The evaluation environment to set.
        """
        if eval_env is not None:
            self.eval_env = eval_env
        if env is not None:
            self.env = env
            self.agent_mobaselines.set_env(env)
            if self.is_single_objective:
                for windex, algo in self.weights_to_algos.algorithm_dict.items():
                    algo.set_env(env)
    def set_reward_vector_function(self, reward_vector, warn_relabel_not_possible=True):
        """
        Set the reward vector for the agent.
        :param reward_vector: The reward vector to set.
        """
        self.env.get_wrapper_attr('set_reward_vector_function')(reward_vector)
        self.set_envs(env=self.env)
        #self.eval_env.get_wrapper_attr('set_reward_vector_function')(reward_vector)
        if self.is_single_objective:
            for windex, algo in self.weights_to_algos.algorithm_dict.items():
                algo.set_reward_vector_function(reward_vector, warn_relabel_not_possible=warn_relabel_not_possible)
        else:
            if hasattr(self.agent_mobaselines, 'set_reward_vector_function'):
                self.agent_mobaselines.set_reward_vector_function(reward_vector, warn_relabel_not_possible=warn_relabel_not_possible)

    def fit(self, dataset=None, **info):
        """
        Fit step for the agent.
        :param dataset: The dataset to fit the agent on.
        :param info: Additional information for fitting.
        """
        if self.is_single_objective:
                #assert isinstance(self.agent_mobaselines, MOAgent), "Agent must be a MOAgent."
            self.train_kwargs['verbose'] = True
            weight=self.get_weights()
            self.agent_mobaselines = self.weights_to_algos.get_algo(self.get_weights())
            self.agent_mobaselines.set_weight(self.get_weights())
            
            print("Training agent with weights:", weight)
            print([p.__class__.__name__ for p in self.weights_to_algos.algorithm_dict.values()])
        else:
            weight = None
        
        train_kwargs = get_signature_params(self.agent_mobaselines.train, self.train_kwargs)
        
        self.policy.set_agent_baselines(self.agent_mobaselines)
        self.policy.set_weight(self.get_weights())
        print("Training agent with weights:", self.get_weights())
        if isinstance(self.policy.weights, th.Tensor):
            assert np.allclose(self.policy.weights.detach().cpu().numpy(), self.get_weights().detach().cpu().numpy())  
        else:
            assert np.allclose(self.policy.weights , self.get_weights())
        #self.policy.set_epsilon(0.5)
        self.agent_mobaselines.train(eval_env=self.eval_env, weight=weight, **train_kwargs)
        
        self.policy.set_agent_baselines(self.agent_mobaselines)
        self.policy.set_weight(self.get_weights())
    
            
        """if isinstance(self.agent_morl, PCN):
            self.agent_morl.train(**self.train_kwargs)
        else:
            raise NotImplementedError(f"Fit method not implemented for MOBaselinesAgent {self.agent_morl.__class__.__name__}.")
"""

    def pareto_front(self, num_eval_episodes_for_front=1000, num_eval_weights_for_front=100, discount=1.0, use_weights=None):
        """
        Get the Pareto front of the agent.
        :param n_points: Number of points in the Pareto front.
        :return: Pareto front points.
        """
        if use_weights is not None:
            eval_weights = use_weights
        
        else:
            if self.is_single_objective:
                eval_weights = list(self.weights_to_algos.weights_to_index.keys())
            else:
                
                eval_weights =  equally_spaced_weights(self.agent_mobaselines.reward_dim, n=num_eval_weights_for_front)
            

        current_front = []
        for w in eval_weights:
            
            trajs_pure, (scalarized_return,
                scalarized_discounted_return,
                vec_return,
                disc_vec_return) = obtain_trajectories_and_eval_mo(self,self.eval_env, ws=[w], ws_eval=[w], reward_vector=None,
                                                                    discount=discount,scalarization=np.dot, seed=0, render=False, agent_name=self.name+f"_{w}",
                                                                    n_seeds= num_eval_episodes_for_front)
            current_front.append(disc_vec_return)
        
            #TODO: eval_weights_to_algos = [k, self.weights_to_algos.get_algo(k) for k in self.weights_to_algos.weights_to_index.keys()]


        """
            elif isinstance(self.agent_mobaselines, Envelope):
            eval_weights =  equally_spaced_weights(self.agent_mobaselines.reward_dim, n=num_eval_weights_for_front)
            prev_gamma = self.agent_mobaselines.gamma
            self.agent_mobaselines.gamma = discount
            current_front = [
                self.agent_mobaselines.policy_eval(self.eval_env, num_episodes=num_eval_episodes_for_front, weights=ew, log=self.agent_mobaselines.log)[3]
                for ew in eval_weights
            ]
            self.agent_mobaselines.gamma = prev_gamma
            raise NotImplementedError(f"Pareto front not implemented for this agent type. {self.agent_mobaselines.__class__.__name__}")
        """
        filtered = get_non_pareto_dominated_inds(current_front)
        return (np.asarray(current_front)[filtered], np.asarray(eval_weights)[filtered]), (np.asarray(current_front), np.asarray(eval_weights))
        
from baraacuda.utils.wrappers import RewardVectorFunctionWrapper

def obtain_trajectory_and_eval_mo(
    agent: MOBaselinesAgent,
    env: gym.Env,
    size_weights: Optional[np.ndarray] = None,
    w_eval: Optional[np.ndarray] = None,
    scalarization=np.dot,
    discount=1.0,
    seed=None,
    render: bool = False,
    collect_real_reward_only=False,
    agent_name: str = 'unk',
) -> Tuple[TrajectoryWithValueSystemRews, Tuple[float, float, np.ndarray, np.ndarray]]:
    """Evaluates one episode of the agent in the environment.

    Args:
        agent: Agent
        env: MO-Gymnasium environment with LinearReward wrapper
        scalarization: scalarization function, taking weights and reward as parameters
        w (np.ndarray): Weight vector
        render (bool, optional): Whether to render the environment. Defaults to False.
        reward_vector: Optional callable for reward vector function
        seed: Random seed for reproducibility
        exploration (float, optional): Exploration rate for the agent. Defaults to 0.0.
        agent_name: Name of the agent for logging purposes


    Returns:
        (TrajectoryWithValueSystemRews, (float, float, np.ndarray, np.ndarray)): Trajectory with value system rewards and evaluation metrics [Scalarized return, scalarized discounted return, vectorized return, vectorized discounted return].
    """
    observations = []
    actions = []
    v_rews = []
    rews = []
    v_rews_real = []
    rews_real = []
    dones = []
    infos = []
    
    vec_return_real =  np.zeros((size_weights, ), dtype=np.float32)
    disc_vec_return_real =  np.zeros((size_weights, ), dtype=np.float32)
    vec_return, disc_vec_return = np.zeros((size_weights, ), dtype=np.float32), np.zeros((size_weights, ), dtype=np.float32)
        

    obs, info = env.reset(seed=seed)


    #w_eval = w_eval if w_eval is not None else w
    #w = np.asarray(w, dtype=np.float32) if w is not None else None
    w_eval = np.asarray(w_eval, dtype=np.float32) if w_eval is not None else None
    
    
    gamma = 1.0 
    
    observations.append(obs)
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        if render:
            env.render()
        a = agent.policy.draw_action(obs, info)[0]
        obs, r, terminated, truncated, info = env.step(a)
        if isinstance(r, th.Tensor): r = r.detach().cpu().numpy()
        assert r.shape == vec_return_real.shape, f"Reward shape {r.shape} does not match weight vector {vec_return_real.shape}"
        assert len(r.shape) == 1, f"Reward shape {r.shape} should be 1-dimensional"
        
        vec_return += r
        disc_vec_return += gamma * r
        observations.append(obs)
        actions.append(a)
        v_rews.append(r)
        infos.append(info)
        rews.append(scalarization(w_eval, r))
        dones.append(terminated)
        
        real_reward = info.get('untransformed_reward', None)
        if real_reward is not None:
            v_rews_real.append(real_reward)
            rews_real.append(scalarization(w_eval, real_reward))
            vec_return_real += real_reward
            disc_vec_return_real += gamma * real_reward

        gamma *= discount

    if w_eval is None:
        scalarized_return = scalarization(vec_return)
        scalarized_discounted_return = scalarization(disc_vec_return)
    else:
        scalarized_return = scalarization(w_eval, vec_return)
        scalarized_discounted_return = scalarization(w_eval, disc_vec_return)
    
    if len(v_rews_real) > 0:
        if w_eval is None:
            scalarized_return_real = scalarization(vec_return_real)
            scalarized_discounted_return_real = scalarization(disc_vec_return_real)
        else:
            scalarized_return_real = scalarization(w_eval, vec_return_real)
            scalarized_discounted_return_real = scalarization(w_eval, disc_vec_return_real)
    # maybe a problem with the terminal state...
    return TrajectoryWithValueSystemRews(obs=np.array(observations), 
                                         acts=np.array(actions), 
                                         infos=np.array(infos), rews=np.array(rews),
                                         dones=np.array(dones, dtype=np.float32),
                                         v_rews_real=np.array(v_rews_real).T if len(v_rews_real) > 0 else None,
                                         rews_real=np.array(rews_real) if len(rews_real) > 0 else None,
                                           terminal=terminated, n_vals=size_weights, v_rews=np.array(v_rews).T, agent=agent_name), ((
        scalarized_return,
        scalarized_discounted_return,
        vec_return,
        disc_vec_return,
) if len(v_rews_real) == 0 else (
    scalarized_return,
        scalarized_discounted_return,
        vec_return,
        disc_vec_return,
        scalarized_return_real,
        scalarized_discounted_return_real,
        vec_return_real,
        disc_vec_return_real,

))

def obtain_trajectories_and_eval_mo(
    agent: MOBaselinesAgent,
    env: gym.Env,
    ws: Optional[List] = [None],
    ws_eval: Optional[np.ndarray] = [None],
    reward_vector: Optional[RewardVectorModule] = None,
    exploration=0.0, 
    discount=1.0,
    scalarization=np.dot,
    seed=None,
    n_seeds: int = 100,
    warn_relabel_not_possible=True,
    repeat_per_seed: int = 1,
    render: bool = False,
    agent_name: str = 'unk') -> Tuple[TrajectoryWithValueSystemRews, Tuple[float, float, np.ndarray, np.ndarray]]:
    """Obtains trajectories from the agent in the environment.
    Args:
        agent: Agent
        env: MO-Gymnasium environment with LinearReward wrapper
        scalarization: scalarization function, taking weights and reward as parameters
        w (np.ndarray): Weight vector
        render (bool, optional): Whether to render the environment. Defaults to False.
        exploration (float, optional): Exploration rate for the agent. Defaults to 0.0.
        reward_vector: Optional callable for reward vector function 
        seed: Random seed for reproducibility
        n_seeds: Number of seeds to use for evaluation
        repeat_per_seed: Number of times to repeat the evaluation for each seed
        agent_name: Name of the agent for logging purposes

    Returns:
        List[TrajectoryWithValueSystemRews], (float, float, np.ndarray, np.ndarray): List of trajectories with value system rewards and evaluation metrics (see eval_mo_w_trajectory for details).
    """
    trajectories = []
    count_evals = 0
    total_eval = None
    old_epsilon = agent.policy.get_epsilon()
    old_weights = agent.get_weights()

    prev_envs = agent.env, agent.eval_env
    if reward_vector is not None:
        if env.has_wrapper_attr('set_reward_vector_function'):
            env.get_wrapper_attr('set_reward_vector_function')(reward_vector)
        else:
            env = RewardVectorFunctionWrapper(env, reward_vector)
        agent.set_envs(env=env, eval_env=env)
        agent.set_reward_vector_function(reward_vector, warn_relabel_not_possible=warn_relabel_not_possible)
        collect_real_reward_only = False
    else:
        collect_real_reward_only = True

    for si in range(n_seeds):
        for w, w_eval in zip(ws, ws_eval):
            agent.set_weights(w)
            agent.policy.set_epsilon(exploration)
            for r in range(repeat_per_seed):
                traj, evaluation = obtain_trajectory_and_eval_mo(
                    agent=agent, env=env, w_eval=w_eval, scalarization=scalarization, collect_real_reward_only=collect_real_reward_only, 
                    seed=seed*n_seeds+si, render=render, agent_name=agent_name, discount=discount,
                    size_weights = len(w)
                    
                )
                trajectories.append(traj)
                if total_eval is None:
                    total_eval = list(evaluation)
                else:
                    for i in range(len(total_eval)):
                        total_eval[i] += evaluation[i]
                count_evals += 1
    for i in range(len(total_eval)):
        total_eval[i] /= count_evals 
    

    agent.set_envs(*prev_envs)
    agent.policy.set_epsilon(old_epsilon)
    agent.set_weights(old_weights)
    return trajectories, tuple(total_eval)
