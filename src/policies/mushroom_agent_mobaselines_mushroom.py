import random
from typing import Callable, Dict, Iterable, Optional, Tuple, Type, Union
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
        return (p1._max_value == p2._max_value) and (p1._min_value == p2._min_value) and (p1._initial_value == p2._initial_value) and np.allclose(p1._n_updates.table, p2._n_updates.table)
    elif isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
        return np.allclose(p1, p2)
    elif p1 is None and p2 is None:
        return True
    elif isinstance(p1, th.Tensor) and isinstance(p2, th.Tensor):
        return th.allclose(p1, p2)
    elif isinstance(p1, list) and isinstance(p2, list):
        
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
        for key, val in p1.items():
            if key not in p2:
                print("keys")
                return False
            v = compare_values(val, p2[key])
            if not v:
                print("dict elements")
                return False
        return True
    else:
        print("Not comparable types", type(p1), type(p2))
        return p1 == p2
class ProxyPolicy(Policy):
    
    def parameters(self):
        base = {'weights': self.weights, 'epsilon': self.epsilon}
        if isinstance(self.agent_morl, Envelope):
            base['params'] = list(self.agent_morl.q_net.parameters())
        elif isinstance(self.agent_morl, PCN):
            base['params'] = list(self.agent_morl.model.parameters())
        else:
            raise NotImplementedError(f"Unsupported agent_morl type: {type(self.agent_morl)}")
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
        
        self.set_agent_morl(agent_morl)
        if weight_default is not None:
            self.set_weight(weight_default)
        else:
            self.set_weight(np.ones(agent_morl.reward_dim)/agent_morl.reward_dim)
        
        self.epsilon = to_parameter(epsilon)
        self._add_save_attr(**{
            'weights': 'numpy',
            'epsilon': 'primitive'
        })

    
    def set_agent_morl(self, agent_morl: MOAgent):
        self.agent_morl = agent_morl
        if hasattr(self, 'weights'):
            self.set_weight(self.weights)

    def set_weight(self, weight):
        self.weights = weight
        
        
    
    def set_epsilon(self, epsilon):
        self.epsilon = to_parameter(epsilon)

    
    def draw_action(self, state, info=None, get_action_info=False):
        if isinstance(self.agent_morl, MOPolicy):
            if random.random() < self.epsilon.get_value():
                a = self.agent_morl.action_space.sample()
                #print("RANDOM ACTION SELECTED:", a)
            else:
                a = self.agent_morl.eval(state, w=self.weights)
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

class WeightsToAlgos(Serializable):
    def __init__(self, weights_to_algos: Dict[str, Agent] ):
        self.weights_to_index = dict()
        self.algorithm_list = []
        counter = 0
        for k, v in weights_to_algos.items():
            if not isinstance(v, Agent):
                raise ValueError(f"Expected MOAgent for key {k}, got {type(v)}")
            self.weights_to_index[k] = counter
            self.algorithm_list.append(v)
            counter += 1
        self._add_save_attr(**{
            'weights_to_index': 'primitive',
            'algorithm_list': 'mushroom',
        })
    def get_algo(self, weight_or_index: Union[int, tuple]) -> Agent:
        """
        Get the algorithm associated with the given weight.
        :param weight: The weight to look up.
        :return: The algorithm associated with the weight.
        """
        if isinstance(weight_or_index, Iterable):
            weights_real = tuple(weight_or_index)
            return self.algorithm_list[self.weights_to_index[weights_real]]
        return self.algorithm_list[weight_or_index]

    def contains(self, weight_or_index: Iterable) -> bool:
        """
        Check if the mapping contains the given weight or index.
        :param weight_or_index: The weight or index to check.
        :return: True if the mapping contains the weight or index, False otherwise.
        """
        if isinstance(weight_or_index, Iterable):
            weights_real = tuple(weight_or_index)
            return weights_real in self.weights_to_index
        else: 
            raise ValueError(f"Expected Iterable, got {type(weight_or_index)}")
    def add_algo(self, weight: Iterable, algo: Agent, warn_override=False):
        """
        Add a new algorithm to the mapping.
        :param weight: The weight to associate with the algorithm.
        :param algo: The algorithm to add.
        """
        if self.contains(weight):
            if warn_override:
                raise Warning(f"Warning: Overriding existing algorithm for weight {weight}")
            self.weights_to_index[tuple(weight)] = len(self.algorithm_list)
            self.algorithm_list[self.weights_to_index[tuple(weight)]] = algo
        else:
            if not isinstance(algo, Agent):
                raise ValueError(f"Expected MOAgent, got {type(algo)}")
            if isinstance(weight, Iterable):
                weights_real = tuple(weight)
                self.weights_to_index[weights_real] = len(self.algorithm_list)
                self.algorithm_list.append(algo)
            else:
                self.weights_to_index[weight] = len(self.algorithm_list)
                self.algorithm_list.append(algo)
class MOBaselinesAgent(Agent):
    """
    Base class for multi-objective baselines agents.
    This class extends the Agent class from mushroom_rl.core.
    """
    def __init__(self, env: gym.Env, eval_env: gym.Env, agent_class: type[MOAgent], agent_kwargs: Dict, mdp_info:  MDPInfo, 
                 features=None,train_kwargs={}, name='MOBaselinesAgent'):
        assert env is not None, "Environment must be provided."
        assert eval_env is not None, "Evaluation environment must be provided."
        assert not isinstance(env, MushroomEnvironment), "Invalid environment type."
        super().__init__(mdp_info, None, features)
        self.name = name
        self.agent_class = agent_class

        self.env = env
        self.agent_kwargs = agent_kwargs
        #self.agent_kwargs['weights'] = np.ones((env.get_wrapper_attr('reward_dim'),)) / float(env.get_wrapper_attr('reward_dim'))
        sparams = get_signature_params(self.agent_class, self.agent_kwargs)
        self.agent_morl = self.agent_class(env=env, **sparams)
        
        if isinstance(self.agent_morl, MOAgent):
            self.is_mushroom = False
            self.policy = ProxyPolicy(self.agent_morl, weight_default=None, epsilon=0.0)
            self.train_kwargs = train_kwargs
            self.set_envs(env, eval_env)
            
            self._add_save_attr(**{
                'policy': 'mushroom',
                'agent_class': 'pickle',
                'agent_kwargs': 'pickle',
                'features': 'pickle' if features is not None else 'none',
            })
            if 'pcn' in self.name:
                self._pcn_desired_return = np.asarray(self.agent_morl.desired_return)
                self._add_save_attr(**{
                    '_pcn_desired_return': 'pickle'
                })
        else:
            self.is_mushroom = True
            self.policy = self.agent_morl.policy

            self.train_kwargs = train_kwargs
            self._add_save_attr(**{
                'agent_morl': 'mushroom',
                'agent_class': 'pickle',
                'agent_kwargs': 'pickle',
                'policy': 'mushroom',
                'features': 'pickle' if features is not None else 'none',
            })
        self._add_save_attr(**{
            'is_mushroom': 'pickle',
            'name': 'pickle',
            'mdp_info': 'pickle',
        })
        if self.is_single_objective:
            self.weights_to_algos = WeightsToAlgos({})
            self._add_save_attr(**{
                'weights_to_algos': 'mushroom',
            })

    def get_weights(self):
        if self.is_mushroom:
            a = self.agent_morl
            a : MO_DQN
            return self.policy.scalarizer.weights
        else:
            return self.policy.weights
    def save_zip(self, zip_file: ZipFile, full_save, folder=''):

        ff = os.path.join(os.path.dirname(zip_file.filename), f'{self.name}_agent')
        os.makedirs(ff, exist_ok=True)
        
        if not self.is_mushroom:
            self.agent_morl.save(save_dir=ff, filename=f'morl_weights')
        
        # save this with dill
            with open(os.path.join(ff, f'agent_class.pkl'), 'wb') as f:
                dill.dump(self.agent_class, f)
        super().save_zip(zip_file, full_save, folder)
        
    
    def load(env, eval_env, path, name=None):
        """
        Load the agent from a given path.
        :param path: Path to the saved agent.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Loading path not found: {path}")
        self = super(MOBaselinesAgent, MOBaselinesAgent).load(path)
        self.name = name if name is not None else self.name
        print(self.name, "Loading from path:", path)
        is_mushroom = any(self.name.endswith(a) for a in MUSHROOM_ALGOS.keys())
        self.is_mushroom = is_mushroom
        if not is_mushroom:
            # Call the superclass load method to load the agent's configuration
            with open(os.path.join(os.path.dirname(path), f'{self.name}_agent', 'agent_class.pkl'), 'rb') as f:
                self.agent_class = dill.load(f)
            
            const_kwargs = get_signature_params(self.agent_class, self.agent_kwargs)
            
            agent_morl = self.agent_class(env=env, **const_kwargs)

            path_t = os.path.join(os.path.dirname(path), f'{self.name}_agent', 'morl_weights')
            if os.path.exists(path_t):
                if os.path.isdir(path_t):
                    agent_morl.load(path=path_t)
                elif os.path.isfile(path_t):
                    agent_morl.load(path=path_t)
            else:
                if 'pcn' in self.name :
                    path_t1 = path_t+'.pt' # PCN
                    if os.path.exists(path_t1) and callable(self.agent_class):
                        agent_morl.load(path=path_t1)
                    self._pcn_desired_return = np.asarray(agent_morl.desired_return)
                    self._add_save_attr(**{
                        '_pcn_desired_return': 'pickle'
                    })
                    
                elif 'envelope' in self.name:
                    path_t1 = path_t+'.tar' # Envelope...
                    agent_morl.load(path=path_t1)
            self.agent_morl = agent_morl
            self.set_envs(env, eval_env)
            self.policy.set_agent_morl(agent_morl)
            #self.policy = ProxyPolicy(agent_morl=self.agent_morl, weight_default=None)
            
            
        else:
            self = super(MOBaselinesAgent, MOBaselinesAgent).load(path)
            self.policy = self.agent_morl.policy
            self.set_envs(env, eval_env)
        
        
        return self

    
    
    def draw_action(self, state, info=None, get_action_info=False):
        
        if self.is_mushroom:
            return super().draw_action(state, info, get_action_info)
        else:
            return self.policy.draw_action(state, info, get_action_info)

    def set_weights(self, weights):
        """
        Set the weights for the agent's policy.
        :param weights: Weights to set for the policy.
        """
        
        if isinstance(weights, list) or isinstance(weights, tuple):
            weights = np.array(weights, dtype=np.float32)
        if self.is_single_objective:
            
            if not self.weights_to_algos.contains(weights):
                uparams = get_signature_params(self.agent_class, self.agent_kwargs)
                self.weights_to_algos.add_algo(weights, self.agent_class(env=self.env, **uparams), warn_override=True)
            
            if isinstance(self.policy, ProxyPolicy):
                self.policy.set_agent_morl(self.weights_to_algos.get_algo(weights))

                self.agent_morl = self.weights_to_algos.get_algo(weights)
            else:
                self.policy = self.weights_to_algos.get_algo(weights).policy
                self.agent_morl = self.weights_to_algos.get_algo(weights)

        if self.is_mushroom:

            if isinstance(self.agent_morl, MO_DQN):
                
                self.policy.scalarizer.weights = weights  # This is DQN

            else:
                raise NotImplementedError(f"Unsupported agent_morl type: {type(self.agent_morl)}")
                
        else:
            self.policy.set_weight(weights)
            if isinstance(self.agent_morl, PCN):
                self.agent_morl.set_desired_return_and_horizon(weights*self._pcn_desired_return, 
                self.agent_morl.desired_horizon)
        

    def set_envs(self, env=None, eval_env=None):
        """
        Set the evaluation environment for the agent.
        :param eval_env: The evaluation environment to set.
        """
        if eval_env is not None:
            self.eval_env = eval_env
        if env is not None:
            self.env = env
            self.agent_morl.env = env
        if self.is_single_objective:
            for windex, algo in enumerate(self.weights_to_algos.algorithm_list):
                algo.env = env
    @property
    def is_single_objective(self):
        ret = False
        for v in SINGLE_OBJECTIVE_ALGOS.values():
            ret = ret or isinstance(self.agent_morl, v)
        return ret

    def fit(self, dataset, **info):
        """
        Fit step for the agent.
        :param dataset: The dataset to fit the agent on.
        :param info: Additional information for fitting.
        """
        if self.is_single_objective:
                if self.is_mushroom:
                    assert isinstance(self.agent_morl, Agent), "Agent must be a Mushroom Agent."
                    return self.weights_to_algos.get_algo(self.get_weights()).fit(dataset, **info)
                else:
                    assert isinstance(self.agent_morl, MOAgent), "Agent must be a MOAgent."
                    self.train_kwargs['verbose'] = True
                    train_kwargs = get_signature_params(self.agent_morl.train, self.train_kwargs)
                    return self.weights_to_algos.get_algo(self.get_weights()).train(eval_env=self.eval_env, **train_kwargs)
        
        elif self.is_mushroom:
            self.agent_morl.fit(dataset, **info)

            
        else:
                assert isinstance(self.agent_morl, MOAgent), "Agent must be a MOAgent."
                self.agent_morl: MOAgent
                # get only the kwargs that are needed for training
                self.train_kwargs['verbose'] = True
                train_kwargs = get_signature_params(self.agent_morl.train, self.train_kwargs)
                
                self.agent_morl.train(eval_env=self.eval_env, **train_kwargs)
    
            
        """if isinstance(self.agent_morl, PCN):
            self.agent_morl.train(**self.train_kwargs)
        else:
            raise NotImplementedError(f"Fit method not implemented for MOBaselinesAgent {self.agent_morl.__class__.__name__}.")
"""

    def pareto_front(self, num_eval_episodes_for_front=1000, num_eval_weights_for_front=100):
        """
        Get the Pareto front of the agent.
        :param n_points: Number of points in the Pareto front.
        :return: Pareto front points.
        """
        if self.is_mushroom:
            raise NotImplementedError("Pareto front not implemented for MushroomRL agents.")
            return self.agent_morl.pareto_front(n_points=num_eval_weights_for_front)
        else:
        
            if isinstance(self.agent_morl, Envelope):
                eval_weights =  equally_spaced_weights(self.agent_morl.reward_dim, n=num_eval_weights_for_front)
                current_front = [
                    self.agent_morl.policy_eval(self.eval_env, num_episodes=num_eval_episodes_for_front, weights=ew, log=self.agent_morl.log)[3]
                    for ew in eval_weights
                ]
            else:
                raise NotImplementedError(f"Pareto front not implemented for this agent type. {self.agent_morl.__class__.__name__}")
            filtered = get_non_pareto_dominated_inds(current_front)
            return np.asarray(current_front)[filtered], np.asarray(eval_weights)[filtered]
        


