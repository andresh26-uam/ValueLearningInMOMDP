from copy import deepcopy
import os
import pprint
from typing import Dict
import numpy as np
import torch



from defines import transform_weights_to_tuple
from generate_dataset_mo import create_environments, make_env, parse_expert_policy_parameters, parse_society_data

from parsing import parse_args_train, parse_feature_extractors, parse_learning_policy_parameters, parse_loss_class, parse_optimizer_data
from src.algorithms.preference_based_vsl_simple import PVSL
from src.dataset_processing.data import VSLPreferenceDataset
from src.dataset_processing.utils import DATASETS_PATH, calculate_dataset_save_path, calculate_expert_policy_save_path
from src.policies.mushroom_agent_mobaselines import MOBaselinesAgent, ProxyPolicy
from src.reward_nets.vsl_reward_functions import ConvexAlignmentLayer, RewardVectorModule, parse_layer_name
from src.utils import filter_none_args, load_json_config

import wandb
from morl_baselines.common.evaluation import seed_everything



import argparse
import os
import pprint
import random
from typing import Dict
import numpy as np

from src.policies.mushroom_agent_mobaselines import MOBaselinesAgent
from mushroom_rl.core import Core
from src.utils import filter_none_args, load_json_config



def retrieve_datasets( environment_data, society_data, dataset_name, rew_epsilon=0.0, split_ratio=0.5):
    try:
        path = os.path.join(
        DATASETS_PATH, calculate_dataset_save_path(dataset_name, environment_data, society_data, epsilon=rew_epsilon))
        dataset_train = VSLPreferenceDataset.load(
            os.path.join(path, "dataset_train.pkl"))
        dataset_test = VSLPreferenceDataset.load(
            os.path.join(path, "dataset_test.pkl"))
        print("LOADING DATASET SPLIT.")
    except FileNotFoundError:
        print("LOADING DATASET FULL. THEN DIVIDE")
        path = os.path.join(
            DATASETS_PATH, calculate_dataset_save_path(dataset_name, environment_data, society_data, epsilon=rew_epsilon))

        dataset = VSLPreferenceDataset.load(os.path.join(path, "dataset.pkl"))

        dataset_test = VSLPreferenceDataset(
            n_values=dataset.n_values, single_agent=False)
        dataset_train = VSLPreferenceDataset(
            n_values=dataset.n_values, single_agent=False)
        for aid, adata in dataset.data_per_agent.items():
            selection = np.arange(int(split_ratio * len(adata)))
            train_selection = np.arange(
                int(split_ratio * len(adata)), len(adata))
            agent_dataset_batch = adata[selection]
            dataset_test.push(fragments=agent_dataset_batch[0], preferences=agent_dataset_batch[1], preferences_with_grounding=agent_dataset_batch[2], agent_ids=[
                aid]*len(selection), agent_data={aid: dataset.agent_data[aid]})
            agent_dataset_batch_t = adata[train_selection]
            dataset_train.push(fragments=agent_dataset_batch_t[0], preferences=agent_dataset_batch_t[1], preferences_with_grounding=agent_dataset_batch_t[2], agent_ids=[
                aid]*len(train_selection), agent_data={aid: dataset.agent_data[aid]})
                
    return dataset_train, dataset_test

from baraacuda.utils.wrappers import RewardVectorFunctionWrapper
from imitation.util.networks import RunningNorm
def pvsl():
    parser_args = filter_none_args(parse_args_train())
    # If a config file is specified, load it and override command line args
    config = load_json_config(parser_args.config_file)
    society_config = load_json_config(parser_args.society_file)
    pprint.pprint(parser_args)

    seed_everything(parser_args.seed)
    environment_data = config[parser_args.environment]

    society_config_env = society_config['ff' if parser_args.environment == 'ffmo' else parser_args.environment]
    society_data = parse_society_data(parser_args, society_config_env)
    alg_config = environment_data['algorithm_config'][parser_args.algorithm]
    dataset_name = parser_args.dataset_name

    experiment_name = parser_args.experiment_name
    experiment_name = experiment_name #+ '_' + str(parser_args.split_ratio)

    
    dataset_train, dataset_test = retrieve_datasets(environment_data, society_data, dataset_name, rew_epsilon=parser_args.reward_epsilon, split_ratio=parser_args.split_ratio)
    

    agent_profiles = [tuple(ag['value_system'])
                      for ag in society_data['agents']]
    
    expert_policy_class = alg_config['expert_policy_class']
    if hasattr(parser_args, 'exp_policy') and parser_args.exp_policy is not None:
        expert_policy_class = parser_args.exp_policy
    expert_policy_kwargs: Dict = alg_config['expert_policy_kwargs'][expert_policy_class]

    agent_profiles = [transform_weights_to_tuple(ag['value_system'])
                      for ag in society_data['agents']]

    learning_policy_class = alg_config['learning_policy_class']
    if hasattr(parser_args, 'policy') and parser_args.policy is not None:
        learning_policy_class = parser_args.policy
    learning_policy_kwargs: Dict = alg_config['learning_policy_kwargs'][learning_policy_class]
    
    eval_creator, train_creator = create_environments(make_env, parser_args, environment_data, alg_config, learning_policy_kwargs)
    
    train_environment, train_environment_mush = train_creator()
    eval_environment, eval_environment_mush = eval_creator()

    epclass, epkwargs_no_train_kwargs, base_policy_train_kwargs,is_single_objective = parse_learning_policy_parameters(parser_args, environment_data, society_data, train_environment, train_environment_mush, learning_policy_kwargs, learning_policy_class)
    

    reward_net_features_extractor_class, features_extractor_kwargs = parse_feature_extractors(
        train_environment, parser_args.environment, environment_data, dtype=parser_args.dtype, device=parser_args.device)

    data_reward_net = environment_data['default_reward_net']
    data_reward_net.update(alg_config['reward_net'])

    features_extractor = reward_net_features_extractor_class(
                                                    use_state=data_reward_net['use_state'], 
                                                    use_action=data_reward_net['use_action'], 
                                                    use_next_state=data_reward_net['use_next_state'], 
                                                    use_done=data_reward_net['use_done'],
                                                    **features_extractor_kwargs)
    
   
    reward_net = RewardVectorModule(
        num_outputs=society_data['n_values'],
        hid_sizes=data_reward_net['hid_sizes'],
        basic_classes=[parse_layer_name(
            l) for l in data_reward_net['basic_layer_classes']],
        activations=[parse_layer_name(l)
                     for l in data_reward_net['activations']],
        #negative_grounding_layer=data_reward_net['negative_grounding_layer'],
        use_bias=data_reward_net['use_bias'],
        clamp_rewards=data_reward_net['clamp_rewards'],
        feature_extractor=features_extractor,
        normalize_output_layer=RunningNorm if data_reward_net.get('normalize_output', False) else None,
        normalize_output=data_reward_net['normalize_output'],
        update_stats=data_reward_net['normalize_output'],
        debug=True,
        
    )
    train_environment = RewardVectorFunctionWrapper(train_environment, reward_vector_function=reward_net)
    mobaselines_agent = MOBaselinesAgent(env=train_environment, eval_env=eval_environment,
                                         agent_class=epclass, agent_kwargs=epkwargs_no_train_kwargs, mdp_info=train_environment_mush.info,
                                         train_kwargs=base_policy_train_kwargs, is_single_objective=is_single_objective, weights_to_sample=None,
                                         name='learner' + learning_policy_class)
    rng_for_algorithms = np.random.default_rng(parser_args.seed)

    # TODO: normalize environment observations... for both reward net and policy?

    opt_kwargs, opt_class = parse_optimizer_data(environment_data, alg_config)

    alignment_layer_class, alignment_layer_kwargs  = parse_layer_name(data_reward_net['basic_layer_classes'][-1]), {
        'in_features': society_data['n_values'],
        'out_features': 1,
        'bias': False,
        'device': parser_args.device,
        #'dtype': parser_args.dtype,
    }
    loss_class, loss_kwargs = parse_loss_class(alg_config['loss_class'], alg_config['loss_kwargs'])
    vsl_algo = PVSL(
        Lmax=parser_args.L_clusters,
        mobaselines_agent=mobaselines_agent,
        alignment_layer_class=alignment_layer_class,
        alignment_layer_kwargs=alignment_layer_kwargs,
         grounding_network=reward_net,
         loss_class=loss_class,
         loss_kwargs=loss_kwargs,
         reward_kwargs_for_config=data_reward_net,
         optim_class=opt_class,
         optim_kwargs=opt_kwargs,
         n_rewards_for_ensemble=alg_config.get('n_rewards_for_ensemble', 3),
         use_wandb=parser_args.use_wandb,
         #,
         debug_mode=alg_config['debug_mode'],
         )
    
    policy_train_kwargs = deepcopy(base_policy_train_kwargs)
    policy_train_kwargs.update(alg_config['train_kwargs']['policy_train_kwargs']) # shorter learning steps, etc.
    alg_config['train_kwargs']['policy_train_kwargs'] = policy_train_kwargs
    if hasattr(parser_args, 'discount_factor_preferences'):
        alg_config['train_kwargs']['discount_factor_preferences'] = parser_args.discount_factor_preferences

    run_dir = f'results/{parser_args.environment}/experiments/{parser_args.experiment_name}/'
    os.makedirs(run_dir, exist_ok=True)


    expert_policy: MOBaselinesAgent = MOBaselinesAgent.load(env=train_environment, eval_env=eval_environment,
                                                            path=calculate_expert_policy_save_path(
                                                                environment_name=parser_args.environment,
                                                                dataset_name=dataset_name,
                                                                society_name=parser_args.society_name,
                                                                class_name=ProxyPolicy.__name__,
                                                                grounding_name='default'), name='exp'+expert_policy_class,)
    #seed_everything(26)
    ret = vsl_algo.train_algo(
        algo=parser_args.algorithm,
        env_name=parser_args.environment,
        experiment_name=parser_args.experiment_name,
        tags=[],
        dataset=dataset_train,
        train_env=train_environment,
        eval_env=eval_environment,
        resume_from=parser_args.resume_from,
        run_dir=run_dir,
        **alg_config['train_kwargs']
    )
    from src.algorithms.plot_utils import evaluate
    
    
    """evaluate(vsl_algo, [parser_args.experiment_name], test_dataset=dataset_test, ref_env=eval_environment, ref_eval_env=eval_environment,
             environment_data=environment_data,
             expert_policy=expert_policy,
             discount=alg_config['discount_factor'],
             run_dir=run_dir,
             known_pareto_front=policy_train_kwargs.get(
                 'known_pareto_front', None),
                                   num_eval_weights_for_front=policy_train_kwargs.get('num_eval_weights_for_front', 20), 
                                   num_eval_episodes_for_front=policy_train_kwargs.get('num_eval_episodes_for_front', 20), font_size=10, sampling_epsilon=0.05, sampling_trajs_per_agent=200)
    """#print("DONE", ret)

    

if __name__ == "__main__":
    #main_minecart()
    pvsl()
