from copy import deepcopy
import os
import pprint
from typing import Dict
import numpy as np
import torch


from defines import transform_weights_to_tuple
from generate_dataset_mo import create_environments, make_env, parse_society_data

from parsing import parse_args_evaluate, parse_args_train, parse_feature_extractors, parse_learning_policy_parameters, parse_loss_class, parse_optimizer_data
from src.algorithms.preference_based_vsl_simple import PVSL
from src.dataset_processing.data import VSLPreferenceDataset
from src.dataset_processing.utils import calculate_expert_policy_save_path
from src.policies.mushroom_agent_mobaselines import MOBaselinesAgent, ProxyPolicy
from src.reward_nets.vsl_reward_functions import RewardVectorModule, parse_layer_name
from src.utils import filter_none_args, load_json_config

from morl_baselines.common.evaluation import seed_everything
from src.algorithms.plot_utils import evaluate as evaluate_solutions


import os
import pprint
from typing import Dict
import numpy as np

from src.policies.mushroom_agent_mobaselines import MOBaselinesAgent
from src.utils import filter_none_args, load_json_config


from baraacuda.utils.wrappers import RewardVectorFunctionWrapper
from imitation.util.networks import RunningNorm


import argparse
import os
import pprint
import random
from typing import Dict, List

import numpy as np
import torch
from src.algorithms.clustering_utils_simple import ClusterAssignment, ClusterAssignmentMemory
from src.dataset_processing.data import VSLPreferenceDataset
from parsing import parse_optimizer_data
from src.utils import filter_none_args, load_json_config
import matplotlib.pyplot as plt
import pandas as pd

from train_movsl import retrieve_datasets


def contextual_feature_analysis(experiment_name, values_names, dataset_reference: VSLPreferenceDataset, assignment: ClusterAssignment, label='train_set', assignment_identifier='', fontsize=16):
    # print(list(dataset_reference.data_per_agent[dataset_reference.agent_ids[0]].fragments1[0].infos))
    all_context_features = [dataset_reference.data_per_agent[agent_id].fragments1[0].infos[0]
                            ['agent_context'] for agent_id in dataset_reference.agent_ids]
    max_context_features = np.max(all_context_features, axis=0)
    context_features_per_cluster = []

    for clust_idx, agent_group in enumerate(assignment.assignment_vs):
        if len(agent_group) == 0:
            context_features_per_cluster.append(None)
            continue
        # Extract and normalize context features for the current cluster
        context_features_cidx = np.array(
            [dataset_reference.data_per_agent[agent_id].fragments1[0].infos[0]['agent_context'] for agent_id in agent_group])

        context_features_per_cluster.append(context_features_cidx)

        value_system = assignment.get_value_system(clust_idx)
    # Plot all features in a single barplot with standard error bars for each cluster
    feature_names = ["Houshold Income", "Car available",
                     "Conmuting", "Shopping", "Business", "Leisure"]
    cluster_data = []

    for cluster_idx, agent_group in enumerate(assignment.assignment_vs):
        if len(agent_group) == 0:
            continue
        cluster_features = context_features_per_cluster[cluster_idx]
        value_system = assignment.get_value_system(cluster_idx)

        num_agents = len(cluster_features)
        perc_increase_over_mean = (np.mean(cluster_features, axis=0) - np.mean(
            all_context_features, axis=0)) / np.mean(all_context_features, axis=0) * 100
        means = np.mean(cluster_features, axis=0)
        std_errors = np.sqrt(np.var(cluster_features, axis=0) / len(cluster_features) + np.var(
            all_context_features, axis=0) / len(all_context_features)) / np.mean(all_context_features, axis=0) * 100

        # Format the value system with names and values
        value_system_str = ", ".join(
            f"{name}: {value:.2f}" for name, value in zip(values_names, value_system))

        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(means) + 1), perc_increase_over_mean,
                yerr=std_errors, capsize=5, alpha=0.7, color='skyblue')
        plt.xticks(range(1, len(means) + 1),
                   [feature_names[i] for i in range(len(means))], fontsize=fontsize)
        plt.ylim(-1.4 * 100, 1.4 * 100)  # Set y-axis
        plt.yticks(np.arange(-1.4 * 100, 1.5 * 100, 0.1 * 100),
                   rotation=45, fontsize=fontsize)
        plt.title(
            f"Barplot of Context Features for Cluster {cluster_idx + 1} (Agents: {num_agents})\n(Value System: {value_system_str})", fontsize=fontsize)
        plt.xlabel("Features", fontsize=fontsize)
        plt.ylabel("Percentage increase/decrease over average",
                   fontsize=fontsize)
        plt.grid(axis='y')

        # Save the plot
        plot_dir = os.path.join('test_results', experiment_name,
                                label, 'plots', 'context_features', assignment_identifier)
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(
            plot_dir, f"cluster_{cluster_idx + 1}_context_features.pdf")
        plt.savefig(plot_path)
        plt.close()

        # Add cluster data to the table
        cluster_row = [f"{mean:.2f} ({{{perc:+.2f}\\%}})" for mean,
                       perc in zip(means, perc_increase_over_mean)]
        cluster_data.append(
            [f"Cluster {cluster_idx + 1}", num_agents] + cluster_row)

    # Add overall mean data to the table
    overall_means = np.mean(all_context_features, axis=0)
    overall_row = [f"{mean:.2f}" for mean in overall_means]
    cluster_data.append(
        ["Overall", len(dataset_reference.agent_ids)] + overall_row)

    # Save the table to CSV and LaTeX
    table_path_csv = os.path.join(
        'test_results', experiment_name, label, 'tables', 'context_features', 'csv')
    table_path_latex = os.path.join(
        'test_results', experiment_name, label, 'tables', 'context_features', 'latex')
    os.makedirs(table_path_csv, exist_ok=True)
    os.makedirs(table_path_latex, exist_ok=True)
    table_path_csv = os.path.join(
        table_path_csv, f"context_features_{assignment_identifier}.csv")
    table_path_latex = os.path.join(
        table_path_latex, f"context_features_{assignment_identifier}.tex")

    # Create DataFrame
    df = pd.DataFrame(cluster_data, columns=[
                      "Cluster", "Number of Agents"] + feature_names)

    # Save to CSV
    df.to_csv(table_path_csv, index=False)

    # Save to LaTeX
    with open(table_path_latex, "w") as f:
        f.write(df.to_latex(index=False, escape=False))

    print(
        f"Saved context features table to {table_path_csv} and {table_path_latex}")



def evaluate():
    parser_args = filter_none_args(parse_args_evaluate())
    # If a config file is specified, load it and override command line args
    config = load_json_config(parser_args.config_file)
    society_config = load_json_config(parser_args.society_file)
    pprint.pprint(parser_args)

    seed_everything(parser_args.seed)
    environment_data = config[parser_args.environment]

    society_config_env = society_config['ff' if parser_args.environment ==
                                        'ffmo' else parser_args.environment]
    society_data = parse_society_data(parser_args, society_config_env)
    alg_config = environment_data['algorithm_config'][parser_args.algorithm]
    dataset_name = parser_args.dataset_name

    experiment_name = parser_args.experiment_name
    experiment_name = experiment_name  # + '_' + str(parser_args.split_ratio)

    dataset_train, dataset_test = retrieve_datasets(
        environment_data, society_data, dataset_name, rew_epsilon=parser_args.reward_epsilon, split_ratio=parser_args.split_ratio)

    agent_profiles = [tuple(ag['value_system'])
                      for ag in society_data['agents']]

    learning_policy_class = alg_config['learning_policy_class']
    if hasattr(parser_args, 'policy') and parser_args.policy is not None:
        learning_policy_class = parser_args.policy
    learning_policy_kwargs: Dict = alg_config['learning_policy_kwargs'][learning_policy_class]

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

    eval_creator, train_creator = create_environments(
        make_env, parser_args, environment_data, alg_config, learning_policy_kwargs)

    train_environment_no_reward, train_environment_no_reward_mush = train_creator()
    eval_environment, eval_environment_mush = eval_creator()

    epclass, epkwargs_no_train_kwargs, base_policy_train_kwargs, is_single_objective = parse_learning_policy_parameters(
        parser_args, environment_data, society_data, train_environment_no_reward, train_environment_no_reward_mush, learning_policy_kwargs, learning_policy_class)

    reward_net_features_extractor_class, features_extractor_kwargs = parse_feature_extractors(
        train_environment_no_reward, parser_args.environment, environment_data, dtype=parser_args.dtype, device=parser_args.device)

    data_reward_net = environment_data['default_reward_net']
    data_reward_net.update(alg_config['reward_net'])

    features_extractor = reward_net_features_extractor_class(
        use_state=data_reward_net['use_state'],
        use_action=data_reward_net['use_action'],
        use_next_state=data_reward_net['use_next_state'],
        use_done=data_reward_net['use_done'],
        **features_extractor_kwargs)

    reward_net = RewardVectorModule(
        hid_sizes=data_reward_net['hid_sizes'],
        num_outputs=society_data['n_values'],
        basic_classes=[parse_layer_name(
            l) for l in data_reward_net['basic_layer_classes']],
        activations=[parse_layer_name(l)
                     for l in data_reward_net['activations']],
        # negative_grounding_layer=data_reward_net['negative_grounding_layer'],
        use_bias=data_reward_net['use_bias'],
        clamp_rewards=data_reward_net['clamp_rewards'],
        feature_extractor=features_extractor,
        normalize_output_layer=RunningNorm if data_reward_net.get(
            'normalize_output', False) else None,
        normalize_output=data_reward_net['normalize_output'],
        update_stats=data_reward_net['normalize_output'],
        debug=True,

    )
    train_environment = RewardVectorFunctionWrapper(
        train_environment_no_reward, reward_vector_function=reward_net)
    mobaselines_agent = MOBaselinesAgent(env=train_environment, eval_env=eval_environment,
                                         agent_class=epclass, agent_kwargs=epkwargs_no_train_kwargs, mdp_info=train_environment_no_reward_mush.info,
                                         train_kwargs=base_policy_train_kwargs, is_single_objective=is_single_objective, weights_to_sample=None,
                                         name='learner' + learning_policy_class)
    rng_for_algorithms = np.random.default_rng(parser_args.seed)

    # TODO: normalize environment observations... for both reward net and policy?

    opt_kwargs, opt_class = parse_optimizer_data(environment_data, alg_config)

    alignment_layer_class, alignment_layer_kwargs = parse_layer_name(data_reward_net['basic_layer_classes'][-1]), {
        'in_features': society_data['n_values'],
        'out_features': 1,
        'bias': False,
        'device': parser_args.device,
        # 'dtype': parser_args.dtype,
    }
    loss_class, loss_kwargs = parse_loss_class(
        alg_config['loss_class'], alg_config['loss_kwargs'])

    plot_di_scores = False
    if not hasattr(parser_args, 'Ltries'):
        parser_args.Ltries = None
    if not hasattr(parser_args, 'seeds'):
        parser_args.seeds = None
    if isinstance(parser_args.Ltries, int):
            parser_args.Ltries = [parser_args.Ltries]
    if isinstance(parser_args.seeds, int):
        parser_args.seeds = [parser_args.seeds]

    if parser_args.seeds is not None and parser_args.Ltries is not None:
        experiments = [parser_args.experiment_name + "_L" + str(Ltry) + "_seed" + str(seed) for seed in parser_args.seeds for Ltry in parser_args.Ltries]
        plot_di_scores = True
    elif parser_args.seeds is not None:
        experiments = [parser_args.experiment_name + "_seed" + str(seed) for seed in parser_args.seeds]
    elif parser_args.Ltries is not None:
        experiments = [parser_args.experiment_name + "_L" + str(Try) for Try in parser_args.Ltries]
        plot_di_scores = True
    else:
        experiments = [parser_args.experiment_name]

    best_assignment_list, historic, trained_agent, global_step, config_ = PVSL.load_state(
        ename=experiments[0], agent_name=mobaselines_agent.name, ref_env=eval_environment, ref_eval_env=eval_environment)
    print("CONFIG.", experiments[0])
    pprint.pprint(config_)
    
    vsl_algo = PVSL.load_from_state(best_assignment_list, historic, trained_agent, global_step, config_)
    
    """parser_args.L_clusters = len(historic[-1].weights_per_cluster)
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

        use_wandb=False,
        # ,
        debug_mode=alg_config['debug_mode'],
    )"""
    vsl_algo.mobaselines_agent.set_envs(train_environment, eval_environment)

    expert_policy: MOBaselinesAgent = MOBaselinesAgent.load(env=train_environment_no_reward, eval_env=eval_environment,
                                                            path=calculate_expert_policy_save_path(
                                                                environment_name=parser_args.environment,
                                                                dataset_name=dataset_name,
                                                                society_name=parser_args.society_name,
                                                                class_name=ProxyPolicy.__name__,
                                                                grounding_name='default'), name='exp'+expert_policy_class,)
    
    expert_policy.set_envs(train_environment_no_reward, eval_environment)

    policy_train_kwargs = deepcopy(base_policy_train_kwargs)
    # shorter learning steps, etc.
    policy_train_kwargs.update(
        alg_config['train_kwargs']['policy_train_kwargs'])
    alg_config['train_kwargs']['policy_train_kwargs'] = policy_train_kwargs
    if hasattr(parser_args, 'discount_factor_preferences'):
        alg_config['train_kwargs']['discount_factor_preferences'] = parser_args.discount_factor_preferences
    """
    #seed_everything(26)
    ret = vsl_algo.train_algo(
        algo=parser_args.algorithm,
        env_name=parser_args.environment,
        experiment_name=parser_args.experiment_name,
        tags=[],
        dataset=dataset_train,
        train_env=train_environment_no_reward,
        eval_env=eval_environment,
        resume_from=parser_args.resume_from,
        **alg_config['train_kwargs']
    )
    print("DONE", ret)
"""
    run_dir = f'results/{parser_args.environment}/experiments/{parser_args.experiment_name}/'

    os.makedirs(run_dir, exist_ok=True)

    

    evaluate_solutions(vsl_algo, enames_all=experiments,
                       algo_name=parser_args.algorithm,
                       plot_di_scores=plot_di_scores,
                       expert_policy=expert_policy,
                       test_dataset=dataset_test, ref_env=train_environment_no_reward, ref_eval_env=eval_environment, environment_data=environment_data,
                       discount=alg_config['discount_factor'],
                       run_dir=run_dir,
                       known_pareto_front=policy_train_kwargs.get(
        'known_pareto_front', None),
        num_eval_weights_for_front=policy_train_kwargs.get(
            'num_eval_weights_for_front', 20),
        num_eval_episodes_for_front=policy_train_kwargs.get(
            'num_eval_episodes_for_front', 20),
        fontsize=parser_args.plot_fontsize, sampling_trajs_per_agent=parser_args.sampling_trajs_per_agent, sampling_epsilon=parser_args.sampling_epsilon)


if __name__ == "__main__":
    # main_minecart()
    evaluate()
