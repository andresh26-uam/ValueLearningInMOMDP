import shutil
from parsing import create_environments, make_env, parse_args_generate_dataset, parse_epsilon_params_from_json, parse_expert_policy_parameters, parse_society_data, parse_wrappers_from_json
from envs.multivalued_car_env import MVFS
from morl_baselines.common.evaluation import seed_everything


import time
from copy import deepcopy
import os
import pprint
import random
from typing import Dict
import imitation
import imitation.data
import imitation.data.rollout
import numpy as np

from baraacuda.agents.mo_agents import MO_DQN
from defines import transform_weights_to_tuple
from parsing import parse_args, parse_policy_approximator, parse_vectorizer_from_json
from src.dataset_processing.utils import calculate_expert_policy_save_path
from envs.firefighters_env_mo import FeatureSelectionFFEnv
from src.dataset_processing.datasets import create_dataset
from src.dataset_processing.preferences import save_preferences
from src.dataset_processing.preferences import load_preferences
from src.dataset_processing.trajectories import compare_trajectories, load_trajectories
from src.dataset_processing.trajectories import save_trajectories
from src.policies.mushroom_agent_mobaselines import MOBaselinesAgent, ProxyPolicy, obtain_trajectories_and_eval_mo
from mushroom_rl.core import Core
from src.utils import filter_none_args, load_json_config, sample_example_profiles
import gymnasium as gym


#from use_cases.roadworld_env_use_case.network_env import FeaturePreprocess, FeatureSelection
from utils import dataset_to_trajectories, evaluate, train, visualize_pareto_front



from mo_gymnasium.wrappers.wrappers import MORecordEpisodeStatistics




if __name__ == "__main__":
    # This script will generate a total of n_agents * trajectory_pairs of trajectories, and a chain of comparisons between them, per agent type, for the society selected
    # IMPORTANT: Default Args are specified depending on the environment in config.json
    parser_args = filter_none_args(parse_args_generate_dataset())
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

    agent_profiles = [transform_weights_to_tuple(ag['value_system'])
                      for ag in society_data['agents']]

    expert_policy_class = alg_config['expert_policy_class']
    if hasattr(parser_args, 'exp_policy') and parser_args.exp_policy is not None:
        expert_policy_class = parser_args.exp_policy
    expert_policy_kwargs: Dict = alg_config['expert_policy_kwargs'][expert_policy_class]

    eval_creator, train_creator = create_environments(
        make_env, parser_args, environment_data, alg_config, expert_policy_kwargs)

    train_environment, train_environment_mush = train_creator()
    eval_environment, eval_environment_mush = eval_creator()

    epclass, epkwargs_no_train_kwargs, train_kwargs, is_single_objective = parse_expert_policy_parameters(
        parser_args, environment_data, society_data, train_environment, train_environment_mush, expert_policy_kwargs, expert_policy_class)

    mobaselines_agent = MOBaselinesAgent(env=train_environment, eval_env=eval_environment,
                                         agent_class=epclass, agent_kwargs=epkwargs_no_train_kwargs, mdp_info=eval_environment_mush.info,
                                         train_kwargs=train_kwargs, is_single_objective=is_single_objective, weights_to_sample=None,
                                         name='exp'+expert_policy_class)

    eval_core = Core(agent=mobaselines_agent, mdp=eval_environment_mush,
                     agent_info_keys=epkwargs_no_train_kwargs.get('agent_info_keys', None))
    train_core = Core(agent=mobaselines_agent, mdp=train_environment_mush,
                      agent_info_keys=epkwargs_no_train_kwargs.get('agent_info_keys', None))
    t = time.time()

    dataset, dataset_info = eval_core.evaluate(
        n_episodes=100, render=False, quiet=False, get_env_info=True, get_action_info=False)
    trajs_pure, returns, v_returns, _, _ = dataset_to_trajectories(
        dataset, eval_weights=mobaselines_agent.get_weights(), dataset_info=dataset_info)
    nt = time.time() - t
    print("RANDOM Evaluation time: ", nt)
    print("Trajectories obtained: ", len(trajs_pure))
    print("Returns: ", np.mean(returns))
    print("Value Returns: ", np.mean(v_returns, axis=0))

    quiet_tqdm = False  # TODO
    train_time = 0.0
    eval_time = 0.0
    epoch = 0

    # TODO...
    # exit()
    # Evaluate
    # TODO: This should be in the algorithm config.

    # This is not used normally...
    run_dir = os.path.join("results", parser_args.environment, "datasets", dataset_name)  # directory for results

    os.makedirs(run_dir, exist_ok=True)
    print(f"Run Name: {run_dir}")

    eval_weights = [1.0]
    for _ in range(environment_data['n_values']-1):
        eval_weights.append(0.0)
    mobaselines_agent: MOBaselinesAgent
    mobaselines_agent.set_weights(eval_weights)
    print("Eval with", mobaselines_agent.get_weights())

    """print("OBSERVATION EXAMPLE", eval_environment_mush.reset()[0], eval_environment_mush.info.observation_space)
    exit(0)"""
    seed_everything(parser_args.seed)

    eval_environment.action_space.seed(parser_args.seed)
    eval_environment.observation_space.seed(parser_args.seed)

    dataset, dataset_info, eval_time = evaluate(core=eval_core,
                                                mdp=eval_environment_mush,
                                                policy=mobaselines_agent.policy,
                                                epsilon_eval=train_kwargs.get(
                                                    'epsilon_config', {'epsilon_eval': 0.0}).get('epsilon_eval', 0.0),
                                                epoch=epoch,
                                                horizon=environment_data['horizon'],
                                                quiet_tqdm=quiet_tqdm,
                                                eval_time=eval_time,
                                                record_video=False,
                                                run_dir=run_dir,
                                                **train_kwargs)

    trajs_pure, returns, v_returns, _, _ = dataset_to_trajectories(
        dataset, eval_weights=mobaselines_agent.get_weights(), dataset_info=dataset_info)
    print("Evaluation time BEFORE: ", eval_time)
    print("Trajectories obtained BEFORE: ", len(trajs_pure))
    print("Returns: ", np.mean(returns))
    print("Value Returns: ", np.mean(v_returns, axis=0))
    # TODO perhaps change eval env.
    mobaselines_agent.set_envs(train_environment, eval_environment)
    seed_everything(parser_args.seed)

    epsilon_config = train_kwargs.get('epsilon_config', None)
    train_kwargs_restarted = deepcopy(train_kwargs)
    if epsilon_config is not None:
        epsilon_config = parse_epsilon_params_from_json(epsilon_config)
    else:
        epsilon_config = {
            "epsilon_threshold": 0.0,
            "epsilon_init": 1.0,
            "epsilon_decay_steps": 1000,
            "epsilon_train": 0.1,
            "epsilon_eval": 0.0
        }
    train_kwargs_restarted['epsilon_config'] = epsilon_config

    if parser_args.retrain and not mobaselines_agent.is_single_objective:
        MO_DQN

        train_time = train(core=train_core,
                           agent_morl=mobaselines_agent,
                           epsilon_train=train_kwargs_restarted.get('epsilon_config', {'epsilon_train': 0.0}).get(
                               'epsilon_train', 0.0),  # purely random policy
                           epsilon_eval=train_kwargs_restarted.get(
                               'epsilon_config', {'epsilon_eval': 0.0}).get('epsilon_eval', 0.0),
                           quiet_tqdm=quiet_tqdm,
                           train_time=train_time,
                           horizon=environment_data['horizon'],
                           run_dir=run_dir,
                           **train_kwargs_restarted)

        mobaselines_agent.save(full_save=True, path=calculate_expert_policy_save_path(
            environment_name=parser_args.environment,
            dataset_name=parser_args.dataset_name,
            society_name=parser_args.society_name,
            class_name=mobaselines_agent.policy.__class__.__name__,
            grounding_name='default'))

        mobaselines_agent2: MOBaselinesAgent = MOBaselinesAgent.load(env=train_environment, eval_env=eval_environment,
                                                                     path=calculate_expert_policy_save_path(
                                                                         environment_name=parser_args.environment,
                                                                         dataset_name=parser_args.dataset_name,
                                                                         society_name=parser_args.society_name,
                                                                         class_name=mobaselines_agent.policy.__class__.__name__,
                                                                         grounding_name='default'), name='exp'+expert_policy_class,)

        assert ProxyPolicy.compare_parameters(
            mobaselines_agent.policy, mobaselines_agent2.policy), f"Parameters of the loaded policy do not match the original policy"

    try:
        mobaselines_agent2: MOBaselinesAgent = MOBaselinesAgent.load(env=train_environment, eval_env=eval_environment,
                                                                     path=calculate_expert_policy_save_path(
                                                                         environment_name=parser_args.environment,
                                                                         dataset_name=parser_args.dataset_name,
                                                                         society_name=parser_args.society_name,
                                                                         class_name=mobaselines_agent.policy.__class__.__name__,
                                                                         grounding_name='default'), name='exp'+expert_policy_class,)
        print("GETTING FROM: ", calculate_expert_policy_save_path(
                                                                         environment_name=parser_args.environment,
                                                                         dataset_name=parser_args.dataset_name,
                                                                         society_name=parser_args.society_name,
                                                                         class_name=mobaselines_agent.policy.__class__.__name__,
                                                                         grounding_name='default'))
        
    except FileNotFoundError as e:
        print(f"Expert policy not found")
        if (not parser_args.retrain) or parser_args.refine:
            raise e
        mobaselines_agent2 = mobaselines_agent

    if parser_args.retrain and not mobaselines_agent.is_single_objective:
        assert mobaselines_agent2.policy.__class__ == mobaselines_agent.policy.__class__
        print("Mobaselines agent policy class: ",
              mobaselines_agent.policy.__class__)

        assert ProxyPolicy.compare_parameters(
            mobaselines_agent.policy, mobaselines_agent2.policy), f"Parameters of the loaded policy do not match the original policy"

        mobaselines_agent.set_weights(eval_weights)

        eval_core = Core(agent=mobaselines_agent, mdp=eval_environment_mush,
                         agent_info_keys=epkwargs_no_train_kwargs.get('agent_info_keys', None))

        dataset, dataset_info, eval_time = evaluate(core=eval_core,
                                                    mdp=eval_environment_mush,
                                                    policy=mobaselines_agent.policy,
                                                    epsilon_eval=train_kwargs_restarted.get(
                                                        'epsilon_config', {'epsilon_eval': 0.0}).get('epsilon_eval', 0.0),
                                                    epoch=epoch,
                                                    horizon=environment_data['horizon'],
                                                    quiet_tqdm=quiet_tqdm,
                                                    eval_time=eval_time,
                                                    record_video=False,
                                                    run_dir=run_dir,
                                                    **train_kwargs_restarted)

        trajs_pure, returns, v_returns, _, _ = dataset_to_trajectories(
            dataset, eval_weights=mobaselines_agent.get_weights(), dataset_info=dataset_info, agent_name="test")
        # train_ = mobaselines_agent.agent_mobaselines.train(eval_env=train_environment_mush.env, weight=mobaselines_agent.get_weights(), **train_kwargs_restarted)

        trajs_pure2, (scalarized_return,
                      scalarized_discounted_return,
                      vec_return,
                      disc_vec_return) = obtain_trajectories_and_eval_mo(n_seeds=100, agent=mobaselines_agent, env=eval_environment_mush.env, ws=[mobaselines_agent.get_weights()], ws_eval=[mobaselines_agent.get_weights()], seed=parser_args.seed)
        
        print("EVAL WEIGHTS??:", mobaselines_agent.get_weights())
        print("Evaluation time AFTER (NOT SAVED): ", eval_time)
        print("Trajectories obtained AFTER (NOT SAVED): ", len(trajs_pure))
        print("Returns: ", np.mean(returns))
        print("Value Returns: ", np.mean(v_returns, axis=0))
        print("PSEUDE EVAL: ", scalarized_return,
              scalarized_discounted_return,
              vec_return,
              disc_vec_return)

    if not parser_args.retrain or parser_args.refine:
        mobaselines_agent: MOBaselinesAgent = mobaselines_agent2

    # TODO : The save and load is not ok...
    eval_core = Core(agent=mobaselines_agent, mdp=eval_environment_mush,
                     agent_info_keys=epkwargs_no_train_kwargs.get('agent_info_keys', None))
    seed_everything(parser_args.seed)
    mobaselines_agent.set_weights(eval_weights)
    dataset, dataset_info, eval_time = evaluate(core=eval_core,
                                                mdp=eval_environment_mush,
                                                policy=mobaselines_agent.policy,
                                                epsilon_eval=train_kwargs_restarted.get(
                                                    'epsilon_config', {'epsilon_eval': 0.0}).get('epsilon_eval', 0.0),
                                                epoch=epoch,
                                                horizon=environment_data['horizon'],
                                                quiet_tqdm=quiet_tqdm,
                                                eval_time=eval_time,
                                                record_video=False,
                                                run_dir=run_dir,
                                                **train_kwargs_restarted)

    trajs_pure, returns, v_returns, _, _ = dataset_to_trajectories(
        dataset, eval_weights=mobaselines_agent.get_weights(), dataset_info=dataset_info, agent_name="test")
    print("Evaluation time AFTER SAVED: ", eval_time)
    print("Trajectories obtained AFTER SAVED: ", len(trajs_pure))
    print("Returns: ", np.mean(returns))
    print("Value Returns: ", np.mean(v_returns, axis=0))
    # exit(0)
    """if not is_single_objective:
        policy.train(**epkwargs['train_kwargs'])"""
    prev_weights = np.zeros(len(agent_profiles[0]), dtype=np.float32)

    if (not parser_args.retrain) and (not parser_args.refine):
        pareto_front_and_weights, unfiltered_front_and_weights = mobaselines_agent.pareto_front(num_eval_weights_for_front=train_kwargs_restarted['num_eval_weights_for_front'],
                                                                                                num_eval_episodes_for_front=train_kwargs[
                                                                                                    'num_eval_episodes_for_front'],
                                                                                                discount=alg_config['discount_factor'])
        
        print("Solution set:", unfiltered_front_and_weights)
        print("LEARNED PARETO FRONT:", pareto_front_and_weights)
        print("REAL PARETO FRONT:", train_kwargs_restarted.get(
            'known_pareto_front', None))
        visualize_pareto_front(title="Expert Pareto Front Comparison",
                               learned_front_data=pareto_front_and_weights,
                               objective_names=environment_data['values_names'],
                               known_front_data=train_kwargs_restarted.get(
                                   'known_pareto_front', None),
                               save_path=os.path.join(run_dir, f'expert_pareto_front'), show=False, fontsize=parser_args.fontsize)
        visualize_pareto_front(title="Expert Solutions",
                               objective_names=environment_data['values_names'],
                               learned_front_data=unfiltered_front_and_weights,
                               
                               known_front_data=train_kwargs_restarted.get(
                                   'known_pareto_front', None),
                               save_path=os.path.join(run_dir, f'expert_solutions'), show=False, fontsize=parser_args.fontsize)
        if parser_args.remain_with_pareto_optimal_agents:
            society_data['agents'] = []
            for iw, weight in enumerate(pareto_front_and_weights[1]):
                new_agent = {}
                new_agent['value_system'] = weight
                new_agent['name'] = f"A{iw}|{new_agent['value_system']}"
                new_agent['data'] = society_data['default_data']
                new_agent['n_agents'] = society_data['default_n_agents']
                society_data['agents'].append(new_agent)

    else:
        pareto_front_and_weights = None
        print("Training now.")

    if parser_args.gen_trajs:

        for iag, ag in enumerate(society_data['agents']):
            rundir_ag = run_dir + f'/{ag["name"]}'
            os.makedirs(rundir_ag, exist_ok=True)
            # remove evertinhg inside the directory
            for f in os.listdir(rundir_ag):
                file_path = os.path.join(rundir_ag, f)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

            n_rational_trajs = int(np.ceil(ag['n_agents']*ag['data']['trajectory_pairs']*(1-ag['data']['random_traj_proportion']))
                                   ) if not society_data['same_trajectories_for_each_agent_type'] else int(np.ceil(ag['data']['trajectory_pairs']*(1-ag['data']['random_traj_proportion'])))
            n_random_trajs = int(np.floor(ag['n_agents']*ag['data']['trajectory_pairs']*ag['data']['random_traj_proportion'])
                                 ) if not society_data['same_trajectories_for_each_agent_type'] else int(np.ceil(ag['data']['trajectory_pairs']*(ag['data']['random_traj_proportion'])))
            # rationality is epsilon rationality.
            w = ag['value_system']
            seeds = np.array(list(range(n_rational_trajs)) +
                             list(range(n_random_trajs)))
            seeds[n_rational_trajs:] = seeds[n_rational_trajs:] + max(seeds)

            if not np.allclose(w, prev_weights):
                seeds += max(seeds)*iag + 1
            prev_weights = w
            mobaselines_agent.set_weights(w)
            if is_single_objective and (parser_args.retrain or parser_args.refine):
                train_environment, train_environment_mush = train_creator()
                eval_environment, eval_environment_mush = eval_creator()

                mobaselines_agent.set_envs(train_environment, eval_environment)

                mobaselines_agent.set_weights(w)

                epsilon_config = train_kwargs.get('epsilon_config', None)
                train_kwargs_restarted = deepcopy(train_kwargs)
                if epsilon_config is not None:
                    epsilon_config = parse_epsilon_params_from_json(
                        epsilon_config)
                else:
                    epsilon_config = {
                        "epsilon_threshold": 0.0,
                        "epsilon_init": 0.0,
                        "epsilon_decay_steps": 1000,
                        "epsilon_train": 0.0,
                        "epsilon_eval": 0.0
                    }
                train_kwargs_restarted['epsilon_config'] = epsilon_config

                train_core = Core(agent=mobaselines_agent, mdp=train_environment_mush,
                                  agent_info_keys=epkwargs_no_train_kwargs.get('agent_info_keys', None))

                train_time = train(core=train_core,
                                   agent_morl=mobaselines_agent,
                                   epsilon_train=train_kwargs_restarted.get('epsilon_config', {'epsilon_train': 0.0}).get(
                                       'epsilon_train', 0.0),  # purely random policy
                                   epsilon_eval=train_kwargs_restarted.get(
                                       'epsilon_config', {'epsilon_eval': 0.0}).get('epsilon_eval', 0.0),
                                   horizon=environment_data['horizon'],
                                   quiet_tqdm=quiet_tqdm,
                                   train_time=train_time,
                                   run_dir=rundir_ag,
                                   **train_kwargs_restarted)
            if n_rational_trajs > 0:

                mobaselines_agent.set_envs(eval_environment, eval_environment)
                eval_core = Core(agent=mobaselines_agent, mdp=eval_environment_mush,
                                 agent_info_keys=epkwargs_no_train_kwargs.get('agent_info_keys', None))
                trajs_pure2, (scalarized_return,
                              scalarized_discounted_return,
                              vec_return,
                              disc_vec_return) = obtain_trajectories_and_eval_mo(n_seeds=100,
                                                exploration=1.0-
                                                            ag['data']['rationality'],
                                                 agent=mobaselines_agent, env=eval_environment_mush.env, ws=[mobaselines_agent.get_weights()], ws_eval=[mobaselines_agent.get_weights()], seed=parser_args.seed)

                print("PSEUDE EVAL: ", scalarized_return,
                      scalarized_discounted_return,
                      vec_return,
                      disc_vec_return)

                dataset, dataset_info, eval_time = evaluate(core=eval_core,
                                                            mdp=eval_environment_mush,
                                                            policy=mobaselines_agent.policy,
                                                            n_eval_episodes_per_epoch=None,
                                                            epsilon_eval=1.0 -
                                                            ag['data']['rationality'],
                                                            epoch=epoch,
                                                            horizon=environment_data['horizon'],
                                                            quiet_tqdm=quiet_tqdm,
                                                            eval_time=eval_time,
                                                            record_video=False,
                                                            run_dir=rundir_ag,
                                                            seeds=seeds[:n_rational_trajs],
                                                            initial_states=train_kwargs_restarted.get(
                                                                'initial_states', None),
                                                            get_action_info=train_kwargs_restarted.get('get_action_info', False))
                ag_rational_trajs, returns, v_returns, _, _ = dataset_to_trajectories(
                    dataset, eval_weights=w, dataset_info=dataset_info, agent_name=ag['name'])
                print(
                    f"Rational Trajs Done {iag} ({len(ag_rational_trajs)}) Weigth: {w}")
                print("Returns: ", np.mean(returns))
                print("Value Returns: ", np.mean(v_returns, axis=0))
                #exit(0)


            else:
                ag_rational_trajs = []

            # random policies are equivalent to having rationality 0:
            if n_random_trajs > 0:
                dataset, dataset_info, eval_time = evaluate(core=eval_core,
                                                            mdp=eval_environment_mush,
                                                            policy=mobaselines_agent.policy,
                                                            n_eval_episodes_per_epoch=None,
                                                            epsilon_eval=1.0,
                                                            epoch=epoch,
                                                            horizon=environment_data['horizon'],
                                                            quiet_tqdm=quiet_tqdm,
                                                            eval_time=eval_time,
                                                            record_video=False,
                                                            run_dir=rundir_ag,
                                                            seeds=seeds[n_rational_trajs:],
                                                            initial_states=train_kwargs_restarted.get(
                                                                'initial_states', None),
                                                            get_action_info=train_kwargs_restarted.get('get_action_info', False))
                print("DAT", len(dataset), len(seeds))
                ag_random_trajs, returns, v_returns, _, _ = dataset_to_trajectories(
                    dataset, eval_weights=mobaselines_agent.get_weights(), dataset_info=dataset_info, agent_name=ag['name'])

                print(f"Random Trajs Done {iag} ({len(ag_random_trajs)})")
                print("Returns: ", np.mean(returns))
                print("Value Returns: ", np.mean(v_returns, axis=0))
            else:
                ag_random_trajs = []

            all_trajs_ag = []
            all_trajs_ag.extend(ag_random_trajs)
            all_trajs_ag.extend(ag_rational_trajs)

            random.shuffle(all_trajs_ag)

            print(f"All Trajs Done {iag} ({len(all_trajs_ag)})", ag)

            if parser_args.test_size > 0.0:

                all_trajs_ag_train = all_trajs_ag[:int(
                    len(all_trajs_ag)*(1-parser_args.test_size))]
                all_trajs_ag_test = all_trajs_ag[int(
                    len(all_trajs_ag)*(1-parser_args.test_size)):]
                if society_data['same_trajectories_for_each_agent_type']:
                    for _ in range(ag['n_agents']-1):
                        all_trajs_ag_train.extend(
                            all_trajs_ag[:int(np.ceil(len(all_trajs_ag)*(1-parser_args.test_size)))])
                        t_insert = all_trajs_ag[int(np.ceil(
                            len(all_trajs_ag)*(1-parser_args.test_size))):]
                        assert len(t_insert) == np.ceil(parser_args.test_size*len(all_trajs_ag)
                                                        ), f"Expected {int(parser_args.test_size*len(all_trajs_ag))} test trajectories, got {len(t_insert)}"
                        all_trajs_ag_test.extend(t_insert)
                        print("INDEX REF??", int(
                            len(all_trajs_ag)*(1-parser_args.test_size)))
                        print("NEXT INDEX", int(
                            np.ceil(parser_args.test_size*len(all_trajs_ag))))
                        print("N_trajs??", len(all_trajs_ag))
                        print("NAGENTS?", ag['n_agents'])
                    if ag['n_agents'] > 1:
                        np.testing.assert_allclose(all_trajs_ag_train[0].obs, all_trajs_ag_train[int(
                        np.ceil((1-parser_args.test_size)*len(all_trajs_ag)))].obs)
                save_trajectories(all_trajs_ag_train, dataset_name=dataset_name+'_train', ag=ag,
                                  society_data=society_data, environment_data=environment_data, dtype=parser_args.dtype)

                save_trajectories(all_trajs_ag_test, dataset_name=dataset_name+'_test', ag=ag,
                                  society_data=society_data, environment_data=environment_data, dtype=parser_args.dtype)
                trajs_test = load_trajectories(dataset_name=dataset_name+'_test', ag=ag,
                                               society_data=society_data, environment_data=environment_data, override_dtype=parser_args.dtype)
                np.testing.assert_equal(
                    trajs_test[0].obs, all_trajs_ag_test[0].obs)
                np.testing.assert_equal(
                    trajs_test[-1].obs, all_trajs_ag_test[-1].obs)
                if society_data['same_trajectories_for_each_agent_type']:
                    print(len(trajs_test), len(all_trajs_ag_test),
                          len(trajs_test)//ag['n_agents'])
                    if ag['n_agents'] > 1:
                        np.testing.assert_allclose(
                            all_trajs_ag_test[0].obs, all_trajs_ag_test[len(trajs_test)//ag['n_agents']].obs)
                        np.testing.assert_allclose(
                            trajs_test[0].obs, trajs_test[len(trajs_test)//ag['n_agents']].obs)
            else:
                if society_data['same_trajectories_for_each_agent_type']:
                    for _ in range(ag['n_agents']):
                        all_trajs_ag.extend(ag_random_trajs)
                        all_trajs_ag.extend(ag_rational_trajs)

                save_trajectories(all_trajs_ag, dataset_name=dataset_name, ag=ag,
                                  society_data=society_data, environment_data=environment_data, dtype=parser_args.dtype)
    mobaselines_agent.save(full_save=True, path=calculate_expert_policy_save_path(
        environment_name=parser_args.environment,
        dataset_name=parser_args.dataset_name,
        society_name=parser_args.society_name,
        class_name=mobaselines_agent.policy.__class__.__name__,
        grounding_name='default'))
    mobaselines_agent2: MOBaselinesAgent = MOBaselinesAgent.load(env=train_environment, eval_env=eval_environment,
                                                                 path=calculate_expert_policy_save_path(
                                                                     environment_name=parser_args.environment,
                                                                     dataset_name=parser_args.dataset_name,
                                                                     society_name=parser_args.society_name,
                                                                     class_name=mobaselines_agent.policy.__class__.__name__,
                                                                     grounding_name='default'), name='exp'+expert_policy_class,)

    assert ProxyPolicy.compare_parameters(
        mobaselines_agent.policy, mobaselines_agent2.policy), f"Parameters of the loaded policy do not match the original policy"

    if is_single_objective:
        for w in agent_profiles:
            assert ProxyPolicy.compare_parameters(
                mobaselines_agent.weights_to_algos, mobaselines_agent2.weights_to_algos), f"Parameters of the loaded policy do not match the original policy"

    if parser_args.gen_preferences:
        discount_factor_preferences = alg_config['train_kwargs']['discount_factor_preferences']

        for iag, ag in enumerate(society_data['agents']):
            for suffix in ['_train', '_test'] if parser_args.test_size > 0.0 else ['']:
                all_trajs_ag = load_trajectories(
                    dataset_name=dataset_name+suffix, ag=ag, society_data=society_data, environment_data=environment_data, override_dtype=parser_args.dtype)

                print(
                    f"1AGENT {iag} ({ag['name']}) - {len(all_trajs_ag)} trajectories loaded", suffix)
                if society_data["same_trajectories_for_each_agent_type"]:
                    # This assumes the trajectories of each agent are the same, and then we will make each agent label the same pairs
                    idxs_unique = np.random.permutation(
                        len(all_trajs_ag)//ag['n_agents'])
                    idxs = []
                    for step in range(ag['n_agents']):
                        idxs.extend(list(idxs_unique + step *
                                    len(all_trajs_ag)//ag['n_agents']))
                    idxs = np.array(idxs, dtype=np.int64)
                    assert len(idxs) == len(all_trajs_ag)
                else:
                    # Indicates the order of comparison. idxs[0] with idxs[1], then idxs[1] with idxs[2], etc...
                    idxs = np.random.permutation(len(all_trajs_ag))
                discounted_sums = np.zeros_like(idxs, dtype=np.float64)

                discounted_sums_per_grounding = np.zeros(
                    (len(environment_data['basic_profiles']), discounted_sums.shape[0]), dtype=np.float64)
                for i in range((len(all_trajs_ag))):

                    discounted_sums[i] = imitation.data.rollout.discounted_sum(
                        all_trajs_ag[i].vs_rews, gamma=discount_factor_preferences)
                    for vi in range(discounted_sums_per_grounding.shape[0]):
                        discounted_sums_per_grounding[vi, i] = imitation.data.rollout.discounted_sum(
                            all_trajs_ag[i].v_rews[vi], gamma=discount_factor_preferences)
                # We save the comparison of 1 vs 2, 2 vs 3 in the order stablished in discounted_sums.
                print(
                    f"2AGENT {iag} ({ag['name']}) - {len(idxs)} idxs generated", len(discounted_sums), suffix)
                assert max(idxs) < len(all_trajs_ag)
                save_preferences(idxs=idxs, discounted_sums=discounted_sums, discounted_sums_per_grounding=discounted_sums_per_grounding,
                                 dataset_name=dataset_name+suffix, epsilon=parser_args.reward_epsilon, environment_data=environment_data, society_data=society_data, ag=ag)

    # TEST preferences load okey.
    print("TESTING DATA COHERENCE. It is safe to stop this program now...")
    for dataset_name_ in [dataset_name+'_train', dataset_name+'_test'] if parser_args.test_size > 0.0 else [dataset_name]:
        for iag, ag in enumerate(society_data['agents']):
            #  Here idxs is the list of trajectory PAIRS of indices from the trajectory list that are compared.
            idxs, discounted_sums, discounted_sums_per_grounding, preferences, preferences_per_grounding = load_preferences(
                epsilon=parser_args.reward_epsilon, dataset_name=dataset_name_, environment_data=environment_data, society_data=society_data, ag=ag, dtype=parser_args.dtype)

            trajs_ag = load_trajectories(dataset_name=dataset_name_, ag=ag,
                                         environment_data=environment_data, society_data=society_data, override_dtype=parser_args.dtype)
            if parser_args.test_size > 0.0:
                trajs_ag_all = load_trajectories(dataset_name=dataset_name+'_train', ag=ag,
                                                 environment_data=environment_data, society_data=society_data, override_dtype=parser_args.dtype)
                trajs_ag_all.extend(load_trajectories(dataset_name=dataset_name+'_test', ag=ag,
                                                      environment_data=environment_data, society_data=society_data, override_dtype=parser_args.dtype))
            else:
                trajs_ag_all = trajs_ag
            n_pairs_per_agent_prime = len(trajs_ag)//ag['n_agents']

            if society_data["same_trajectories_for_each_agent_type"]:
                for t in range(ag['n_agents']-2):
                    np.testing.assert_allclose(idxs[0:n_pairs_per_agent_prime], (idxs[(
                        t+1)*n_pairs_per_agent_prime:(t+2)*n_pairs_per_agent_prime] - n_pairs_per_agent_prime*(t+1)))
                    # print(len(trajs_ag))
                    # print(idxs[0:n_pairs_per_agent_prime])
                    # print(idxs[(t+1)*(n_pairs_per_agent_prime):(t+2)*(n_pairs_per_agent_prime)])
                    # print("A", ag['name'], t+1, t, trajs_ag[idxs[(t+1)*(n_pairs_per_agent_prime)][0]].obs, trajs_ag[idxs[0][0]].obs)

            for i in range((len(trajs_ag))):
                np.testing.assert_almost_equal(discounted_sums[i], imitation.data.rollout.discounted_sum(
                    trajs_ag[i].vs_rews, gamma=discount_factor_preferences), decimal=4)

            for idx, pr in zip(idxs, preferences):
                assert isinstance(discounted_sums, np.ndarray)
                assert isinstance(idx, np.ndarray)
                idx = [int(ix) for ix in idx]

                np.testing.assert_almost_equal(discounted_sums[idx[0]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                    trajs_ag[idx[0]].vs_rews, gamma=discount_factor_preferences)), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)
                np.testing.assert_almost_equal(discounted_sums[idx[1]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                    trajs_ag[idx[1]].vs_rews, gamma=discount_factor_preferences)), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)
                np.testing.assert_almost_equal(compare_trajectories(
                    discounted_sums[idx[0]], discounted_sums[idx[1]], epsilon=parser_args.reward_epsilon), pr, decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)
            for vi in range(len(environment_data['basic_profiles'])):
                for i in range((len(trajs_ag))):
                    np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, i], parser_args.dtype(imitation.data.rollout.discounted_sum(
                        trajs_ag[i].v_rews[vi], gamma=discount_factor_preferences)), decimal=4 if parser_args.dtype in [np.float32, np.float64] else 3)
                    
                for idx, pr in zip(idxs, preferences_per_grounding[:, vi]):
                    idx = [int(ix) for ix in idx]

                    np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, idx[0]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                        trajs_ag[idx[0]].v_rews[vi], gamma=discount_factor_preferences)), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)

                    np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, idx[1]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                        trajs_ag[idx[1]].v_rews[vi], gamma=discount_factor_preferences)), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)

                    np.testing.assert_almost_equal(compare_trajectories(
                        discounted_sums_per_grounding[vi, idx[0]], discounted_sums_per_grounding[vi, idx[1]], epsilon=parser_args.reward_epsilon), pr, decimal=4 if parser_args.dtype in [np.float32, np.float64] else 3)
    if parser_args.test_size > 0.0:
        print("Dataset generated correctly.")
        dataset_train = create_dataset(
            parser_args, config, society_data, train_or_test='train')
        print("TRAIN SIZE", len(dataset_train))
        dataset_test = create_dataset(
            parser_args, config, society_data, train_or_test='test')
        print("TEST SIZE", len(dataset_test))
    else:
        dataset = create_dataset(parser_args, config, society_data)
        print("Dataset generated correctly.")

        # print("Dataset size", len(dataset))
