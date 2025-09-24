import itertools
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import wandb
from baraacuda.utils.miscellaneous import pickle_obj
from baraacuda.utils.stats import get_stats
from baraacuda.utils.wrappers import start_recording, stop_recording
from mushroom_rl.core import Core
from defines import transform_weights_to_tuple
from src.algorithms.clustering_utils_simple import ClusterAssignment
from src.dataset_processing.data import TrajectoryWithValueSystemRews
from src.policies.mushroom_agent_mobaselines import MOBaselinesAgent
import json


def select_output_shape(regressor_type, n_actions, n_objectives):
    if n_objectives is None:
         mo = False
    else:
         mo = True
    
    if regressor_type == "QRegressor":
        output_shape = (n_actions, n_objectives) if mo else (n_actions,)
    elif regressor_type == "ActionRegressor":
        output_shape = (1, n_objectives) if mo else (1,)
    else:
        raise ValueError(f"{regressor_type=} must be 'QRegressor' or 'ActionRegressor'")
    return output_shape

def train(core: Core,
          agent_morl: MOBaselinesAgent,
          epsilon_train,
          
          n_train_steps_per_epoch,
          n_train_steps_per_fit,
          quiet_tqdm,
          train_time,
          horizon=None,
          run_dir=None,
          epsilon_eval=None,
          **kwargs):
    #policy.policy.set_epsilon(0.0)
    checkpoint_time = time.time()
    if isinstance(agent_morl, MOBaselinesAgent):
        # This does not work. core.learn(n_steps=n_train_steps_per_epoch, n_steps_per_fit=n_train_steps_per_fit, quiet=quiet_tqdm)
        agent_morl.fit(None)
    else:
        # TODO: MO_DQN NOT USEd
        n_epochs = kwargs.get('n_epochs', 1)
        agent_morl.policy.set_epsilon(epsilon_train)

        epoch = 0
                
        train_time = 0
        eval_time = 0
        stats_time = 0
        times = []


        epochs_per_save = kwargs.get('epochs_per_save', 10)
        save_episode_returns = lambda: True
        save_dataset = lambda: epoch % epochs_per_save == 0
        save_dataset_info = lambda: epoch % epochs_per_save == 0
        save_agent = lambda: epoch % epochs_per_save == 0

        agent_morl.policy.set_epsilon(kwargs.get('epsilon_init', 1.0))
        checkpoint_time = time.time()
        core.learn(n_steps=agent_morl.agent_mobaselines.initial_replay_size-1, n_steps_per_fit=agent_morl.agent_mobaselines.initial_replay_size-1, quiet=quiet_tqdm)
        train_time += time.time() - checkpoint_time
        
#exit()
# Evaluate
        normalizer = agent_morl.env.get_wrapper_attr('normalizer')
        scalarizer = agent_morl.policy.scalarizer
        gamma = kwargs.get('gamma', None)
        if gamma is None:
            raise ValueError("gamma must be provided in kwargs")
        stats_params = {"scalarizer": scalarizer, "gamma": gamma}  # Include gamma if stats of discounted return are desired
        
        normalizer.track_stats = False
        dataset, dataset_info, eval_time = evaluate(core=core,
                                                        mdp=agent_morl.eval_env,
                                                        policy=agent_morl.policy,
                                                        epsilon_eval=epsilon_eval,
                                                        epoch=epoch,
                                                        n_eval_episodes_per_epoch=kwargs.get('n_eval_episodes_per_epoch', 50),
                                                        horizon=horizon,
                                                        quiet_tqdm=quiet_tqdm,
                                                        eval_time=eval_time,
                                                        record_video=False,
                                                        run_dir=run_dir,
                                                        initial_states=kwargs.get('initial_states', None),
                                                        seeds=kwargs.get('seeds', None),
                                                        get_action_info= kwargs.get('get_action_info', False))

        # Stats
        stats = {f"normalizer_mean[{i}]": x for i, x in enumerate(normalizer.mean)}
        stats.update({f"normalizer_std[{i}]": x for i, x in enumerate(normalizer.std)})
        save_normalizer_history(normalizer=normalizer, run_dir=run_dir, epoch=epoch, clear_history=True)
        print("EPOCH", epoch)
        stats_time = log_stats(dataset=dataset,
                            dataset_info=dataset_info,
                            agent=agent_morl,
                            epoch=epoch,
                            n_train_steps_per_epoch=n_train_steps_per_epoch,
                            train_time=train_time,
                            eval_time=eval_time,
                            stats_time=stats_time,
                            stats_params=stats_params,
                            run_dir=run_dir,
                            save_episode_returns=save_episode_returns(),
                            save_dataset=save_dataset(),
                            save_dataset_info=save_dataset_info(),
                            save_agent=save_agent(),
                            stats=stats)
        times.append({"Epoch": epoch, "Train_Time": train_time, "Eval_Time": eval_time, "Stats_Time": stats_time})


        for epoch in range(1, n_epochs + 1):
            core : Core
            agent_morl.policy.set_epsilon(epsilon_train)
            checkpoint_time = time.time()
            
            core.learn(n_steps=n_train_steps_per_epoch, n_steps_per_fit=n_train_steps_per_fit, quiet=quiet_tqdm)
            
            train_time += time.time() - checkpoint_time

            # Evaluate
            

            normalizer.track_stats = True
            dataset, dataset_info, eval_time = evaluate(core=core,
                                                        mdp=agent_morl.eval_env,
                                                        policy=agent_morl.policy,
                                                        epsilon_eval=epsilon_eval,
                                                        epoch=epoch,
                                                        n_eval_episodes_per_epoch=kwargs.get('n_eval_episodes_per_epoch', None),
                                                        horizon=horizon,
                                                        quiet_tqdm=quiet_tqdm,
                                                        eval_time=eval_time,
                                                        record_video=False,
                                                        run_dir=run_dir,
                                                        initial_states=kwargs.get('initial_states', None),
                                                        seeds=kwargs.get('seeds'),
                                                        get_action_info= kwargs.get('get_action_info', False))

            # Stats
            stats = {f"normalizer_mean[{i}]": x for i, x in enumerate(normalizer.mean)}
            stats.update({f"normalizer_std[{i}]": x for i, x in enumerate(normalizer.std)})
            save_normalizer_history(normalizer=normalizer, run_dir=run_dir, epoch=epoch, clear_history=True)
            stats_time = log_stats(dataset=dataset,
                                dataset_info=dataset_info,
                                agent=agent_morl,
                                epoch=epoch,
                                n_train_steps_per_epoch=n_train_steps_per_epoch,
                                train_time=train_time,
                                eval_time=eval_time,
                                stats_time=stats_time,
                                stats_params=stats_params,
                                run_dir=run_dir,
                                save_episode_returns=save_episode_returns(),
                                save_dataset=save_dataset(),
                                save_dataset_info=save_dataset_info(),
                                save_agent=save_agent(),
                                stats=stats)
            times.append({"Epoch": epoch, "Train_Time": train_time, "Eval_Time": eval_time, "Stats_Time": stats_time})
            print("EPSILON", epsilon_train.get_value())
            print("EPSILON", epsilon_train.get_value())
        
    train_time += time.time() - checkpoint_time
    return train_time

from morl_baselines.common.performance_indicators import (
    cardinality,
    expected_utility,
    hypervolume,
    igd,
    maximum_utility_loss,
)

def visualize_pareto_front(learned_front_data=None, known_front_data=None, weights_in_pareto_front={}, title="Pareto Front Visualization",
                          save_path=None, show_weights=True, objective_names=None, show=False, with_clusters: ClusterAssignment =None, cluster_colors=None, fontsize=12, calculate_metrics=False, ref_point=None, clusters_not_in_pf=None):
    """
    Visualize the Pareto front with learned and known fronts.
    
    Args:
        learned_front_data: Tuple of (points, weights) where:
            - points: array of shape (n_points, n_objectives) with Pareto points
            - weights: array of shape (n_points, n_objectives) with corresponding weights
        known_front_data: List or array of known Pareto points (no weights)
        objective_names: List of objective names for axis labels
        title: Plot title
        save_path: Path to save the plot (optional)
        show_weights: Whether to annotate points with weights
    """
    metrics = None
    metrics_cl = None
    if learned_front_data is not None:
        np.save(os.path.join(save_path + '_learned_front.npy'), np.array(learned_front_data))
    if known_front_data is not None:
        np.save(os.path.join(save_path + '_known_front.npy'), np.array(known_front_data))
    if with_clusters is not None:
        assert cluster_colors is not None
        assert len(cluster_colors) == with_clusters.Lmax, "Cluster colors must match the number of clusters."
        
    if calculate_metrics:
        assert learned_front_data is not None, "Learned front data must be provided for metric calculation."

        assert ref_point is not None, "Reference point must be provided for hypervolume calculation."
        if known_front_data is not None:
            igd_metric = igd(known_front=known_front_data, current_estimate=learned_front_data[0])
        hypervolume_metric = hypervolume(ref_point=ref_point, points=learned_front_data[0])
        cardinality_metric = cardinality(front=learned_front_data[0])
        expected_utility_metric = expected_utility(front=learned_front_data[0], weights_set={tuple(w) for w in learned_front_data[1]})
        maximum_utility_loss_metric = maximum_utility_loss(reference_set=known_front_data, front=learned_front_data[0], weights_set={tuple(w) for w in learned_front_data[1]})
        metrics = {
            "Hypervolume": hypervolume_metric,
            "Cardinality": cardinality_metric,
            "Expected Utility": expected_utility_metric,
            "Maximum Utility Loss": maximum_utility_loss_metric,
            "IGD": igd_metric if known_front_data is not None else 0.0
        }
        if save_path is not None:
            metrics_save_path = save_path + '_metrics.json'
            with open(metrics_save_path, 'w') as f:
                json.dump(metrics, f, indent=4)
    n_goals = learned_front_data[0].shape[-1]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d' if n_goals == 3 else None)

    assert n_goals <= 3, "Number of objectives must be 2 or 3 for visualization."
    # Default objective names
    if objective_names is None:
        objective_names = [f'Value {i}' for i in range(1, n_goals + 1)]
    assert len(objective_names) == n_goals, "At least two objectives are required for Pareto front visualization."
    cluster_color_array = []
    # Plot known Pareto front
    if known_front_data is not None:
        known_points = np.array(known_front_data)
        if known_points.ndim == 1:
            known_points = known_points.reshape(-1, 2)
        
        # Plot points
        if n_goals == 2:
            ax.scatter(known_points[:, 0], known_points[:, 1], 
                  c='black', s=160, alpha=1.0, 
                  label='Known Pareto Front', marker='s', edgecolors='darkred',zorder=1)
        elif n_goals == 3:
            ax.scatter3D(known_points[:, 0], known_points[:, 1], known_points[:, 2], 
                  c='black',  alpha=1.0, 
                  label='Known Pareto Front', marker='s', edgecolors='darkred',zorder=1)

    # Plot learned Pareto front   
    clusters_painted = list()
    cluster_points = list()
    cluster_colors_simple = list()
    if clusters_not_in_pf is not None:
        learned_front_data = list(learned_front_data)
        learned_front_data[0] = np.vstack([learned_front_data[0], clusters_not_in_pf[0]])
        learned_front_data[1] = np.vstack([learned_front_data[1], clusters_not_in_pf[1]])
        learned_front_data = tuple(learned_front_data)
    if learned_front_data is not None:
        learned_points, learned_weights = learned_front_data
        learned_points = np.array(learned_points)
        learned_weights = np.array(learned_weights)
        if cluster_colors is not None:
            for il, w in enumerate(learned_weights):
                
                closest_weight = None
                index_ = 0
                target_val2 = weights_in_pareto_front.get(transform_weights_to_tuple(w), None)
                closest_weights = []
                for icl, clw in enumerate(with_clusters.value_systems):
                    if clw not in with_clusters.value_systems_active:
                        continue           
                    target_val = weights_in_pareto_front.get(transform_weights_to_tuple(clw), None)
                    
                    if target_val is not None and target_val2 is not None and np.allclose(target_val, target_val2):
                        closest_weight = transform_weights_to_tuple(clw)
                        index_ = icl
                        closest_weights.append((closest_weight, icl))
                    
                if closest_weight is None and clusters_not_in_pf is not None and transform_weights_to_tuple(transform_weights_to_tuple(w)
                    ) in {tuple(t__) for t__ in clusters_not_in_pf[1].tolist()}:
                        closest_weight = transform_weights_to_tuple(w)
                        _, index_ = with_clusters.find_cluster_with_weights(w)
                        closest_weights.append((closest_weight, index_))
                if closest_weight is not None:
                    cluster_color = cluster_colors[index_] if cluster_colors is not None else (0,0,1)
                    
                    clusters_painted.append(closest_weights)
                    cluster_points.append(learned_points[il])
                    cluster_colors_simple.append(cluster_color)
                else:
                    cluster_color = (0.5,0.5,0.5)  # Gray for unassigned
                    # TODO: put weights of ALL clusters with points in the front.
                cluster_color_array.append(cluster_color)
            cluster_points = np.array(cluster_points)
            assert len(cluster_color_array) == len(learned_points), "Cluster colors must match the number of learned points."
            # Add legend for each cluster color
            for idx, color in enumerate(cluster_colors_simple):
                ax.scatter([cluster_points[idx, 0]], [cluster_points[idx, 1]], c=[color], label="\n".join([f'Cl. {cp[1]} - VS {tuple([f"{float(c):.3f}" for c in cp[0]])}' for cp in clusters_painted[idx] ]), marker='o', s=200)
            if calculate_metrics:
                assert learned_front_data is not None, "Learned clustered front data must be provided for metric calculation."

                assert ref_point is not None, "Reference point must be provided for hypervolume calculation."
                if known_front_data is not None:
                    igd_metric = igd(known_front=known_front_data, current_estimate=cluster_points)
                hypervolume_metric = hypervolume(ref_point=ref_point, points=cluster_points)
                cardinality_metric = cardinality(front=cluster_points)
                expected_utility_metric = expected_utility(front=cluster_points, weights_set={tuple(w) for w in learned_front_data[1]})
                maximum_utility_loss_metric = maximum_utility_loss(reference_set=known_front_data, front=cluster_points, weights_set={tuple(w) for w in learned_front_data[1]})
                metrics_cl = {
                    "Hypervolume": hypervolume_metric,
                    "Cardinality": cardinality_metric,
                    "Expected Utility": expected_utility_metric,
                    "Maximum Utility Loss": maximum_utility_loss_metric,
                    "IGD": igd_metric if known_front_data is not None else 0.0
                }
                if save_path is not None:
                    metrics_save_path = save_path + '_clusters_metrics.json'
                    with open(metrics_save_path, 'w') as f:
                        json.dump(metrics_cl, f, indent=4)
        else:
            cluster_color_array = ['yellow'] * len(learned_points)
        if n_goals == 2:
            # Plot learned points over known points by plotting after known front
            print(cluster_color_array)
            ax.scatter(learned_points[:, 0], learned_points[:, 1], 
                               c='white', s=90, alpha=1.0, 
                               label='Learned Pareto Front', marker='o', edgecolors='darkblue', zorder=3000)
            if len(clusters_painted) > 0:
                # Add legend for clusters
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=cluster_colors_simple, alpha=0, edgecolors=cluster_colors_simple,s=200, label=None, marker='o', zorder=3000)
               
        elif n_goals == 3:
            assert len(cluster_color_array) == len(learned_points), "Cluster colors must match the number of learned points."
           
            ax.scatter3D(learned_points[:, 0], learned_points[:, 1], learned_points[:, 2], 
                  c='white', alpha=0.8 ,
                  label='Learned Pareto Front', marker='o', edgecolors='darkblue', zorder=3000)
            if len(clusters_painted) > 0:
                # Add legend for clusters
                ax.scatter3D(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=cluster_colors_simple, alpha=0, edgecolors=cluster_colors_simple,s=200, label='Cluster policies in Pareto Front', marker='o', zorder=3001)

        # Annotate with weights if requested
        if show_weights and learned_weights is not None:
            for i, (point, weight) in enumerate(zip(learned_points, learned_weights)):
                weight_str = "("+(', ').join([f"{weight[i]:.2f}" for i in range(n_goals)]) + ")"
                if n_goals == 2:
                    ax.annotate(weight_str, (point[0], point[1]), 
                               xytext=(-7, 8), textcoords='offset points',
                               fontsize=fontsize-1, alpha=0.8, color='blue')
                elif n_goals == 3:
                    ax.text(point[0], point[1], point[2], weight_str, fontsize=fontsize-3, alpha=0.8, color='blue')

    
            # Note: 3D plotting requires a different setup, this is a placeholder for 3D visualization.
    # Formatting
    ax.set_xlabel(objective_names[0], fontsize=fontsize+1)
    ax.set_ylabel(objective_names[1], fontsize=fontsize+1)
    ax.tick_params(axis='both', labelsize=fontsize)  # Set fontsize for tick labels
    if n_goals == 3:
        ax.set_zlabel(objective_names[2], fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize+3, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fontsize-1)

    # Add some styling
    if n_goals == 2:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        print("Saving pareto plot:", save_path+'.pdf')
        plt.savefig(save_path+'.pdf', dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    plt.close(fig)

    # If n_goals >= 3: plot the projection of the pareto front into each pair of goals.
    if n_goals >= 3:
        # Projections: (0,1), (0,2), (1,2)
        proj_names = list(itertools.combinations(range(n_goals), 2))
        
        for i, (a, b) in enumerate(proj_names):
            
            fig_proj, ax_proj = plt.subplots(figsize=(7, 6))
            # Plot known front projection
            if known_front_data is not None:
                other_goal = list(set(range(n_goals)) - {a, b})[0]
                min_size, max_size = 50, 300
                values = known_points[:, other_goal]
                # Normalize values to [0, 1]
                norm_values = (values - values.min()) / (values.max() - values.min() + 1e-8)
                sizes = min_size + norm_values * (max_size - min_size)

                ax_proj.scatter(known_points[:, a], known_points[:, b], 
                                c='black', s=sizes, alpha=1.0, 
                                label='Known Pareto Front', marker='s', edgecolors='darkred', zorder=1)

            # Plot learned front projection
            if learned_front_data is not None:
                # Size is proportional to the value of the third goal (not in projection)
                other_goal = list(set(range(n_goals)) - {a, b})[0]
                min_size, max_size = 50, 300
                values = learned_points[:, other_goal]
                # Normalize values to [0, 1]
                norm_values = (values - values.min()) / (values.max() - values.min() + 1e-8)
                sizes = min_size + norm_values * (max_size - min_size)
                ax_proj.scatter(learned_points[:, a], learned_points[:, b],
                                c='white', s=sizes, alpha=0.8,
                                label='Learned Pareto Front', marker='o', edgecolors='darkblue', zorder=3000)
                if show_weights and learned_weights is not None:
                    for point, weight in zip(learned_points, learned_weights):
                        weight_str = "("+(', ').join([f"{weight[j]:.2f}" for j in range(n_goals)]) + ")"
                        ax_proj.annotate(weight_str, (point[a], point[b]), 
                                         xytext=(-7, 8), textcoords='offset points',
                                         fontsize=fontsize-4, alpha=0.8, color='blue')

            ax_proj.set_xlabel(objective_names[a], fontsize=fontsize)
            ax_proj.set_ylabel(objective_names[b], fontsize=fontsize)
            ax_proj.set_title(f"{title} (Projection: {objective_names[a]} vs {objective_names[b]})", fontsize=fontsize+2)
            ax_proj.grid(True, alpha=0.3)
            ax_proj.legend(fontsize=fontsize-1)
            ax.tick_params(axis='both', labelsize=fontsize)  # Set fontsize for tick labels
    
            ax_proj.spines['top'].set_visible(False)
            ax_proj.spines['right'].set_visible(False)
            plt.tight_layout()
            if save_path is not None:
                proj_save_path = save_path +  f'-proj{a}{b}.pdf'
                print("Saving projection plot:", proj_save_path)
                plt.savefig(proj_save_path, dpi=300, bbox_inches='tight')
            if show:
                plt.show()
            plt.close(fig_proj)

    if metrics_cl is not None:
        return metrics, metrics_cl
    else:
        return metrics
def evaluate(core: Core,
             mdp,
             policy,
             epsilon_eval,
             epoch,
             n_eval_episodes_per_epoch,
             horizon,
             quiet_tqdm,
             eval_time,
             record_video,
             run_dir,
             initial_states=None,
             seeds=None,
             get_action_info=False, **kwargs):

    policy.set_epsilon(epsilon_eval)
    if record_video:
        start_recording(mdp=mdp,
                        video_folder=run_dir,
                        episode_trigger=lambda n: n == 0,
                        step_trigger=None,
                        video_length=n_eval_episodes_per_epoch * horizon,  # All episodes in a single video
                        name_prefix=f"epoch{epoch:03}-video",
                        fps=None,
                        disable_logger=True,
                        version="100a2")  # 100a2 allows multiple episodes per video file
    checkpoint_time = time.time()
    print("EVALUATING", len(seeds), n_eval_episodes_per_epoch)
    dataset, dataset_info = core.evaluate(initial_states=initial_states,
                                          n_episodes=n_eval_episodes_per_epoch,
                                          quiet=quiet_tqdm,
                                          get_env_info=True,
                                          seeds=seeds,
                                          get_action_info=get_action_info)
    print("EVALUATION DONE")
    eval_time += time.time() - checkpoint_time
    if record_video:
        stop_recording(mdp)
    core.reset()
    return dataset, dataset_info, eval_time

def dataset_to_trajectories(dataset, eval_weights, dataset_info=None, agent_name='unk', collect_real_reward=False):
    
    episode_rewards = []
    r_episode_rewards = []
    trajectories = []
    ep_rewards = []
    r_eprewards = []
    obs = []
    acts = []
    dones = []
    infos = []
    rews =[]
    real_vrews = []
    real_rews = []
    v_rews = []

    eval_weights = np.asarray(eval_weights, dtype=np.float32)
    n_vals = eval_weights.shape[0]

    for i, sample in enumerate(dataset):
        state, action, rewards, next_state, absorbing, last = sample
        rewards = np.asarray(rewards, dtype=np.float32)
        assert isinstance(rewards, np.ndarray), "Rewards must be a numpy array"
        assert len(rewards.shape) == 1, "Rewards must be a 1D array"
        assert rewards.shape[0] == n_vals, f"Rewards shape {rewards.shape} does not match eval_weights shape {eval_weights.shape}"
        obs.append(state)
        acts.append(action[0] if isinstance(action, (list, tuple)) else action)
        v_rews.append(rewards)
        dones.append(absorbing)
        rews.append(rewards.dot(eval_weights))
        info = {}
        if collect_real_reward:
            assert dataset_info is not None, "dataset_info must be provided if collect_real_reward is True"
        if dataset_info is not None or collect_real_reward:
            #print(len(dataset_info))
            info = {k: v[min(i,len(v)-1)] if dataset_info is not None else None for k,v in dataset_info.items()}
            #print(info)
        infos.append(info)
        if collect_real_reward or (info.get('untransformed_reward', None) is not None):
            collect_real_reward = True
            real_vrews.append(np.array(info['untransformed_reward']))
            real_rews.append(np.array(info['untransformed_reward']).dot(eval_weights))
            r_eprewards.append(real_vrews)
            assert r_eprewards[-1].shape == (n_vals,), f"Episode real rewards shape {r_eprewards[-1].shape} does not match eval_weights shape {eval_weights.shape}"

        ep_rewards.append(rewards)
        
        assert ep_rewards[-1].shape == (n_vals,), f"Episode rewards shape {ep_rewards[-1].shape} does not match eval_weights shape {eval_weights.shape}"
        
        if last:
            obs.append(next_state)
            
            trajectory = TrajectoryWithValueSystemRews(n_vals=n_vals, obs=np.asarray(obs), acts=np.asarray(acts), 
                                                       rews=np.asarray(rews), v_rews=np.asarray(v_rews).T, 
                                                         v_rews_real=np.asarray(real_vrews).T if collect_real_reward else None,
                                                         rews_real=np.asarray(real_rews) if collect_real_reward else None,
                                                         dones=np.asarray(dones, dtype=np.float32),
                                                       terminal=absorbing, infos=infos, agent=agent_name)
            
            episode_rewards.append(np.sum(ep_rewards, axis=0))
            if collect_real_reward:
                r_episode_rewards.append(np.sum(r_eprewards, axis=0))
                assert len(r_episode_rewards[-1]) == n_vals, f"Trajectory rewards shape {trajectory.value_rews_real.shape} does not match eval_weights shape {eval_weights.shape}"
                assert len(v_rews) == len(real_vrews) == len(real_rews), f"Length mismatch: {len(v_rews)}, {len(real_vrews)}, {len(real_rews)}"
            assert len(episode_rewards[-1]) == n_vals, f"Trajectory rewards shape {trajectory.value_rews.shape} does not match eval_weights shape {eval_weights.shape}"
            assert len(dones) == len(obs)-1 == len(acts) == len(rews) == len(v_rews) == len(infos), f"Length mismatch: {len(dones)}, {len(obs)-1}, {len(acts)}, {len(rews)}, {len(v_rews)}, {len(real_vrews)}, {len(real_rews)}, {len(infos)}"
            ep_rewards = []
            r_eprewards = []
            obs = []
            acts = []
            rews = []
            v_rews = []
            real_vrews = []
            real_rews = []
            dones = []
            infos=[]
            trajectories.append(trajectory)
    print("END")
    print(np.array(episode_rewards).shape, eval_weights.shape)

    if not collect_real_reward:
        real_reward = np.array(episode_rewards).dot(eval_weights)
        real_episode_rewards = episode_rewards
    else:
        real_reward = np.array(r_episode_rewards).dot(eval_weights)
        real_episode_rewards = r_episode_rewards
    
    return trajectories, np.array(episode_rewards).dot(eval_weights), episode_rewards, real_reward, real_episode_rewards
def log_stats(dataset,
              dataset_info,
              agent,
              epoch,
              n_train_steps_per_epoch,
              train_time,
              eval_time,
              stats_time,
              stats_params,
              run_dir,
              save_episode_returns,
              save_dataset,
              save_dataset_info,
              save_agent,
              stats=None):

        checkpoint_time = time.time()
        if stats is None:
            stats = {}
        stats.update({"train_steps": epoch * n_train_steps_per_epoch,
                      "epoch": epoch,
                      "train_time": train_time,
                      "eval_time": eval_time})
        stats, episode_returns, discounted_episode_returns = get_stats(dataset=dataset,
                                              stats=stats,
                                              quiet=False,
                                              round_digits=2,
                                              **stats_params)  # Include gamma if stats of discounted return are desired
        #stats = compute_LL_terminations(dataset_info=dataset_info, stats=stats, quiet=False)

        if save_episode_returns:
            pickle_obj(f"{run_dir}/epoch{epoch:03}-returns.pkl", episode_returns)
            if discounted_episode_returns is not None:
                pickle_obj(f"{run_dir}/epoch{epoch:03}-discounted_returns.pkl", episode_returns)
        if save_dataset:
            pickle_obj(f"{run_dir}/epoch{epoch:03}-dataset.pkl", dataset)
        if save_dataset_info:
            pickle_obj(f"{run_dir}/epoch{epoch:03}-dataset_info.pkl", dataset_info)
        if save_agent:
            agent.save(f"{run_dir}/epoch{epoch:03}-agent.msh")
        stats_time += time.time() - checkpoint_time
        stats["stats_time"] = stats_time
        print(f"stats_time: {round(stats_time, 2)}")
        wandb.log(stats)
        return stats_time

def get_wrapper(mdp, wrapper_name):
    env = mdp.env
    while env != env.unwrapped:
        if env.class_name() == wrapper_name:
            return env
        else:
            env = env.env
    raise ValueError(f"Wrapper '{wrapper_name}' not found in {mdp.env}")

def save_normalizer_history(normalizer, run_dir, epoch=None, clear_history=False):
    normalizer_history_x = np.array(normalizer.history_x)
    normalizer_history_mean = np.array(normalizer.history_mean)
    normalizer_history_std = np.array(normalizer.history_std)

    if clear_history:
        normalizer.history_counter = 0
        normalizer.history_x = []
        normalizer.history_mean = []
        normalizer.history_std = []
    if epoch is None:
        epoch_str = ""
    else:
        epoch_str = f"epoch{epoch:03}-"
    np.save(os.path.join(run_dir, (epoch_str + "normalizer_history_x.npy")), normalizer_history_x)
    np.save(os.path.join(run_dir, (epoch_str + "normalizer_history_mean.npy")), normalizer_history_mean)
    np.save(os.path.join(run_dir, (epoch_str + "normalizer_history_std.npy")), normalizer_history_std)



"""agent_kwargs_PCNFF = {
        "scaling_factor": np.array([1, 1, 1.0]),
        "learning_rate": 0.002,
        "batch_size": 64,
        "hidden_dim": 64,
        "project_name": "MORL-Baselines",
        "experiment_name": "PCNFF",
        "log": True,
    }

def main_minecart():
    def make_env():
        env = mo_gym.make("minecart-deterministic-v0")
        env = MORecordEpisodeStatistics(env, gamma=1.0)
        return env

    env = make_env()

    agent = PCN(
        env,
        scaling_factor=np.array([1, 1, 0.1, 0.1]),
        learning_rate=1e-3,
        batch_size=256,
        project_name="MORL-Baselines",
        experiment_name="PCN",
        log=True,
    )

    agent.train(
        eval_env=make_env(),
        total_timesteps=int(1e5),
        ref_point=np.array([0, 0, -200.0]),
        num_er_episodes=20,
        max_buffer_size=50,
        num_model_updates=50,
        max_return=np.array([1.5, 1.5, -0.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=1.0),
    )"""