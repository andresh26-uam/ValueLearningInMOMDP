
# %%
import argparse
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
import wandb
from baraacuda.agents.mo_agents import MO_DQN
from baraacuda.envs import lunar_lander, breakable_bottles, living_room
from baraacuda.networks.networks import Network_SOMO
from baraacuda.policies.mo_policies import ScalarizedEpsGreedy
from baraacuda.utils.normalizers import ExpWeighted_MeanStd_Normalizer
from baraacuda.utils.scalarizers import Linear_Scalarizer
from baraacuda.utils.wrappers import LL_Terminations, RewardAccumulator, AccumulatedRewardsStateAugmentor, TransformVectorReward
from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator
from mushroom_rl.core import Core
from mushroom_rl.environments import MO_Gymnasium
from mushroom_rl.utils.parameters import LinearParameter, Parameter
from gymnasium.wrappers import FlattenObservation
#from utils import select_output_shape
from utils import select_output_shape, train, evaluate, log_stats, save_normalizer_history
import sys
config = {}

# Parse Slurm args
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", help="path to directory where results should be stored", default=".")
parser.add_argument("--tmp_dir", help="path to directory where temporary files should be stored")
parser.add_argument("--slurm_array_task_id", type=int, help="$SLURM_ARRAY_TASK_ID")
parser.add_argument("--slurm_id", help=r"'$SLURM_JOB_ID' for individual jobs or '${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}' for job arrays")
parser.add_argument("--slurm_job_partition", help="$SLURM_JOB_PARTITION")
parser.add_argument("--slurm_cpus_per_task", help="$SLURM_CPUS_PER_TASK")
parser.add_argument("--slurm_mem_per_cpu", help="$SLURM_MEM_PER_CPU")
parser.add_argument("--slurm_tmp_mem", help="Local storage specified in sbatch --tmp")
parser.add_argument("--slurm_time", help="Allocated maximum wallclock time")
args = parser.parse_args()

config.update({"slurm_id": args.slurm_id,
               "slurm_job_partition": args.slurm_job_partition,
               "slurm_cpus_per_task": args.slurm_cpus_per_task,
               "slurm_mem_per_cpu": args.slurm_mem_per_cpu,
               "slurm_tmp_mem": args.slurm_tmp_mem,
               "slurm_time": args.slurm_time})

# Select variant
file_path = Path(__file__)
file_stem = file_path.stem

variants_path = file_path.parent / f"{file_stem}variants.yaml"
with open(variants_path, "r") as file:
    variants = yaml.safe_load(file)

slurm_array_task_id = args.slurm_array_task_id

num_variants = len(variants)
n_exp = int(file_stem[-4:])
n_var = slurm_array_task_id % num_variants
n_rep = slurm_array_task_id // num_variants

config.update({"experiment": n_exp,
               "variant": n_var,
               "replicate": n_rep})

# Run name and directory for results
run_name = f"{file_stem}_var{n_var:04}_rep{n_rep:03}"
output_dir = Path(args.output_dir)
run_dir = output_dir / "results" / run_name  # directory for results

os.makedirs(run_dir, exist_ok=True)
print(f"Run Name: {run_name}")

tmp_dir = args.tmp_dir
if tmp_dir is not None:
    with open(Path(tmp_dir) / "export_variables.sh", "w") as file:
        file.write(f"export EXP_DIR='{run_dir}'\n")

# Load variant params
variant_params = variants[n_var]
print("Variant Params:")
for k, v in variant_params.items():
    print(f"{k}: {v}")
with open(run_dir / "variant_params.yaml", "w") as file:
    yaml.dump(variant_params, file)

# Wandb settings
wandb_project = "Official"
wandb_dir = output_dir  # directory for wandb files

# Core settings
n_epochs = 30
n_train_steps_per_epoch = 2000
n_train_steps_per_fit = 1
n_eval_episodes_per_epoch = 100
initial_states = None
seeds = list(range(n_eval_episodes_per_epoch))
epochs_per_save = 10
get_action_info = lambda: epoch % epochs_per_save == 0
config.update({"n_epochs": n_epochs,
               "n_train_steps_per_epoch": n_train_steps_per_epoch,
               "n_train_steps_per_fit": n_train_steps_per_fit,
               "n_eval_episodes_per_epoch": n_eval_episodes_per_epoch,
               "initial_states": initial_states,
               "seeds": seeds})
if (initial_states is not None) or (seeds is not None):
    n_eval_episodes_per_epoch = None

# Results settings
quiet_tqdm = True
record_video = False
save_episode_returns = lambda: True
save_dataset = lambda: epoch % epochs_per_save == 0
save_dataset_info = lambda: epoch % epochs_per_save == 0
save_agent = lambda: epoch % epochs_per_save == 0

# MDP settings
mdp_cls = MO_Gymnasium
env_id = "living-room-v0"
horizon = 500
gamma = 0.999
transform_reward_idxs = [0,1, 2]
acc_reward_idxs = None
normalizer_cls = ExpWeighted_MeanStd_Normalizer
normalizer_params = {"input_vec_len": len(transform_reward_idxs),
                     "buffer_len": 1000,
                     "update_rate": 0.01,
                     "track_stats": True,
                     "eps_std": 1e-4,
                     "history_freq": 1}
normalizer = normalizer_cls(**normalizer_params)
wrappers = [#(LL_Terminations, {}),
            (FlattenObservation, {}),(TransformVectorReward, {"transformer": lambda vec: vec[transform_reward_idxs],
                                     "output_vec_len": len(transform_reward_idxs),
                                     "untransformed_info": True}),
            (RewardAccumulator, {"gamma": gamma}),
            (AccumulatedRewardsStateAugmentor, {"acc_reward_idxs": acc_reward_idxs,
                                                "normalizer": normalizer,
                                                "append_gamma_decay": True}),
                                                ]

if 'breakable' in env_id:
    env_args = dict(size=5,
        prob_drop=0.1,
        time_penalty=-1,
        bottle_reward=25,
        unbreakable_bottles=False,)
elif 'living' in env_id:
    env_args = dict(width=4, height=4,
                    start_location=3,
                    goal_location=3,
                    rubbish_location=5,
                    table_location=5, cat_location=6,
                    obstacle_locations=[8, 10],
                    time_penalty=-1, 
                    goal_reward=50, 
                    displacement_penalty=-50, 
                    cat_penalty=-50)
        
elif 'lunar' in env_id:
    env_args = {"step_random": 1.0,
            "train_seeds": None}

config.update({"mdp": mdp_cls.__name__,
               "env_id": env_id,
               "horizon": horizon,
               "gamma": gamma,
               "normalizer": normalizer_cls.__name__,
               "normalizer_params": normalizer_params})
for i, wrapper in enumerate(wrappers):
    if isinstance(wrapper, tuple):
        wrapper_cls, wrapper_params = wrapper
    else:
        wrapper_cls = wrapper
        wrapper_params = None
    config[f"wrapper[{i}]"] = wrapper_cls.__name__
    config[f"wrapper[{i}]_params"] = wrapper_params
for k, v in env_args.items():
    config[f"env_args[{k}]"] = v

# Agent settings
agent_cls = MO_DQN
batch_size = 32
target_update_frequency = 200
initial_replay_size = 1000
max_replay_size = 10000
agent_info_keys = ["accumulated_reward", "gamma_decay"]

config.update({"agent": agent_cls.__name__,
               "batch_size": batch_size,
               "target_update_frequency": target_update_frequency,
               "initial_replay_size": initial_replay_size,
               "max_replay_size": max_replay_size,
               "agent_info_keys": agent_info_keys})

# Policy settings
policy_cls = ScalarizedEpsGreedy
epsilon_threshold = 0.2
epsilon_decay_steps = 100000
scalarizer_cls = Linear_Scalarizer

print(variant_params, type(variant_params))
variant_params: dict
w = []
while len(variant_params.keys()) > 0:
    itemm = variant_params.popitem()
    w.append(itemm[1])
    print(f"Pop item: {w}")

scalarizer_params = {"weights": list(reversed(w))}
use_acc_rewards = True
randomize_ties = False

config.update({"policy": policy_cls.__name__,
               "epsilon_threshold": epsilon_threshold,
               "epsilon_decay_steps": epsilon_decay_steps,
               "scalarizer": scalarizer_cls.__name__,
               "use_acc_rewards": use_acc_rewards,
               "acc_reward_idxs": acc_reward_idxs,
               "randomize_ties": randomize_ties})
if scalarizer_params is not None:
    for k, v in scalarizer_params.items():
        config[f"scalarizer[{k}]"] = v

# Approximator settings
approximator_cls = TorchApproximator
regressor_type = "QRegressor"  # Choose from "QRegressor" or "ActionRegressor"
network_cls = Network_SOMO
hidden_layers = [128, 128, 128, 128]
use_cuda = torch.cuda.is_available()
loss_func = F.smooth_l1_loss
optimizer_cls = optim.Adam
optimizer_params = {'lr': 0.00005}
activation_cls = torch.nn.LeakyReLU
activation_params = {}

config.update({"approximator": approximator_cls.__name__,
               "regressor_type": regressor_type,
               "network": network_cls.__name__,
               "hidden_layers": hidden_layers,
               "use_cuda": use_cuda,
               "loss_func": loss_func.__name__,
               "optimizer": optimizer_cls.__name__,
               "activation": activation_cls.__name__})
for k, v in optimizer_params.items():
    config[f"optimizer[{k}]"] = v
for k, v in activation_params.items():
    config[f"activation[{k}]"] = v

# End settings
assert len(variant_params) == 0, f"All variant params must be popped. {variant_params=}"

# MDP
mdp: MO_Gymnasium = mdp_cls(name=env_id,
              horizon=horizon,
              gamma=gamma,
              wrappers=wrappers,
              **env_args)

observation_shape = mdp.info.observation_space.shape
n_actions = mdp.info.action_space.n
n_objectives = mdp.info.reward_space.shape[0]
#print("NOOBJS", n_objectives, mdp.info, mdp.info.reward_space, env_id)
assert len(mdp.info.reward_space.shape) == 1

# Policy
epsilon_init = Parameter(value=1.0)
epsilon_train = LinearParameter(value=1.0,
                                threshold_value=epsilon_threshold,
                                n=epsilon_decay_steps)
epsilon_eval = Parameter(value=0.)
scalarizer = scalarizer_cls(**scalarizer_params)

assert policy_cls.__name__ == "ScalarizedEpsGreedy"
policy = policy_cls(observation_shape=observation_shape,
                    n_actions=n_actions,
                    n_objectives=n_objectives, 
                    epsilon=epsilon_init,
                    scalarizer=scalarizer,
                    use_acc_rewards=use_acc_rewards,
                    acc_reward_idxs=acc_reward_idxs,
                    randomize_ties=randomize_ties)

# Approximator
output_shape = select_output_shape(regressor_type=regressor_type,
                                   n_actions=n_actions,
                                   n_objectives=n_objectives)
assert approximator_cls.__name__ == "TorchApproximator"
approximator_params = {"input_shape": observation_shape,
                       "output_shape": output_shape,
                       "n_actions": n_actions,
                       "network": network_cls,
                       "optimizer": {'class': optimizer_cls, 'params': optimizer_params},
                       "loss": loss_func,
                       "use_cuda": use_cuda,
                       "hidden_layers": hidden_layers,
                       "activation_cls": activation_cls,
                       "activation_params": activation_params}

# Agent
assert agent_cls.__name__ == "MO_DQN"
agent = agent_cls(mdp_info=mdp.info,
                  policy=policy,
                  approximator=approximator_cls,
                  approximator_params=approximator_params,
                  batch_size=batch_size,
                  target_update_frequency=target_update_frequency,
                  initial_replay_size=initial_replay_size,
                  max_replay_size=max_replay_size,
                  agent_info_keys=agent_info_keys)

# Core
core = Core(agent=agent, mdp=mdp, agent_info_keys=agent_info_keys)

# Initialize Wandb
wandb.init(project=wandb_project,
           config=config,
           name=run_name,
           save_code=True,
           dir=wandb_dir)
wandb_name = str(Path(wandb.run.dir).parent.name)
print(f"Wandb Name: {wandb_name}")
wandb.define_metric("*", step_metric="train_steps")

# Initialize times
train_time = 0
eval_time = 0
stats_time = 0
times = []

# Initialize replay memory
strong_line = "##############################"
print(strong_line)
epoch = 0
normalizer.track_stats = True
train_time = train(core=core,
                   agent_morl=policy,
                   epsilon_train=epsilon_init,  # purely random policy
                   n_train_steps_per_epoch=initial_replay_size - 1,  # almost fills replay memory
                   n_train_steps_per_fit=initial_replay_size - 1,  # does not fit network while filling replay memory
                   quiet_tqdm=quiet_tqdm,
                   train_time=train_time)

#exit()
# Evaluate
normalizer.track_stats = False
dataset, dataset_info, eval_time = evaluate(core=core,
                                            mdp=mdp,
                                            policy=policy,
                                            epsilon_eval=epsilon_eval,
                                            epoch=epoch,
                                            n_eval_episodes_per_epoch=n_eval_episodes_per_epoch,
                                            horizon=horizon,
                                            quiet_tqdm=quiet_tqdm,
                                            eval_time=eval_time,
                                            record_video=record_video,
                                            run_dir=run_dir,
                                            initial_states=initial_states,
                                            seeds=seeds,
                                            get_action_info=get_action_info())


# Stats
stats_params = {"scalarizer": scalarizer, "gamma": gamma}  # Include gamma if stats of discounted return are desired
stats = {f"normalizer_mean[{i}]": x for i, x in enumerate(normalizer.mean)}
stats.update({f"normalizer_std[{i}]": x for i, x in enumerate(normalizer.std)})
save_normalizer_history(normalizer=normalizer, run_dir=run_dir, epoch=epoch, clear_history=True)
stats_time = log_stats(dataset=dataset,
                       dataset_info=dataset_info,
                       agent=agent,
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
    print(strong_line)
    # Train
    normalizer.track_stats = True
    train_time = train(core=core,
                       agent_morl=policy,
                       epsilon_train=epsilon_train,
                       n_train_steps_per_epoch=n_train_steps_per_epoch,
                       n_train_steps_per_fit=n_train_steps_per_fit,
                       quiet_tqdm=quiet_tqdm,
                       train_time=train_time)

    # Evaluate
    normalizer.track_stats = False
    dataset, dataset_info, eval_time = evaluate(core=core,
                                                mdp=mdp,
                                                policy=policy,
                                                epsilon_eval=epsilon_eval,
                                                epoch=epoch,
                                                n_eval_episodes_per_epoch=n_eval_episodes_per_epoch,
                                                horizon=horizon,
                                                quiet_tqdm=quiet_tqdm,
                                                eval_time=eval_time,
                                                record_video=record_video,
                                                run_dir=run_dir,
                                                initial_states=initial_states,
                                                seeds=seeds,
                                                get_action_info=get_action_info())

    # Stats
    stats = {f"normalizer_mean[{i}]": x for i, x in enumerate(normalizer.mean)}
    stats.update({f"normalizer_std[{i}]": x for i, x in enumerate(normalizer.std)})
    save_normalizer_history(normalizer=normalizer, run_dir=run_dir, epoch=epoch, clear_history=True)
    stats_time = log_stats(dataset=dataset,
                           dataset_info=dataset_info,
                           agent=agent,
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

df_times = pd.DataFrame(data=times)
df_times.to_pickle(run_dir / "df_times.pkl")
wandb.finish()
