import numpy as np
import gymnasium as gym
import mo_gymnasium as mo_gym
import time
from baraacuda.agents.TLO_agent import TLO_agent
from baraacuda.utils.state_augmentor import ActualRewardsStateAugmentor
from baraacuda.utils.wrappers import RewardAccumulator
from baraacuda.TLO.action_selector import SoftThresholdingNonLinearActionSelector, TLOActionSelector, LexActionSelector, LinearScalarizerActionSelector
from baraacuda.value_functions import QTableValueFunction, AccumulatedValueFunction
from mo_gymnasium.utils import MORecordEpisodeStatistics
from baraacuda.utils.transform_reward_for_experiment import TransformRewardForExperiment
from baraacuda.utils.exploration_policy import SoftmaxTournamentExploration, ExponentialInterpolateScheduler
import wandb

from baraacuda.envs import doors, sokoban

if __name__ == "__main__":
    NUM_EPISODES = 5000
    GAMMA = 1.0
    LAM = 0.95
    ALPHA = 0.1
    
    rng = np.random.default_rng()
    
    # We should be able to use this code with any MO Gym environment
    env_bottles = mo_gym.make('breakable-bottles-v0', max_episode_steps=1000)
    env_sokoban = mo_gym.make('sokoban-v0', max_episode_steps=1000)
    env_doors = mo_gym.make('doors-v0', max_episode_steps=1000)
    
    # We make separate eval versions so we can do mid-episode eval
    env_eval_bottles = mo_gym.make('breakable-bottles-v0', max_episode_steps=1000)
    env_eval_sokoban = mo_gym.make('sokoban-v0', max_episode_steps=1000)
    env_eval_bottles = mo_gym.make('doors-v0', max_episode_steps=1000)
    
    # Let's pick one
    env = env_sokoban
    env_eval = env_eval_sokoban
    
    # Helper wrapper for logging
    env = MORecordEpisodeStatistics(env=env, gamma=1.0)
    env_eval = MORecordEpisodeStatistics(env=env_eval, gamma=1.0)
    
    # The MO Gym format environments return 3 rewards, so we use this helper wrapper to combine them into the 2 rewards used in the Potential-based MORL paper
    # You can omit this wrapper, but you'll need to make changes to the rest of the code in this file to handle the environment having 3 rewards
    env = TransformRewardForExperiment(env)
    env_eval = TransformRewardForExperiment(env_eval)
    
    obs, info = env.reset()
    
    action_space = env.get_wrapper_attr('action_space')
    reward_space = env.get_wrapper_attr('reward_space')
    
    # Some of the algorithms require state augmentation, so we make a state augmentor that wraps the environment.
    # When we don't need this, we use the environment directly.
    # We can optionally discretise any augmented objectives by providing the number of resultant discrete values, a minimum value, and a maximum value
    state_augmentor = ActualRewardsStateAugmentor(env, objectives=[0], discretisations={0: (10, -100, 50)})
    eval_state_augmentor = ActualRewardsStateAugmentor(env_eval, objectives=[0], discretisations={0: (10, -100, 50)})
    
    # We create a value function that generates values from state-action pairs.
    # In this case it's a Q Table, but we could replace it with DQN etc.
    value_function = QTableValueFunction(action_space=action_space, reward_space=reward_space, gamma=GAMMA, learning_rate=ALPHA)
    
    # Next, we need an action selector that evaluates the best (greedy) action from a set of action values.
    # We can use the following examples.
    # Note that all of these are subject to the weights and scalarization function passed to the value function above, and I'm assuming that the
    # first objective is the 'primary' objective, and the second objective is the 'safety' objective
    
    # Basic linear scalarizer (scalarizes using the supplied function and picks the maximum value).
    action_selector_linscale = LinearScalarizerActionSelector(scalarization_func=lambda vals: np.dot(vals, [1.0, 1.0]))
    # Lex-P that optimises the primary objective and then the safety objective
    action_selector_lexP = LexActionSelector(objective_lex_order=[0, 1], rng=rng)
    # Lex-A that optimises the safety objective and then the primary objective
    action_selector_lexA = LexActionSelector(objective_lex_order=[1, 0], rng=rng)
    # TLO-A that has a single threshold on the safety objective (needs accumulated safety reward).
    action_selector_tloA = TLOActionSelector(reward_space=reward_space, thresholds={1: -0.1}, rng=rng)
    # TLO-P that has a single threshold on the primary objective (needs augmented state with primary reward, and accumulated primary reward).
    action_selector_tloP = TLOActionSelector(reward_space=reward_space, thresholds={0: -500.0}, rng=rng)
    # TLO-PA with thresholds on both objectives, and tiebreaking via the primary and then safety objectives (needs augmented state with primary reward, and accumulated primary and safety rewards).
    action_selector_tloPA = TLOActionSelector(reward_space=reward_space, thresholds={0: -500.0, 1: -0.1}, rng=rng)
    # TLO-AP with thresholds on both objectives, and tiebreaking via the safety and then primary objectives (needs augmented state with primary reward, and accumulated primary and safety rewards).
    action_selector_tloAP = TLOActionSelector(reward_space=reward_space, thresholds={1: -0.1, 0: -500.0}, rng=rng)
    
    # Richard's soft-thresholding non-linear scalarization algorithm with a single threshold on the safety objective, and a slope of 1/3 on the primary objective (needs augmented state).
    action_selector_richard = SoftThresholdingNonLinearActionSelector(thresholds={1: -0.1}, slopes={0: 1.0/3.0, 1: 1.0})
    
    # Let's pick one of the above
    action_selector = action_selector_tloA
    
    # Comment these out if we aren't using an algorithm that needs an augmented state
    #env = state_augmentor
    #env_eval = eval_state_augmentor
    
    # If we're using an algorithm that expects the accumulated rewards for the episode to have been added to the Q-values, we need to wrap the environment and value function to store and use those accumulated rewards
    # Set the 'accumulated_objectives' list here to contain which objectives should incorporate accumulated rewards (see action selector section above)
    accumulated_objectives = [1]
    # Comment these out if we aren't using an algorithm that needs accumulated rewards
    value_function = AccumulatedValueFunction(value_function, objectives=accumulated_objectives)
    env = RewardAccumulator(env)
    env_eval = RewardAccumulator(env_eval)
    
    TAU_INITIAL = 10.0
    #TAU_INITIAL = 50.0 # for TLO-P and TLO-PA
    TAU_FINAL = 0.01
    exploration_policy = SoftmaxTournamentExploration(temperature_scheduler=ExponentialInterpolateScheduler(initial=TAU_INITIAL, final=TAU_FINAL, period=NUM_EPISODES))
    
    # Set 'wandb_entity' to be the username or team name that you want to upload the logs to (e.g. 'araac'. A value of None will use your personal account)
    agent = TLO_agent(env_name=env.spec.id, env=env, action_selector=action_selector, value_function=value_function, exploration_policy=exploration_policy, project_name="W&B Demo", experiment_name="TLO-A", wandb_entity=None, td_lambda=LAM, td_trace_max=20, rng=rng)
    
    # Add any hyperparameters we want to log to the Weights & Biases config
    # Note that W&B is initialised by the agent, so we need to do this after the agent is created
    # eg wandb.config['hyperparam_x'] = None
    
    agent.train(start_time=time.time(), max_timesteps=np.inf, max_episodes=NUM_EPISODES, eval_freq=100, eval_env=env_eval, log_freq=100)
    
    # Post-training evaluation
    agent.close_wandb()
    agent.setup_wandb(project_name="W&B Demo", experiment_name="TLO-A", wandb_entity=None)
    agent.post_train_eval(env_eval, 100)