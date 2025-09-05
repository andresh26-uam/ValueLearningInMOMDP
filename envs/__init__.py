from gymnasium.envs.registration import register
from envs.moral_envs import MoralGymWrapperProxy, MoralVecEnvProxy
from use_cases.moral_envs.randomized_v3 import MAX_STEPS

register(
    'vec-env-MORAL-v0',
    MoralVecEnvProxy,
    max_episode_steps=MAX_STEPS+1
)

register(
    'env-MORAL-v0',
    MoralGymWrapperProxy,
    max_episode_steps=MAX_STEPS+1
)

register(
     id="FireFighters-v0",
     entry_point="envs.firefighters_env:FireFightersEnv",
     max_episode_steps=40,
)
register(
     id="FireFightersEnvWithObservation-v0",
     entry_point="envs.firefighters_env:FireFightersEnvWithObservation",
)

register(
     id="RoadWorldEnvMO-v0",
     entry_point="envs.roadworld_env:RoadWorldEnvMO",
     max_episode_steps=500,
)

register(
     id="RouteChoiceEnvironmentApollo-v0",
     entry_point="envs.routechoiceApollo:RouteChoiceEnvironmentApollo",
     max_episode_steps=1,
)

register(
     id="RouteChoiceEnvironmentApolloComfort-v0",
     entry_point="envs.routechoiceApollo:RouteChoiceEnvironmentApolloComfort",
     max_episode_steps=1,
)

register(
     id="MultiValuedCarEnv-v0",
     entry_point="envs.multivalued_car_env:MultiValuedCarEnv",
)

register(
    id='FireFightersMO-v0',
    entry_point='envs.firefighters_env_mo:FireFightersEnvMO',
    max_episode_steps=50,
)