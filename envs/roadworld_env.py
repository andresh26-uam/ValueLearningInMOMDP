import numpy as np
from use_cases.roadworld_env_use_case.network_env import ILLEGAL_REWARD, FeaturePreprocess, FeatureSelection, RoadWorldGym

from gymnasium.spaces import Box

from use_cases.roadworld_env_use_case.values_and_costs import BASIC_PROFILES
class RoadWorldEnvMO(RoadWorldGym):
    def __init__(self, horizon=50, masked=True, feature_selection=(FeatureSelection.ONLY_COSTS, FeatureSelection.DEFAULT), feature_preprocessing=FeaturePreprocess.NORMALIZATION, render_mode=None, destination_method=64):
        super().__init__(horizon, masked=masked, feature_selection=feature_selection, feature_preprocessing=feature_preprocessing, visualize_example=True, destination_method=destination_method)
    
        self.reward_space = Box(low=ILLEGAL_REWARD, high=0.0, shape=(len(BASIC_PROFILES),), dtype=np.float32)
        self.reward_dim = len(BASIC_PROFILES)

        
    
    metadata = {'render.modes': ['human']}