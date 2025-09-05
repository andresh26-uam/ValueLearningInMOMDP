
import os
from typing import Iterable

import torch


MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = os.path.join(MODULE_PATH, "checkpoints/")
TRAIN_RESULTS_PATH = os.path.join(MODULE_PATH, "train_results/")


os.makedirs(CHECKPOINTS, exist_ok=True)


def transform_weights_to_tuple(weights: Iterable|str, size_should_be: int = None) -> tuple:
        if isinstance(weights, str):
            weights = weights.split(",")
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().tolist()
        assert len(weights) > 1, f"Unrecognized weights {weights}"
        if size_should_be is not None:
            assert len(weights) == size_should_be, f"Expected {size_should_be} weights, got {len(weights)}"
        weights_real = tuple([float(f"{float(a):.3f}") for a in weights])
        return weights_real