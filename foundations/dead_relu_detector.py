import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []

        with torch.no_grad():
            out = x
            # Iterate over layers in order. Works for nn.Sequential (and any
            # module whose children are applied sequentially).
            for layer in model.children():
                out = layer(out)
                if isinstance(layer, nn.ReLU):
                    # A neuron is dead if its output is 0 for every sample in the batch
                    dead_mask = (out == 0).all(dim=0)
                    frac = dead_mask.float().mean().item()
                    dead_fractions.append(round(frac, 4))

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        if not dead_fractions:
            return 'healthy'

        # 1. Severe: any layer more than half dead
        if any(f > 0.5 for f in dead_fractions):
            return 'use_leaky_relu'

        # 2. Early-layer death
        if dead_fractions[0] > 0.3:
            return 'reinitialize'

        # 3. Strictly increasing with depth and last layer > 0.1
        strictly_increasing = all(
            dead_fractions[i] < dead_fractions[i + 1]
            for i in range(len(dead_fractions) - 1)
        )
        if strictly_increasing and len(dead_fractions) > 1 and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'

        # 4. / 5. Healthy (minor dead neurons, or no critical pattern)
        return 'healthy'