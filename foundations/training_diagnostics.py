import math
import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        activation_stats = []

        def hook(module, inputs, output):
            # Fraction of neurons with output <= 0 for ALL samples in the batch
            dead_fraction = (output <= 0).all(dim=0).float().mean().item()

            activation_stats.append({
                "mean": round(output.mean().item(), 4),
                "std": round(output.std().item(), 4),
                "dead_fraction": round(dead_fraction, 4),
            })

        handles = [
            m.register_forward_hook(hook)
            for m in model.modules()
            if isinstance(m, nn.Linear)
        ]

        try:
            with torch.no_grad():
                model(x)
        finally:
            for h in handles:
                h.remove()

        return activation_stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad()

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        gradient_stats = []

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad

                gradient_stats.append({
                    "mean": round(grad.mean().item(), 4),
                    "std": round(grad.std().item(), 4),
                    "norm": round(torch.norm(grad).item(), 4),
                })

        return gradient_stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:

        def is_bad(v: float) -> bool:
            return math.isnan(v) or math.isinf(v)

        # 1. Dead neurons
        if any(s["dead_fraction"] > 0.5 for s in activation_stats):
            return "dead_neurons"
        # All gradients and vanishing in the layers

        # 2. Exploding gradients (NaN/inf also counts as exploded)
        if any(is_bad(s["norm"]) or s["norm"] > 1000 for s in gradient_stats):
            return "exploding_gradients"

        # 3. Vanishing gradients in the last layer
        if gradient_stats and gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # 4. Activation std checks across all layers
        if any(s["std"] < 0.1 for s in activation_stats):
            return "vanishing_gradients"

        if any(is_bad(s["std"]) or s["std"] > 10.0 for s in activation_stats):
            return "exploding_gradients"

        # 5. Healthy
        return "healthy"
