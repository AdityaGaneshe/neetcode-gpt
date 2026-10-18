import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        W = torch.randn(fan_out, fan_in) * std
        return [[round(v, 4) for v in row] for row in W.tolist()]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        W = torch.randn(fan_out, fan_in) * std
        return [[round(v, 4) for v in row] for row in W.tolist()]

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)

        # 1) Build all weight matrices first
        weights = []
        fan_in = input_dim
        for _ in range(num_layers):
            fan_out = hidden_dim
            W = torch.randn(fan_out, fan_in)
            if init_type == 'xavier':
                W = W * math.sqrt(2.0 / (fan_in + fan_out))
            elif init_type == 'kaiming':
                W = W * math.sqrt(2.0 / fan_in)
            # 'random': plain N(0, 1)
            weights.append(W)
            fan_in = fan_out

        # 2) Then draw the random input
        x = torch.randn(input_dim)

        # 3) Forward pass, recording std after each layer
        stds = []
        for W in weights:
            x = torch.relu(W @ x)
            stds.append(round(x.std().item(), 2))
        return stds