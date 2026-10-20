
import numpy as np
from typing import List

class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float = 1e-5) -> List[float]:
        # Convert input to NumPy arrays
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)

        # Calculate Root Mean Square
        rms = np.sqrt(np.mean(x ** 2) + eps)

        # Normalize x
        normalized_x = x / rms

        # Scale using gamma
        output = gamma * normalized_x

        # Round to 4 decimal places and return as a list
        return np.round(output, 4).tolist()
