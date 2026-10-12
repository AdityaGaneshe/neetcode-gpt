import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        return np.round(1 / (1 + np.exp(-z)), 5)

        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        pass

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        return np.maximum(0, z)
        # Formula: max(0, z) element-wise
        pass
