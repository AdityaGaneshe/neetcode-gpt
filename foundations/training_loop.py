
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

class Solution:
    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        epochs: int,
        lr: float
    ) -> Tuple[NDArray[np.float64], float]:

        # Get the number of samples and features
        n_samples, n_features = X.shape

        # Initialize weights and bias
        w = np.zeros(n_features)
        b = 0.0

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            y_hat = X @ w + b

            # Calculate prediction errors
            error = y_hat - y

            # Calculate gradients
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            # Update weights and bias
            w = w - lr * dw
            b = b - lr * db

        # Round results to 5 decimal places
        return np.round(w, 5), round(float(b), 5)
