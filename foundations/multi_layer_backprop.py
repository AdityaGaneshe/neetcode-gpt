import numpy as np
from typing import List

class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float]
    ) -> dict:

        # Convert inputs to NumPy arrays
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # Forward pass: First linear layer
        z1 = W1 @ x + b1

        # ReLU activation
        a1 = np.maximum(0, z1)

        # Forward pass: Second linear layer
        predictions = W2 @ a1 + b2

        # Mean Squared Error loss
        loss = np.mean((predictions - y_true) ** 2)

        # Backward pass: Output layer
        d_pred = 2 * (predictions - y_true) / len(y_true)

        dW2 = np.outer(d_pred, a1)
        db2 = d_pred

        # Backward pass: ReLU activation
        d_a1 = W2.T @ d_pred
        d_z1 = d_a1 * (z1 > 0)

        # Backward pass: First linear layer
        dW1 = np.outer(d_z1, x)
        db1 = d_z1

        # Round all values to 4 decimal places
        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }