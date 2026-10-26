import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        pos = np.arange(seq_len, dtype=np.float64)[:, None]        # (seq_len, 1)
        two_i = np.arange(0, d_model, 2, dtype=np.float64)[None, :]  # (1, d_model/2)

        angles = pos / np.power(10000.0, two_i / d_model)          # (seq_len, d_model/2)

        PE = np.zeros((seq_len, d_model), dtype=np.float64)
        PE[:, 0::2] = np.sin(angles)
        PE[:, 1::2] = np.cos(angles)

        return np.round(PE, 5)