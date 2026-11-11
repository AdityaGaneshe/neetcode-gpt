import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        torch.manual_seed(0)

        # Valid start indices: start + context_length must be a valid index for Y,
        # so start can range from 0 to len(data) - context_length - 1 (inclusive).
        # torch.randint's upper bound is exclusive, hence len(data) - context_length.
        ix = torch.randint(0, len(data) - context_length, (batch_size,))

        X = torch.stack([data[i : i + context_length] for i in ix])
        Y = torch.stack([data[i + 1 : i + 1 + context_length] for i in ix])

        return X, Y