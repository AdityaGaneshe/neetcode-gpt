import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        torch.manual_seed(0)

        # 1. Tokenize by splitting on whitespace
        tokens = raw_dataset.split()

        # 2. Sample batch_size random start indices in [0, len(tokens) - context_length)
        #    (upper bound is exclusive, so i + context_length is always a valid label index)
        indices = torch.randint(low=0, high=len(tokens) - context_length, size=(batch_size,))

        # 3. Slice inputs and labels (labels are shifted one position to the right)
        X, Y = [], []
        for i in indices.tolist():
            X.append(tokens[i : i + context_length])
            Y.append(tokens[i + 1 : i + 1 + context_length])

        return X, Y