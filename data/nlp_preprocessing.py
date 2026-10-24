import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign IDs starting at 1
        words = set()
        for sentence in positive + negative:
            words.update(sentence.split())
        vocab = {word: i + 1 for i, word in enumerate(sorted(words))}

        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        tensors = []
        for sentence in positive + negative:
            ids = [vocab[word] for word in sentence.split()]
            tensors.append(torch.tensor(ids, dtype=torch.float))

        # 4. Pad shorter sequences with 0s
        return nn.utils.rnn.pad_sequence(tensors, batch_first=True)