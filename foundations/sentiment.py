import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding_layer = nn.Embedding(vocabulary_size, 16)
        self.linear_layer = nn.Linear(16, 1)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Convert input to LongTensor (required by nn.Embedding if passed as float)
        embeddings = self.embedding_layer(x.long())
        
        # Average across the sequence dimension (dim=1)
        # Shape transitions from (batch_size, seq_len, 16) -> (batch_size, 16)
        averaged = torch.mean(embeddings, dim=1)
        
        # Project to a single logit and apply Sigmoid for [0, 1] probability
        projected = self.linear_layer(averaged)
        output = self.sigmoid_layer(projected)
        
        return torch.round(output, decimals=4)