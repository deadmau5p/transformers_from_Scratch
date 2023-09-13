import torch.nn as nn
from torch import Tensor


class FeedForwardNN(nn.Module):
    def __init__(self, d_model, d_hidden_size) -> None:
        super(FeedForwardNN, self).__init__(d_model, d_hidden_size)

        self.input_W = nn.Linear(d_model, d_hidden_size)
        self.out_W = nn.Linear(d_hidden_size, d_model)

        self.relu = nn.ReLU()

    def forward(self, word_embeddings: Tensor):
        out = self.out_W(self.relu(self.input_W(word_embeddings)))
        return out
