import torch
from torch import Tensor
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from FeedForwardNetwork import FeedForwardNN


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm = nn.LayerNorm(d_model, d_model)
        self.ffn = FeedForwardNN(d_model, d_hidden_size=0)

    def forward(self, positional_embeddings: Tensor):
        