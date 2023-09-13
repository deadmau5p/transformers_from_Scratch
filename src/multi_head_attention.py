import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads) -> None:
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.W_out = nn.Linear(d_model, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None):
        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))

        attention_scores: torch.Tensor = self.scaled_dot_product(Q, K, V, mask)

        attention_scores = self.combine_heads(attention_scores)

        output = self.W_out(self.combine_heads(attention_scores))

        return output

    def scaled_dot_product(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask
    ) -> torch.Tensor:
        K = K.transpose(2, 3)
        attention_scores: torch.Tensor = torch.matmul(Q, K) / math.sqrt(self.d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_scores, V)
        return out

    def split_heads(self, QKV: torch.Tensor):
        batch_size, num_tokens, _ = QKV.size()
        QKV = QKV.view(batch_size, num_tokens, self.num_heads, self.d_k)

        return QKV.transpose(1, 2)

    def combine_heads(self, attentions_scores: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_tokens, d_k = attentions_scores.size()
        return (
            attentions_scores.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.d_model)
        )
