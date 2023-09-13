import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """
    This class is neural network module used to add positional
    embeddings to word embeddings. This way transformers get
    information about order of sequence, which is vital for transformers to work.
    """

    def __init__(self, d_model: int, max_sequence_length: int) -> None:
        super(PositionalEncoding, self).__init__()
        self.positional_encodings = torch.zeros(max_sequence_length, d_model)

        position = torch.arange(0, max_sequence_length).reshape(-1, 1)

        geometric_progression_term = 1 / (
            10000 ** (torch.arange(0, d_model, 2) / d_model)
        )

        self.positional_encodings[:, 0::2] = torch.sin(
            position * geometric_progression_term
        )
        self.positional_encodings[:, 1::2] = torch.cos(
            position * geometric_progression_term
        )

    def forward(self, word_embeddings: Tensor) -> Tensor:
        """
        Method receives word embeddings and combines it with sinusoidal positional
        encoding outputs new embeddings.

        Args:
            word_embeddings (Tensor): embeddings of each token of size (batch_size x
            num_tokens x embeddings_dim)

        Returns:
            word embeddings with information about position of each token in sequence
        """

        return word_embeddings + self.positional_encodings[:, : word_embeddings.size(1)]


if __name__ == "__main__":
    pe = PositionalEncoding(8, 5)
