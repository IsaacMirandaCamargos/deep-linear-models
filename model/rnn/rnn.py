"""RNN simples (Elman) para as tarefas sintéticas: embedding -> RNN -> projeção ao vocabulário.

Saída: logits (B, T, vocab_size), alinhados com `labels` do dataset (use ignore_index=-100).
"""
import torch
from torch import nn


class RNNModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.RNN(d_model, d_model, num_layers=num_layers, batch_first=True,
                          nonlinearity="tanh", dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.embed(x))
        return self.head(h)
