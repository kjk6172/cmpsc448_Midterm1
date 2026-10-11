"""Compact neural classifiers with explicit padding-aware representations."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int = 64,
                 hidden_dim: int = 48, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.kernels = (3, 4, 5)
        self.convolutions = nn.ModuleList([nn.Conv1d(embedding_dim, hidden_dim, k) for k in self.kernels])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * len(self.kernels), num_classes)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if token_ids.size(1) < max(self.kernels):
            token_ids = nn.functional.pad(token_ids, (0, max(self.kernels) - token_ids.size(1)))
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = []
        for kernel, convolution in zip(self.kernels, self.convolutions):
            activation = torch.relu(convolution(embedded))
            # Only fully contained windows count. Very short texts get one
            # window completed by zero padding, consistently at every batch size.
            valid_windows = (lengths - kernel + 1).clamp(min=1)
            starts = torch.arange(activation.size(2), device=activation.device)[None, :]
            valid = starts < valid_windows.to(activation.device)[:, None]
            activation = activation.masked_fill(~valid[:, None, :], float("-inf"))
            pooled.append(activation.max(dim=2).values)
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int = 64,
                 hidden_dim: int = 48, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.encoder(packed)
        return self.classifier(self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1)))


def make_model(name: str, **kwargs) -> nn.Module:
    if name == "cnn":
        return TextCNN(**kwargs)
    if name == "lstm":
        return BiLSTM(**kwargs)
    raise ValueError(f"Unknown model architecture: {name}")
