"""Transformer causal (decoder-only, pre-norm) com RoPE para as tarefas sintéticas.

Saída: logits (B, T, vocab_size), alinhados com `labels` do dataset (use ignore_index=-100).
RoPE não tem tabela de posições aprendida, então o modelo roda em T maior que o do treino
(necessário para o ar_extrapolate: treina em T=64, testa em T=128).
"""
import torch
import torch.nn.functional as F
from torch import nn


def rope(x: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding em x (B, heads, T, head_dim)."""
    T, d = x.size(-2), x.size(-1)
    freqs = 10000 ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    angles = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * freqs[None]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Linear(dim_feedforward, d_model))
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, C)))
        return x + self.drop(self.ff(self.norm2(x)))


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, num_layers: int = 2, num_heads: int = 4,
                 dim_feedforward: int | None = None, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0 and (d_model // num_heads) % 2 == 0
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            Block(d_model, num_heads, dim_feedforward or 4 * d_model, dropout) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))
