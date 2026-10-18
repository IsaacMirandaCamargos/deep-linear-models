"""Cumulative majority. Código de zoology/data/circuits.py (cumulative_majority).

Label em cada posição = 1 se, até ali, #uns >= #zeros.
No zoology o default do zoology é 16 (só 65536 sequências possíveis, gera duplicatas), aqui usamos 64 e a função recebe a config; aqui
a config é um SimpleNamespace com os mesmos campos.

Uso: python data_gen/zoology/cumulative_majority.py --input-seq-len 64
Saída: data/cumulative_majority/{train,test}.pt + config.json
"""
from types import SimpleNamespace

import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DataSegment, make_parser, run


def cumulative_majority(
    config,
    seed: int,
    **kwargs
) -> DataSegment:
    """
    Generate majority sequences.
    """
    np.random.seed(seed)

    inputs = np.random.randint(0, 2, size=(config.num_examples, config.input_seq_len))
    labels = ((inputs * 2 - 1).cumsum(axis=1) >= 0).astype(int)

    return DataSegment(
        torch.tensor(inputs),
        torch.tensor(labels),
        slices={"input_seq_len": config.input_seq_len}
    )


def build(num_examples: int, input_seq_len: int, seed: int, **kwargs) -> DataSegment:
    config = SimpleNamespace(num_examples=num_examples, input_seq_len=input_seq_len)
    return cumulative_majority(config, seed=seed)


if __name__ == "__main__":
    run("cumulative_majority", build, make_parser("cumulative_majority", __doc__, vocab_size=2, input_seq_len=64))
