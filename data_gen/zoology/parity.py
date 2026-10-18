"""Parity. Código de zoology/data/circuits.py (parity).

Sequência de bits, token 2 (end) e a paridade como alvo; só a última posição tem label.

Uso: python data_gen/zoology/parity.py --input-seq-len 64
Saída: data/parity/{train,test}.pt + config.json
"""
import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DataSegment, make_parser, run


def parity(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    **kwargs
) -> DataSegment:
    """
    Generate parity sequences.
    """
    np.random.seed(seed)

    one = 1
    zero = 0
    end = 2

    inputs = []
    labels = []
    for _ in range(num_examples):
        # Generate a random sequence of 0s and 1s
        seq = np.random.randint(0, 2, size=input_seq_len-1)
        parity = len([b for b in seq if b == 1]) % 2

        # convert to vocab
        seq = [one if s == 1 else zero for s in seq]
        if parity: parity = one
        else: parity = zero

        # Append the parity to the sequence
        input = np.array(seq + [end] + [parity], dtype=np.int64)
        input = torch.tensor(input)
        label = torch.full_like(input[:-1], -100) # -100 for labels, except last position
        label[-1] = input[-1]
        input = input[:-1]

        inputs.append(input)
        labels.append(label)

    inputs = torch.stack(inputs)
    labels = torch.stack(labels)

    return DataSegment(
        inputs,
        labels,
        slices={"input_seq_len": input_seq_len}
    )


if __name__ == "__main__":
    run("parity", parity, make_parser("parity", __doc__, vocab_size=3))
