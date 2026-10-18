"""Cumulative parity. Código de zoology/data/circuits.py (cumulative_parity).

Label em cada posição = paridade acumulada dos bits até ali.

Uso: python data_gen/zoology/cumulative_parity.py --input-seq-len 64
Saída: data/cumulative_parity/{train,test}.pt + config.json
"""
import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DataSegment, make_parser, run


def cumulative_parity(
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
        parities = []
        cur_parity = 0
        for i in range(input_seq_len-1):
            cur_parity = (cur_parity + seq[i]) % 2
            parities.append(cur_parity)

        # Append the parity to the sequence
        input = np.array(seq, dtype=np.int64)
        input = torch.tensor(input)
        label = torch.tensor(parities)

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
    run("cumulative_parity", cumulative_parity, make_parser("cumulative_parity", __doc__, vocab_size=2))
