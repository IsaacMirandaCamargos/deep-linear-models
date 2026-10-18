"""Vocab majority (token mais frequente). Código de zoology/data/circuits.py (vocab_majority).

Sequência de tokens em [1, vocab_size), token 0 (end) e o token mais frequente como alvo.
Único desvio do original: removido o `print(seq)` que imprimia cada exemplo.

Uso: python data_gen/zoology/vocab_majority.py --vocab-size 16 --input-seq-len 64
Saída: data/vocab_majority/{train,test}.pt + config.json
"""
import numpy as np
import torch
from collections import Counter

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DataSegment, make_parser, run


def vocab_majority(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    **kwargs
) -> DataSegment:
    """
    Generate majority sequences.
    """
    np.random.seed(seed)
    end = 0

    inputs = []
    labels = []
    for _ in range(num_examples):

        # Generate a random sequence of 0s and 1s
        seq = np.random.randint(1, vocab_size, size=input_seq_len-1)
        seq = [int(s) for s in seq]
        counter = Counter(seq)
        max_count = max(counter.values())
        keys_with_max_count = [k for k, v in counter.items() if v == max_count]
        most_key = min(keys_with_max_count)
        if len(keys_with_max_count) > 1:
            other_key = [k for k in counter.keys() if k != most_key][0]
            # replace an instance of other_key with most_key
            idx = seq.index(other_key)
            seq[idx] = most_key

        # Full sequence
        input = np.array(seq + [end] + [most_key], dtype=np.int64)

        # Inputs and outputs
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
        slices={"vocab_size": vocab_size}
    )


if __name__ == "__main__":
    run("vocab_majority", vocab_majority, make_parser("vocab_majority", __doc__, vocab_size=16))
