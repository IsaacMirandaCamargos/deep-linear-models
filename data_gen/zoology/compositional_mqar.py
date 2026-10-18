"""MQAR composicional (chave = par K1,K2). Código de zoology/data/compositional_mqar.py.

Uso: python data_gen/zoology/compositional_mqar.py --num-kv-pairs 4 --input-seq-len 64
Saída: data/compositional_mqar/{train,test}.pt + config.json
"""
import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DataSegment, make_parser, run


def compositional_mqar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 4,
    random_non_queries: bool = True,
    include_slices: bool = True,
    **kwargs
) -> DataSegment:
    """
    Usa chaves compostas (K1, K2) em grade: com num_kv_pairs=4 (2x2),
    (A,X)->v1, (A,Y)->v2, (B,X)->v3, (B,Y)->v4. Nem K1 nem K2 sozinhos determinam o valor.
    Vocab dividido em terços: K1 em [1, v/3), K2 em [v/3, 2v/3), valores no resto.
    num_kv_pairs deve ser quadrado perfeito (4, 9, 16, ...).
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    # 3 tokens for context (K1, K2, V) + 3 tokens for query region (K1, K2, gap) per pair
    assert num_kv_pairs * 6 <= input_seq_len, \
        f"Need at least 6 tokens per kv pair. Got {input_seq_len} for {num_kv_pairs} pairs."

    # num_kv_pairs must be a perfect square for grid structure
    grid_size = int(np.sqrt(num_kv_pairs))
    assert grid_size * grid_size == num_kv_pairs, \
        f"num_kv_pairs must be a perfect square (e.g., 4, 9, 16). Got {num_kv_pairs}."

    np.random.seed(seed)

    # Three tokens per "pair": K1, K2, Value
    context_size = num_kv_pairs * 3

    # Split vocabulary: K1s, K2s, and Values from separate pools
    third = vocab_size // 3
    k1_choices = np.arange(1, third)
    k2_choices = np.arange(third, 2 * third)
    value_choices = np.arange(2 * third, vocab_size)

    # For each example, pick grid_size K1s and grid_size K2s
    # Then form all grid_size^2 combinations
    k1s_unshuffled = np.tile(k1_choices, (num_examples, 1))
    k1_set = np.apply_along_axis(
        np.random.choice, 1, k1s_unshuffled, replace=False, size=grid_size
    )

    k2s_unshuffled = np.tile(k2_choices, (num_examples, 1))
    k2_set = np.apply_along_axis(
        np.random.choice, 1, k2s_unshuffled, replace=False, size=grid_size
    )

    # Create grid: all combinations of K1 x K2
    # k1s[i] = [A, A, B, B] if k1_set[i] = [A, B] and grid_size = 2
    # k2s[i] = [X, Y, X, Y] if k2_set[i] = [X, Y] and grid_size = 2
    k1s = np.repeat(k1_set, grid_size, axis=1)  # [A, A, B, B, ...]
    k2s = np.tile(k2_set, (1, grid_size))        # [X, Y, X, Y, ...]

    # Shuffle the order of pairs within each example (so position doesn't give it away)
    for i in range(num_examples):
        perm = np.random.permutation(num_kv_pairs)
        k1s[i] = k1s[i, perm]
        k2s[i] = k2s[i, perm]

    # Assign random unique values to each pair
    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(
        np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs
    )

    # Create context sequences with triplets: K1, K2, V, K1, K2, V, ...
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::3] = k1s
    kvs[:, 1::3] = k2s
    kvs[:, 2::3] = values

    # Compute power law distribution for query placement
    # Each query is 2 tokens (K1, K2), so we divide available space by 3
    # (2 for query + 1 minimum gap)
    space = (input_seq_len - context_size) // 3
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(
        np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs
    )

    # Build query region: place K1, K2 pairs at positions determined by gaps
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 3), values=k1s, axis=1)
    np.put_along_axis(queries, (gaps * 3) + 1, values=k2s, axis=1)

    examples = np.concatenate([kvs, queries], axis=1)

    # Labels: value should be predicted after seeing K2 in query
    # Position is (gaps * 3) + 2 relative to start of query region
    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 3) + context_size + 2, values=values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])

    # Replace all zeros with random values
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]

    return DataSegment(
        inputs,
        labels,
        slices={"num_kv_pairs": num_kv_pairs, "input_seq_len": input_seq_len}
        if include_slices
        else {},
    )


if __name__ == "__main__":
    p = make_parser("compositional_mqar", __doc__)
    p.add_argument("--power-a", type=float, default=0.01)
    p.add_argument("--num-kv-pairs", type=int, default=4, help="quadrado perfeito")
    p.add_argument("--no-random-non-queries", dest="random_non_queries", action="store_false")
    run("compositional_mqar", compositional_mqar, p, ["power_a", "num_kv_pairs", "random_non_queries"])
