"""MQAR com esquecimento (algumas chaves são reatribuídas). Código de zoology/data/forgetting_mqar.py.

Uso: python data_gen/zoology/forgetting_mqar.py --num-kv-pairs 8 --num-updates 4
Saída: data/forgetting_mqar/{train,test}.pt + config.json
"""
import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DataSegment, make_parser, run


def forgetting_mqar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    num_updates: int = 4,
    random_non_queries: bool = True,
    include_slices: bool = True,
    **kwargs
) -> DataSegment:
    """
    Algumas chaves aparecem duas vezes com valores diferentes; a resposta é o valor MAIS RECENTE.
    Contexto: [KVs originais] [KVs de update], depois as queries.
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_updates <= num_kv_pairs, "num_updates must be <= num_kv_pairs"

    # Total KV slots: original pairs + updates
    total_kv_slots = num_kv_pairs + num_updates
    # 2 tokens per KV slot in context, 2 tokens per query (key + gap)
    assert total_kv_slots * 2 + num_kv_pairs * 2 <= input_seq_len, \
        f"Need at least {total_kv_slots * 2 + num_kv_pairs * 2} tokens. Got {input_seq_len}."

    np.random.seed(seed)

    # Context size: original KVs + updated KVs
    context_size = total_kv_slots * 2

    # Split vocabulary
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    # Generate unique keys for each example
    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(
        np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs
    )

    # Generate original values (one per key)
    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    original_values = np.apply_along_axis(
        np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs
    )

    # Select which keys get updated (first num_updates keys, then we'll shuffle)
    # Generate new values for updated keys (must be different from original)
    updated_values = np.zeros((num_examples, num_updates), dtype=np.int64)
    for i in range(num_examples):
        for j in range(num_updates):
            # Pick a value different from the original
            available = value_choices[value_choices != original_values[i, j]]
            updated_values[i, j] = np.random.choice(available)

    # For each example, randomly select which keys get updated
    update_indices = np.stack([
        np.random.choice(num_kv_pairs, size=num_updates, replace=False)
        for _ in range(num_examples)
    ])

    # Get the keys and their new values for updates
    update_keys = np.take_along_axis(keys, update_indices, axis=1)

    # Final values: start with original, then overwrite updated ones
    final_values = original_values.copy()
    for i in range(num_examples):
        for j, idx in enumerate(update_indices[i]):
            final_values[i, idx] = updated_values[i, j]

    # Build context sequence
    # Structure: [original KVs] [update KVs]
    original_context = np.zeros((num_examples, num_kv_pairs * 2), dtype=np.int64)
    original_context[:, 0::2] = keys
    original_context[:, 1::2] = original_values

    update_context = np.zeros((num_examples, num_updates * 2), dtype=np.int64)
    update_context[:, 0::2] = update_keys
    update_context[:, 1::2] = updated_values

    kvs = np.concatenate([original_context, update_context], axis=1)

    # Compute power law distribution for query placement
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(
        np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs
    )

    # Build query region
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)

    examples = np.concatenate([kvs, queries], axis=1)

    # Labels: should be the FINAL value (after updates)
    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=final_values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])

    # Replace zeros with random values
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]

    return DataSegment(
        inputs,
        labels,
        slices={
            "num_kv_pairs": num_kv_pairs,
            "num_updates": num_updates,
            "input_seq_len": input_seq_len,
        }
        if include_slices
        else {},
    )


if __name__ == "__main__":
    p = make_parser("forgetting_mqar", __doc__)
    p.add_argument("--power-a", type=float, default=0.01)
    p.add_argument("--num-kv-pairs", type=int, default=8)
    p.add_argument("--num-updates", type=int, default=4)
    p.add_argument("--no-random-non-queries", dest="random_non_queries", action="store_false")
    run("forgetting_mqar", forgetting_mqar, p, ["power_a", "num_kv_pairs", "num_updates", "random_non_queries"])
