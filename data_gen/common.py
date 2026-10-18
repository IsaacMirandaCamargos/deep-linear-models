"""Utilitários compartilhados pelos geradores de dataset.

Replica o que o zoology faz em `zoology/data/utils.py` (DataSegment, seeds de
train/test e salvamento com `torch.save(asdict(segment))`), sem depender dele.
"""
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class DataSegment:
    inputs: torch.Tensor
    labels: torch.Tensor
    slices: Dict[str, any] = None

    def __len__(self):
        assert len(self.inputs) == len(self.labels)
        return len(self.inputs)


def split_seeds(seed: int) -> Tuple[int, int]:
    """Mesma lógica de `prepare_data`: seeds de train e test nunca se sobrepõem."""
    MAX_SEED = 2 ** 32
    np.random.seed(seed)
    train_seed = int(np.random.randint(0, MAX_SEED // 2, size=1)[0])
    test_seed = int(np.random.randint(MAX_SEED // 2, MAX_SEED, size=1)[0])
    return train_seed, test_seed


def save_segment(segment: DataSegment, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(segment), path)
    print(f"saved {path}  inputs={tuple(segment.inputs.shape)} labels={tuple(segment.labels.shape)}")


def make_parser(name: str, doc: str, *, vocab_size=8_192, input_seq_len=64,
                num_train=100_000, num_test=3_000) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DATA_DIR / name)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--vocab-size", type=int, default=vocab_size)
    p.add_argument("--input-seq-len", type=int, default=input_seq_len)
    p.add_argument("--num-train", type=int, default=num_train)
    p.add_argument("--num-test", type=int, default=num_test)
    return p


def run(name: str, build, parser: argparse.ArgumentParser, task_args=()):
    """Gera train.pt e test.pt em `out_dir`. `task_args` = nomes dos args específicos da tarefa."""
    args = parser.parse_args()
    train_seed, test_seed = split_seeds(args.seed)
    task_kwargs = {k: getattr(args, k) for k in task_args}
    common = dict(vocab_size=args.vocab_size, input_seq_len=args.input_seq_len, **task_kwargs)

    for split, n, seed in [("train", args.num_train, train_seed), ("test", args.num_test, test_seed)]:
        segment = build(num_examples=n, seed=seed, **common)
        save_segment(segment, args.out_dir / f"{split}.pt")

    (args.out_dir / "config.json").write_text(json.dumps(
        {"name": name, **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, indent=2))
