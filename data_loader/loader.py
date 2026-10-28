"""Carrega os train.pt/test.pt gerados por gen.py como Dataset/DataLoader do PyTorch."""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class SegmentDataset(Dataset):
    """Cada item é (inputs, labels); labels == -100 são ignorados na loss."""

    def __init__(self, path: Path):
        seg = torch.load(path, weights_only=False)
        self.inputs = seg["inputs"].long()
        self.labels = seg["labels"].long()
        self.slices = seg["slices"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.labels[i]


def load_split(name: str, split: str) -> SegmentDataset:
    return SegmentDataset(DATA_DIR / name / f"{split}.pt")


def load_config(name: str) -> dict:
    return json.loads((DATA_DIR / name / "config.json").read_text())


def make_loader(name: str, split: str, batch_size: int = 64, shuffle=None, **kw) -> DataLoader:
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(load_split(name, split), batch_size=batch_size, shuffle=shuffle, **kw)


if __name__ == "__main__":
    import argparse
    import time

    p = argparse.ArgumentParser(description="Mede batches/s do DataLoader")
    p.add_argument("name", nargs="?", help="dataset (default: todos em data/)")
    p.add_argument("--split", default="train")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    a = p.parse_args()

    names = [a.name] if a.name else sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    print(f"{'dataset':<22}{'exemplos':>9}{'batches':>9}{'batch/s':>10}{'exemplos/s':>12}")
    for n in names:
        loader = make_loader(n, a.split, a.batch_size, num_workers=a.num_workers)
        n_batches = 0
        t0 = time.perf_counter()
        for _ in range(a.epochs):
            for _ in loader:
                n_batches += 1
        dt = time.perf_counter() - t0
        print(f"{n:<22}{len(loader.dataset):>9}{n_batches // a.epochs:>9}"
              f"{n_batches / dt:>10.0f}{len(loader.dataset) * a.epochs / dt:>12.0f}")
