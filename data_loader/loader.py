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
