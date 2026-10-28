"""Avalia os checkpoints de uma receita (os mesmos que train.py gerou).

Uso: python eval.py recipes/train/rnn.yaml

Lê checkpoints/<name>/*.pt (epoch_*.pt, best.pt, last.pt) e avalia cada um no dataset de treino
e nos `eval_datasets` da receita. Escreve checkpoints/<name>/eval.csv (resumo) e eval.json
(com acurácia por slice quando o dataset tem, como o ar_extrapolate).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data_loader import load_config, load_split
from model import build_model

ROOT = Path(__file__).resolve().parent


def pick_device(cfg: dict) -> str:
    if cfg["device"] == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg["device"]


def run_dir(cfg: dict) -> Path:
    return ROOT / cfg["checkpoint"]["dir"] / cfg["name"]


@torch.no_grad()
def per_example(model, ds, device, batch_size):
    """Retorna (loss_sum, hits, n) por exemplo, na ordem do dataset."""
    model.eval()
    loss_sum, hits, count = [], [], []
    for x, y in DataLoader(ds, batch_size=batch_size, shuffle=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        ce = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100, reduction="none")
        mask = y != -100
        loss_sum.append(ce.sum(1).cpu())
        hits.append(((logits.argmax(-1) == y) & mask).sum(1).cpu())
        count.append(mask.sum(1).cpu())
    return torch.cat(loss_sum), torch.cat(hits), torch.cat(count)


def evaluate_model(model, ds, device, batch_size=256) -> dict:
    """loss e acc nas posições com label != -100; `by_slice` quando ds.slices é uma lista."""
    loss_sum, hits, count = per_example(model, ds, device, batch_size)
    total = max(1, count.sum().item())
    result = {"loss": loss_sum.sum().item() / total, "acc": hits.sum().item() / total,
              "examples": len(ds), "seq_len": ds.inputs.shape[1]}
    if isinstance(ds.slices, list):
        groups = defaultdict(list)
        for i, s in enumerate(ds.slices):
            groups[tuple(sorted(s.items()))].append(i)
        result["by_slice"] = []
        for key, idx in sorted(groups.items()):
            idx = torch.tensor(idx)
            result["by_slice"].append({"slice": dict(key), "examples": len(idx),
                                       "acc": hits[idx].sum().item() / max(1, count[idx].sum().item())})
    return result


def eval_targets(cfg: dict) -> list[tuple[str, str]]:
    """(dataset, split) a avaliar: o do treino primeiro, depois os `eval_datasets`."""
    split = cfg["dataset"]["eval_split"]
    names = [cfg["dataset"]["name"]] + [n for n in cfg.get("eval_datasets", []) if n != cfg["dataset"]["name"]]
    return [(n, split) for n in names]


def eval_checkpoints(cfg: dict, paths: list[Path], batch_size=256, verbose=True) -> list[dict]:
    """Avalia cada checkpoint em todos os eval_targets; escreve eval.csv e eval.json na pasta da run."""
    device = pick_device(cfg)
    datasets = {}
    rows = []
    for path in paths:
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        model = build_model(ckpt["cfg"]["model"], ckpt["vocab_size"]).to(device)
        model.load_state_dict(ckpt["model"])
        for name, split in eval_targets(cfg):
            vocab = load_config(name)["vocab_size"]
            if vocab != ckpt["vocab_size"]:
                if verbose:
                    print(f"  pulando {name}: vocab_size {vocab} != {ckpt['vocab_size']} do modelo")
                continue
            if (name, split) not in datasets:
                datasets[(name, split)] = load_split(name, split)
            res = evaluate_model(model, datasets[(name, split)], device, batch_size)
            rows.append({"checkpoint": path.name, "epoch": ckpt["epoch"], "dataset": name, "split": split, **res})
            if verbose:
                print(f"{path.name:<16} epoch {ckpt['epoch']:>3} | {name}/{split} T={res['seq_len']} | "
                      f"loss {res['loss']:.4f} | acc {res['acc']:.4f}")
                for s in res.get("by_slice", []):
                    label = ", ".join(f"{k}={v}" for k, v in s["slice"].items())
                    print(f"{'':<16}   {label:<34} n={s['examples']:<5} acc {s['acc']:.4f}")

    out = run_dir(cfg)
    cols = ["checkpoint", "epoch", "dataset", "split", "examples", "seq_len", "loss", "acc"]
    with open(out / "eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "eval.json").write_text(json.dumps(rows, indent=2))
    return rows


def list_checkpoints(cfg: dict) -> list[Path]:
    d = run_dir(cfg)
    return sorted(d.glob("epoch_*.pt")) + [p for p in (d / "best.pt", d / "last.pt") if p.exists()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("recipe", type=Path)
    a = p.parse_args()

    cfg = yaml.safe_load(a.recipe.read_text())
    paths = list_checkpoints(cfg)
    if not paths:
        raise SystemExit(f"nenhum checkpoint em {run_dir(cfg)} (rode train.py primeiro)")
    print(f"{cfg['name']}: {len(paths)} checkpoints em {run_dir(cfg)}")
    eval_checkpoints(cfg, paths)
    print(f"\nsalvo em {run_dir(cfg) / 'eval.csv'} e eval.json")


if __name__ == "__main__":
    main()
