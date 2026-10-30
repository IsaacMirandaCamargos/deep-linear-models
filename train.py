"""Treina um modelo a partir de uma receita yaml.

Uso: python train.py recipes/train/rnn.yaml

Escreve em checkpoints/<name>/: epoch_NNN.pt (a cada save_every épocas), best.pt, last.pt,
recipe.yaml e train.csv. Com checkpoint.resume: true retoma do last.pt.
"""
import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from torch.utils.data import DataLoader

from data_loader import load_config, load_split
from eval import eval_checkpoints, evaluate_model, list_checkpoints, pick_device, run_dir
from model import build_model


def build_optimizer(params, cfg: dict):
    kind, lr, wd = cfg["type"], cfg["lr"], cfg.get("weight_decay", 0.0)
    if kind == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if kind == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"optimizer desconhecido: {kind}")


def lr_factor(step: int, total: int, warmup: int, kind: str, min_ratio: float = 0.0) -> float:
    """Multiplicador do lr: warmup linear e depois cosine até min_ratio (= min_lr / lr) ou constante."""
    if step < warmup:
        return (step + 1) / warmup
    if kind == "constant":
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def batch_size_at(ds_cfg: dict, epoch: int) -> int:
    """batch_size da época: o último item de `batch_size_schedule` com from_epoch <= epoch, senão `batch_size`."""
    size = ds_cfg["batch_size"]
    for item in sorted(ds_cfg.get("batch_size_schedule", []), key=lambda i: i["from_epoch"]):
        if item["from_epoch"] <= epoch:
            size = item["batch_size"]
    return size


def loss_and_hits(logits, labels):
    """Loss média e (acertos, total) nas posições com label != -100."""
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
    mask = labels != -100
    hits = ((logits.argmax(-1) == labels) & mask).sum().item()
    return loss, hits, mask.sum().item()


def save_checkpoint(path, model, opt, sched, cfg, vocab_size, epoch, step, best_acc, eval_acc):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "cfg": cfg, "vocab_size": vocab_size, "epoch": epoch, "step": step,
                "best_acc": best_acc, "eval_acc": eval_acc}, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("recipe", type=Path)
    a = p.parse_args()

    cfg = yaml.safe_load(a.recipe.read_text())
    torch.manual_seed(cfg["seed"])
    device = pick_device(cfg)

    ds = cfg["dataset"]
    vocab_size = load_config(ds["name"])["vocab_size"]
    train_ds = load_split(ds["name"], ds["train_split"])

    def make_train_loader(epoch):
        return DataLoader(train_ds, batch_size=batch_size_at(ds, epoch), shuffle=True,
                          num_workers=ds.get("num_workers", 0))
    eval_ds = load_split(ds["name"], ds["eval_split"])

    model = build_model(cfg["model"], vocab_size).to(device)
    opt = build_optimizer(model.parameters(), cfg["optimizer"])
    tr, ck = cfg["train"], cfg["checkpoint"]
    total_steps = sum(math.ceil(len(train_ds) / batch_size_at(ds, e)) for e in range(1, tr["epochs"] + 1))
    sc = cfg["scheduler"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, total_steps, sc.get("warmup_steps", 0), sc["type"],
                                 sc.get("min_lr", 0.0) / cfg["optimizer"]["lr"]))
    clip = cfg["optimizer"].get("grad_clip")

    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "recipe.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    start_epoch, step, best_acc = 1, 0, -1.0
    last = out / "last.pt"
    if ck.get("resume") and last.exists():
        state = torch.load(last, weights_only=False, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch, step, best_acc = state["epoch"] + 1, state["step"], state["best_acc"]
        print(f"retomando de {last} (epoca {state['epoch']})")
    elif not ck.get("resume"):
        for old in list(out.glob("*.pt")) + [out / "train.csv"]:
            old.unlink(missing_ok=True)

    wb = None
    if cfg.get("logging", {}).get("wandb"):
        import wandb
        wb = wandb.init(project=cfg["logging"].get("project", "isaac"), name=cfg["name"], config=cfg)

    csv_cols = ["epoch", "step", "train_loss", "train_acc", "eval_loss", "eval_acc", "lr", "seconds"]
    csv_path = out / "train.csv"
    new_csv = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    log = csv.DictWriter(csv_file, fieldnames=csv_cols)
    if new_csv:
        log.writeheader()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg['name']} | {ds['name']} vocab={vocab_size} | {n_params:,} params | {device} | {total_steps} steps")

    t0 = time.perf_counter()
    train_loader, bs = None, None
    for epoch in range(start_epoch, tr["epochs"] + 1):
        if batch_size_at(ds, epoch) != bs:
            bs = batch_size_at(ds, epoch)
            train_loader = make_train_loader(epoch)
            print(f"epoca {epoch}: batch_size {bs} ({len(train_loader)} steps/epoca)")
        model.train()
        win_loss = win_hits = win_n = 0
        ep_loss = ep_hits = ep_n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss, h, n = loss_and_hits(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            sched.step()
            step += 1
            win_loss += loss.item() * n
            win_hits += h
            win_n += n
            ep_loss += loss.item() * n
            ep_hits += h
            ep_n += n
            if step % tr["log_every"] == 0:
                print(f"  step {step:>6} | loss {win_loss / max(1, win_n):.4f} | acc {win_hits / max(1, win_n):.4f}"
                      f" | lr {sched.get_last_lr()[0]:.2e}")
                win_loss = win_hits = win_n = 0

        row = {"epoch": epoch, "step": step, "train_loss": ep_loss / max(1, ep_n), "train_acc": ep_hits / max(1, ep_n),
               "lr": sched.get_last_lr()[0], "seconds": round(time.perf_counter() - t0)}
        ev_acc, mark = None, ""
        if epoch % tr["eval_every"] == 0 or epoch == tr["epochs"]:
            ev = evaluate_model(model, eval_ds, device)
            ev_acc = ev["acc"]
            row.update(eval_loss=ev["loss"], eval_acc=ev_acc)
            if ev_acc > best_acc:
                best_acc, mark = ev_acc, " *"
                save_checkpoint(out / "best.pt", model, opt, sched, cfg, vocab_size, epoch, step, best_acc, ev_acc)
            print(f"epoch {epoch:>3}/{tr['epochs']} | train loss {row['train_loss']:.4f} | eval loss {ev['loss']:.4f}"
                  f" | eval acc {ev_acc:.4f} | {row['seconds']}s{mark}")

        save_checkpoint(last, model, opt, sched, cfg, vocab_size, epoch, step, best_acc, ev_acc)
        if epoch % ck["save_every"] == 0 or epoch == tr["epochs"]:
            save_checkpoint(out / f"epoch_{epoch:03d}.pt", model, opt, sched, cfg, vocab_size, epoch, step, best_acc, ev_acc)
            for old in sorted(out.glob("epoch_*.pt"))[:-ck["keep_last"]]:
                old.unlink()
        log.writerow(row)
        csv_file.flush()
        if wb:
            wb.log({k: v for k, v in row.items() if v is not None}, step=step)
    csv_file.close()
    if wb:
        wb.finish()

    print(f"\nmelhor eval acc: {best_acc:.4f}  ->  {out}\navaliando checkpoints (eval.py):")
    eval_checkpoints(cfg, list_checkpoints(cfg))


if __name__ == "__main__":
    main()
