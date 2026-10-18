"""Gera dados a partir de uma receita yaml.

Uso: python gen.py recipes/data_gen.yaml [--only NAME ...]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def to_flags(args: dict) -> list:
    flags = []
    for key, value in (args or {}).items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            flags.append(flag)
        elif value is False or value is None:
            continue
        else:
            flags += [flag, str(value)]
    return flags


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recipe", type=Path, help="caminho do data_gen.yaml")
    p.add_argument("--only", nargs="*", help="gera só estes names")
    opts = p.parse_args()

    cfg = yaml.safe_load(opts.recipe.read_text())
    for ds in cfg["datasets"]:
        if opts.only and ds["name"] not in opts.only:
            continue
        script = ROOT / "data_gen" / "zoology" / f"{ds['task']}.py"
        if not script.exists():
            sys.exit(f"task desconhecida: {ds['task']} ({script})")
        args = dict(ds.get("args") or {})
        args.setdefault("out_dir", ROOT / "data" / ds["name"])
        if ds.get("out_dir"):
            args["out_dir"] = ROOT / ds["out_dir"]
        cmd = [sys.executable, str(script), *to_flags(args)]
        print(f"==> {ds['name']}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
