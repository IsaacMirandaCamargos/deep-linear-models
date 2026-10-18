"""AR extrapolate. Ideia de zoology/data/ar_extrapolate.py (o original está quebrado: chama `_mqar`,
que não existe). Aqui: mistura de configs MQAR com tamanhos diferentes, padded até o maior.

Cada config é "seq_len:num_kv_pairs". Os exemplos são divididos igualmente entre as configs.
Padding: input 0 e label -100 no fim. `slices` guarda a config de cada exemplo.
Típico: treinar em sequências curtas e testar em mais longas.

Uso: python data_gen/zoology/ar_extrapolate.py --train-configs 32:4,64:8 --test-configs 64:8,128:16
Saída: data/ar_extrapolate/{train,test}.pt + config.json
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DataSegment, make_parser, run
from mqar import multiquery_ar


def parse(spec: str):
    return [tuple(int(v) for v in c.split(":")) for c in spec.split(",")]


def ar_extrapolate(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    configs: str,
    **kwargs
) -> DataSegment:
    cfgs = parse(configs)
    max_len = max(c[0] for c in cfgs)
    per = [num_examples // len(cfgs)] * len(cfgs)
    per[-1] += num_examples - sum(per)

    inputs, labels, slices = [], [], []
    for idx, ((seq_len, kv), n) in enumerate(zip(cfgs, per)):
        seg = multiquery_ar(vocab_size=vocab_size, num_examples=n, input_seq_len=seq_len,
                            num_kv_pairs=kv, seed=seed + idx)
        pad = max_len - seq_len
        inputs.append(F.pad(seg.inputs, (0, pad), value=0))
        labels.append(F.pad(seg.labels, (0, pad), value=-100))
        slices += [{"input_seq_len": seq_len, "num_kv_pairs": kv}] * n
    return DataSegment(torch.cat(inputs), torch.cat(labels), slices=slices)


if __name__ == "__main__":
    p = make_parser("ar_extrapolate", __doc__)
    p.add_argument("--train-configs", default="32:4,64:8")
    p.add_argument("--test-configs", default="64:8,128:16")
    args, _ = p.parse_known_args()
    # train e test usam configs diferentes: o 1º build é train, o 2º é test
    state = {"n": 0}

    def build(**kw):
        cfg = args.train_configs if state["n"] == 0 else args.test_configs
        state["n"] += 1
        return ar_extrapolate(configs=cfg, **kw)

    run("ar_extrapolate", build, p, [])
