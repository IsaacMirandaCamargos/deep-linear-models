"""Exporta amostras + estatísticas de cada dataset de data/ para data_gen/view/view_data.js (lido pelo view.html).

Uso: python data_gen/view/build_view.py [--n 100] [--stats-n 20000]
Depois abra data_gen/view/view.html no navegador.

--n        exemplos por split que dá para navegar no viewer
--stats-n  máximo de exemplos por split usados nas estatísticas (o test inteiro costuma caber)
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MQAR_LIKE = {"mqar", "forgetting_mqar", "compositional_mqar", "ar_extrapolate"}
BITS = {"parity", "majority", "cumulative_parity", "cumulative_majority"}


def br(v, d=1):
    """Número no formato pt-BR (vírgula decimal, ponto de milhar)."""
    return f"{v:,.{d}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def chart(title, note, labels, values, xlabel="", ylabel="", highlight=None, fmt="int"):
    return {"title": title, "note": note, "labels": [str(l) for l in labels], "values": [float(v) for v in values],
            "xlabel": xlabel, "ylabel": ylabel, "highlight": highlight, "fmt": fmt}


def hist(values, lo, hi, bins=None):
    """Histograma: exato quando o intervalo é pequeno, senão em `bins` faixas."""
    values = np.asarray(values)
    if bins is None or hi - lo + 1 <= bins:
        labels = list(range(lo, hi + 1))
        counts = np.bincount(values - lo, minlength=hi - lo + 1)[: hi - lo + 1]
        return labels, counts
    edges = np.linspace(lo, hi + 1, bins + 1).astype(int)
    counts, _ = np.histogram(values, bins=edges)
    return [f"{edges[i]}–{edges[i + 1] - 1}" for i in range(bins)], counts


def context_len(task, cfg):
    kv = cfg.get("num_kv_pairs", 0)
    if task in ("mqar", "ar_extrapolate"):
        return 2 * kv * cfg.get("num_passes", 1)
    if task == "forgetting_mqar":
        return 2 * (kv + cfg["num_updates"])
    if task == "compositional_mqar":
        return 3 * kv
    return 0


def example_cfgs(cfg, slices, n):
    if isinstance(slices, list):
        return [{**cfg, **s} for s in slices]
    return [cfg] * n


def mqar_stats(task, X, Y, cfgs):
    """Distância de recall, distratores e (forgetting) chaves atualizadas."""
    dist, dist_orig, collide_q, ex_collide = [], [], 0, 0
    n_q, upd_q, upd_same = 0, 0, 0
    for x, y, cfg in zip(X, Y, cfgs):
        ctx = context_len(task, cfg)
        L = cfg.get("input_seq_len", len(x))
        q = np.nonzero(y[:L] != -100)[0]
        qpos = set(q.tolist())
        if task == "compositional_mqar":
            qpos |= {i - 1 for i in q}
            keys = set(zip(x[0:ctx:3].tolist(), x[1:ctx:3].tolist()))
            ctx_keys = set(x[0:ctx:3].tolist()) | set(x[1:ctx:3].tolist())
        else:
            ctx_keys = set(x[0:ctx:2].tolist())
        noise = [i for i in range(ctx, L) if i not in qpos]
        noise_tok = x[noise]
        if any(t in ctx_keys for t in noise_tok.tolist()):
            ex_collide += 1
        if task == "forgetting_mqar":
            kv = cfg["num_kv_pairs"]
            orig = dict(zip(x[0:2 * kv:2].tolist(), x[1:2 * kv:2].tolist()))
            for j in range(2 * kv, ctx, 2):
                upd_same += orig.get(int(x[j])) == int(x[j + 1])
        for i in q:
            n_q += 1
            if task == "compositional_mqar":
                hits = [j for j in range(0, ctx, 3) if x[j] == x[i - 1] and x[j + 1] == x[i]]
                if hits:
                    dist.append(i - hits[-1])
                before = set(x[[k for k in noise if k < i]].tolist())
                collide_q += x[i] in before or x[i - 1] in before
            else:
                hits = [j for j in range(0, ctx, 2) if x[j] == x[i]]
                if hits:
                    dist.append(i - hits[-1])
                    dist_orig.append(i - hits[0])
                    if task == "forgetting_mqar" and len(hits) > 1:
                        upd_q += 1
                collide_q += any(x[k] == x[i] for k in noise if k < i)
    out = {"charts": [], "facts": []}
    if dist:
        lab, cnt = hist(dist, 1, max(dist), bins=40)
        out["charts"].append(chart(
            "Distância de recall", "Quantas posições a query está depois da chave (a mais recente) no contexto. "
            "É o quanto o modelo precisa 'olhar para trás'.", lab, cnt, "distância (tokens)", "queries"))
        out["facts"].append({"label": "distância média de recall", "value": br(np.mean(dist)),
                             "note": f"mediana {int(np.median(dist))}, máx {max(dist)}"})
    out["facts"].append({"label": "queries com a chave repetida no ruído antes", "value": br(100 * collide_q / max(n_q, 1), 2) + "%",
                         "note": "o ruído é sorteado do vocab inteiro, então às vezes uma chave reaparece como ruído (distrator)"})
    out["facts"].append({"label": "exemplos com algum ruído igual a uma chave", "value": br(100 * ex_collide / len(X)) + "%",
                         "note": "ruído = posições fora do contexto que não são query"})
    if task == "forgetting_mqar":
        out["facts"].append({"label": "queries sobre chave atualizada", "value": br(100 * upd_q / max(n_q, 1)) + "%",
                             "note": "nessas a resposta é o valor do update, não o original"})
        out["facts"].append({"label": "updates com valor igual ao original", "value": str(upd_same),
                             "note": "o gerador evita repetir original_values[j], mas a chave atualizada é update_indices[j]; então pode ocorrer (raro)"})
        if dist_orig:
            lab, cnt = hist(dist_orig, 1, max(dist_orig), bins=40)
            out["charts"].append(chart(
                "Distância até a 1ª ocorrência da chave", "Para chaves atualizadas, a 1ª ocorrência tem o valor ERRADO (antigo). "
                "Compare com a distância de recall acima.", lab, cnt, "distância (tokens)", "queries"))
    return out


def bits_stats(task, X, Y):
    out = {"charts": [], "facts": []}
    if task in ("parity", "majority"):
        bits = X[:, :-1]
        ones = bits.sum(1)
        L = bits.shape[1]
        if task == "parity":
            lab, cnt = hist(ones, 0, L)
            out["charts"].append(chart("Número de uns por exemplo", "Binomial em torno de L/2. Ímpar → label 1.",
                                       lab, cnt, "#uns", "exemplos"))
        else:
            margin = 2 * ones - L
            lab, cnt = hist(margin, -L, L)
            keep = [i for i, c in enumerate(cnt) if c] or [0]
            lab, cnt = lab[keep[0]:keep[-1] + 1], cnt[keep[0]:keep[-1] + 1]
            out["charts"].append(chart("Margem: #uns − #zeros", "Perto de 0 = exemplo difícil (quase empate). "
                                       f"Com {L} bits (ímpar) nunca há empate.", lab, cnt, "#uns − #zeros", "exemplos"))
            out["facts"].append({"label": "exemplos com |margem| ≤ 3", "value": br(100 * np.mean(np.abs(margin) <= 3)) + "%",
                                 "note": "um erro de contagem de 1–2 bits troca a resposta"})
    else:
        lab = list(range(Y.shape[1]))
        p1 = (Y == 1).mean(0)
        out["charts"].append(chart("P(label = 1) por posição", "Fração de exemplos com label 1 em cada posição. "
                                   + ("Empate (uns = zeros) vira 1, por isso as posições ímpares (comprimento par) puxam para cima."
                                      if task == "cumulative_majority" else "Deve ficar perto de 50% em todas."),
                                   lab, p1, "posição", "P(1)", fmt="pct"))
    return out


def vocab_majority_stats(X, Y):
    seq = X[:, :-1]
    V = int(seq.max()) + 1
    margins = []
    for row in seq:
        c = sorted(np.bincount(row, minlength=V)[1:], reverse=True)
        margins.append(c[0] - c[1])
    lab, cnt = hist(np.array(margins), 0, max(margins))
    out = {"charts": [chart("Margem: contagem do 1º − do 2º token", "Margem 1 = quase empate. O gerador desfaz empates "
                            "trocando uma ocorrência de outro token pelo menor token empatado.", lab, cnt, "margem", "exemplos")],
           "facts": [{"label": "exemplos com margem 1", "value": br(100 * np.mean(np.array(margins) == 1)) + "%",
                      "note": "inclui os empates que o gerador desfez artificialmente"}]}
    lv = Y[:, -1]
    out["facts"].append({"label": f"label = 1 (menor token)", "value": br(100 * np.mean(lv == 1)) + "%",
                         "note": f"uniforme seria {br(100 / (V - 1))}%: o desempate pelo menor token enviesa para tokens baixos"})
    return out


def split_stats(task, cfg, d, n_max):
    total = len(d["inputs"])
    # amostra espaçada: no ar_extrapolate os exemplos estão ordenados por config
    ids = np.arange(total) if total <= n_max else np.arange(n_max) * total // n_max
    X = d["inputs"][ids].numpy()
    Y = d["labels"][ids].numpy()
    sl = d.get("slices")
    cfgs = example_cfgs(cfg, [sl[i] for i in ids] if isinstance(sl, list) else sl, len(X))
    sup = Y != -100
    sup_n = sup.sum(1)
    lv = Y[sup]
    vocab = int(cfg.get("vocab_size", int(X.max()) + 1))
    big = vocab > 64
    s = {"n": int(len(X)), "seq_len": int(X.shape[1]),
         "sup_mean": float(sup_n.mean()), "sup_min": int(sup_n.min()), "sup_max": int(sup_n.max()),
         "tok_min": int(X.min()), "tok_max": int(X.max()), "n_tokens_used": int(len(np.unique(X))),
         "label_distinct": int(len(np.unique(lv)))}
    top = Counter(lv.tolist()).most_common(1)[0]
    s["baseline_major"] = top[1] / len(lv)
    s["baseline_major_label"] = int(top[0])
    charts = [chart("Onde o modelo é cobrado", "Fração dos exemplos que têm label (≠ −100) em cada posição.",
                    list(range(X.shape[1])), sup.mean(0), "posição", "fração", fmt="pct")]
    lens = np.array([c.get("input_seq_len", X.shape[1]) if isinstance(sl, list) else X.shape[1] for c in cfgs])
    valid = np.arange(X.shape[1])[None, :] < lens[:, None]  # sem padding
    lab, cnt = hist(X[valid], 0, max(vocab - 1, int(X.max())), bins=64 if big else None)
    charts.append(chart("Tokens de entrada", "Histograma de todos os tokens dos inputs" +
                        (" (em faixas do vocab). Chaves, valores e ruído vêm de faixas diferentes." if big else "."),
                        lab, cnt, "token", "ocorrências"))
    lab, cnt = hist(lv, 0, max(vocab - 1, int(lv.max())), bins=64 if big else None)
    charts.append(chart("Labels (respostas)", "Distribuição dos valores-alvo nas posições cobradas.",
                        lab, cnt, "label", "ocorrências", highlight=None if big else s["baseline_major_label"]))
    extra = {"charts": [], "facts": []}
    if task in MQAR_LIKE:
        extra = mqar_stats(task, X, Y, cfgs)
    elif task in BITS:
        extra = bits_stats(task, X, Y)
    elif task == "vocab_majority":
        extra = vocab_majority_stats(X, Y)
    if isinstance(sl, list):
        c = Counter(f"len {e['input_seq_len']} · {e['num_kv_pairs']} pares" for e in sl)
        s["configs"] = list(c)
        extra["charts"].insert(0, chart("Configs (seq_len · pares)", "Quantos exemplos de cada configuração. "
                                        "Train e test usam configs diferentes.", list(c), list(c.values()), "", "exemplos"))
    s["charts"] = charts + extra["charts"]
    s["facts"] = extra["facts"]
    return s


def overlap(train, test):
    tr = {r.tobytes() for r in train}
    te = [r.tobytes() for r in test]
    return {"test_in_train": sum(r in tr for r in te), "train_unique": len(tr),
            "test_unique": len(set(te))}


def warnings_for(task, cfg, item):
    w = []
    S = item["splits"]
    ov = item.get("overlap")
    if ov and ov["test_in_train"]:
        pct = 100 * ov["test_in_train"] / S["test"]["total"]
        w.append(f"<b>{br(pct, 0)}% do test aparece idêntico no train</b> ({br(ov['test_in_train'], 0)} de {br(S['test']['total'], 0)}). "
                 f"Só existem {br(S['test']['stats']['n_tokens_used'] ** S['test']['stats']['seq_len'], 0)} sequências possíveis; "
                 "a acurácia no test mede memorização também.")
    if ov and ov["train_unique"] < S["train"]["total"]:
        w.append(f"O train tem {br(S['train']['total'] - ov['train_unique'], 0)} exemplos duplicados.")
    st = S.get("test", S.get("train"))["stats"]
    if "vocab_size" in cfg and st["tok_max"] < cfg["vocab_size"] - 1 and cfg["vocab_size"] <= 64:
        w.append(f"vocab_size={cfg['vocab_size']}, mas os inputs só usam tokens {st['tok_min']}..{st['tok_max']}.")
    if task in ("parity", "majority", "vocab_majority"):
        w.append("Só 1 posição por exemplo tem label: o sinal de treino é bem esparso.")
    return w


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100, help="exemplos por split")
    p.add_argument("--stats-n", type=int, default=20000, help="exemplos por split nas estatísticas")
    args = p.parse_args()
    n = args.n

    out = []
    for cfg_path in sorted((ROOT / "data").glob("*/config.json")):
        folder = cfg_path.parent
        cfg = json.loads(cfg_path.read_text())
        task = cfg["name"]
        item = {"folder": folder.name, "task": task, "config": cfg, "splits": {}}
        raw = {}
        for split in ("train", "test"):
            f = folder / f"{split}.pt"
            if not f.exists():
                continue
            d = torch.load(f, weights_only=False)
            raw[split] = d
            total = len(d["inputs"])
            sl = d.get("slices")
            # slices por exemplo (ar_extrapolate): amostra espaçada para pegar todas as configs
            per_example = isinstance(sl, list)
            ids = [i * total // n for i in range(n)] if per_example and total > n else list(range(min(n, total)))
            item["splits"][split] = {
                "total": total,
                "shape": list(d["inputs"].shape),
                "inputs": d["inputs"][ids].tolist(),
                "labels": d["labels"][ids].tolist(),
                "stats": split_stats(task, cfg, d, args.stats_n),
            }
            if per_example:
                item["splits"][split]["slices"] = [sl[i] for i in ids]
                item["splits"][split]["ids"] = ids
            print(f"  {folder.name}/{split}: {total} exemplos")
        if "train" in raw and "test" in raw and raw["train"]["inputs"].shape[1] == raw["test"]["inputs"].shape[1]:
            item["overlap"] = overlap(raw["train"]["inputs"].numpy(), raw["test"]["inputs"].numpy())
        item["warnings"] = warnings_for(task, cfg, item)
        out.append(item)

    dest = Path(__file__).resolve().parent / "view_data.js"
    dest.write_text("window.VIEW_DATA = " + json.dumps(out) + ";\n")
    print(f"{len(out)} datasets -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
