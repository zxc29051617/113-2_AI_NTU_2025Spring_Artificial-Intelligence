"""Candidate selection, chance-corrected beat alignment, and the result tables.

python make_tables.py --exp a    # text + music -> dance (input: AIST music)
python make_tables.py --exp b    # text -> music + dance
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from judges import GTZAN
from metrics import avg_pairwise_distance, bas, fid, normalize_with
from motion_style import GENRES

KS = [1, 2, 4, 8, 16]
EXP = {
    "a": {"run": "runs/exp_a", "jobs": "configs/exp_a_music2dance.json"},
    "b": {"run": "runs/exp_b", "jobs": "configs/exp_b_text2musicdance.json"},
}


def chance(db, others, kind):
    """Mean BAS of the motion beats db against the music beats of the other clips."""
    return float(np.mean([bas(o[kind], db) for o in others]))


def load_rows(exp):
    cfg = EXP[exp]
    jobs = {j["id"]: j for j in json.load(open(cfg["jobs"]))["jobs"]}
    res = pickle.load(open(os.path.join(cfg["run"], "scores.pkl"), "rb"))
    rows, kin = [], {}
    for (jid, seed), out in res.items():
        others_by_cand = {}
        for r in out["rows"]:
            c, v = r["cand"], r["variant"]
            if c not in others_by_cand:
                others_by_cand[c] = [res[(j2, s2)]["music_beats"][c] for (j2, s2) in res
                                     if s2 == seed and j2 != jid]
            db = out["motion_beats"][(c, v)]
            r = {k: val for k, val in r.items() if k != "style_probs"} | {"style_probs": r["style_probs"]}
            r["dbas_beat"] = r["bas_beat"] - chance(db, others_by_cand[c], "beat")
            r["dbas_mm"] = r["bas_mm"] - chance(db, others_by_cand[c], "mm")
            tgt = GENRES.index(jobs[jid]["genre"])
            r["style_acc"] = float(np.argmax(r["style_probs"]) == tgt)
            r["style_p"] = float(r["style_probs"][tgt])
            rows.append(r)
            kin[(jid, seed, c, v)] = out["kin"][(c, v)]
    df = pd.DataFrame(rows).drop(columns=["style_probs"])
    if exp == "b":
        df = add_audio(df, cfg, jobs)
    return df, kin, jobs


def add_audio(df, cfg, jobs):
    au = pickle.load(open(os.path.join(cfg["run"], "audio_scores.pkl"), "rb"))
    tidx = {t: i for i, t in enumerate(au["texts"])}
    extra = []
    for (jid, seed) in au["keys"]:
        ti = tidx[jobs[jid]["music_desc"]]
        g = GTZAN.index(jobs[jid]["gtzan_genre"])
        p = au["gtzan_probs"][(jid, seed)]
        for c in range(len(p)):
            extra.append({
                "job": jid, "seed": seed, "cand": c,
                "clap": max(0.0, float(au["clap_orig_text"][ti] @ au["clap_orig_audio"][(jid, seed)][c])),
                "clap_sel": float(au["clap_sel_text"][ti] @ au["clap_sel_audio"][(jid, seed)][c]),
                "gtzan_acc": float(np.argmax(p[c]) == g), "gtzan_p": float(p[c, g])})
    return df.merge(pd.DataFrame(extra), on=["job", "seed", "cand"], how="left")


def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9) if len(x) > 1 else np.zeros_like(x)


SELECTORS = {
    "bas": lambda g: zscore(g["bas_beat"]),
    "clap": lambda g: zscore(g["clap_sel"]),
    "clap+bas": lambda g: zscore(g["clap_sel"]) + zscore(g["bas_beat"]),
}


def select(df, selector, k):
    """Per (job, seed): index of the chosen candidate among the first k (scored on raw motion)."""
    raw = df[(df.variant == "raw") & (df.cand < k)]
    chosen = {}
    for (jid, seed), g in raw.groupby(["job", "seed"]):
        g = g.sort_values("cand")
        chosen[(jid, seed)] = int(g.cand.values[int(np.argmax(SELECTORS[selector](g)))]) if k > 1 else 0
    return chosen


def pick(df, chosen, variant):
    idx = df.set_index(["job", "seed", "cand", "variant"])
    keys = [(j, s, c, variant) for (j, s), c in sorted(chosen.items())]
    missing = [k for k in keys if k not in idx.index]
    if missing:
        print(f"warning: {len(missing)} selections have no '{variant}' row")
    return idx.loc[[k for k in keys if k not in missing]].reset_index()


def set_metrics(sel, kin, real_kin, variant):
    """Dist_k per seed (mean, sd) and FID_k pooled over seeds, kinetic features normalised by real data."""
    feats = {s: np.stack([kin[(r.job, r.seed, r.cand, variant)] for r in g.itertuples()])
             for s, g in sel.groupby("seed")}
    real_n, *_ = normalize_with(real_kin, real_kin)
    dist = [avg_pairwise_distance(normalize_with(real_kin, f)[0]) for f in feats.values()]
    pooled = normalize_with(real_kin, np.concatenate(list(feats.values())))[0]
    return float(np.mean(dist)), float(np.std(dist)), fid(pooled, real_n)


def summarize(sel, cols):
    seed_means = sel.groupby("seed")[cols].mean()
    return {c: (float(seed_means[c].mean()), float(seed_means[c].std(ddof=0))) for c in cols}


def fmt(m, s=None, digits=3):
    m = 0.0 if round(m, digits) == 0 else m  # no "-0.000"
    return f"{m:.{digits}f}" if s is None else f"{m:.{digits}f} ± {s:.{digits}f}"


MD_COLS = {
    "a": [("bas_beat", "BAS<br>librosa（挑選用）"), ("bas_onset", "BAS<br>onset（原報告）"),
          ("bas_mm", "BAS<br>madmom"), ("dbas_mm", "ΔBAS<br>madmom"), ("style_acc", "舞風<br>正確率"),
          ("pfc", "PFC ↓"), ("fid_k", "FID_k ↓"), ("dist_k", "Dist_k")],
    "b": [("clap_sel", "CLAP-fused<br>（挑選用）"), ("clap", "CLAP<br>（原報告）"), ("gtzan_acc", "曲風<br>正確率"),
          ("bas_beat", "BAS<br>librosa（挑選用）"), ("bas_mm", "BAS<br>madmom"), ("dbas_mm", "ΔBAS<br>madmom"),
          ("style_acc", "舞風<br>正確率"), ("pfc", "PFC ↓")],
}
NAMES_ZH = {
    "UniMuMo, 1 sample": "UniMuMo 原始輸出（1 個樣本）",
    "+ retime (1 sample)": "＋節拍重定時",
    "+ retime to another song (control)": "＋重定時到另一首歌的節拍（對照）",
    "best-of-16 by BAS": "16 選 1（依 BAS）",
    "best-of-16 by BAS + retime": "16 選 1（依 BAS）＋節拍重定時",
    "best-of-16 by CLAP": "16 選 1（依 CLAP）",
    "best-of-16 by CLAP+BAS": "16 選 1（依 CLAP + BAS）",
    "best-of-16 by CLAP+BAS + retime": "16 選 1（依 CLAP + BAS）＋節拍重定時",
}


def write_md(table, exp, path):
    cols = MD_COLS[exp]
    lines = ["| 方法 | " + " | ".join(n for _, n in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for row in table:
        name = NAMES_ZH.get(row["method"], row["method"])
        if name.startswith("real dancers"):
            name = "真人舞者（AIST++）"
        cells = []
        for c, _ in cols:
            v = row.get(c)
            if v is None or v[0] is None or (isinstance(v[0], float) and np.isnan(v[0])):
                cells.append("–")
            else:
                digits = 1 if c in ("fid_k", "dist_k") else (2 if c == "pfc" else 3)
                cells.append(fmt(v[0], v[1], digits) if v[1] is not None else fmt(v[0], None, digits))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    open(path, "w").write("\n".join(lines) + "\n")


def real_row(real, cols):
    rows = pd.DataFrame(real["rows"])
    others = real["music_beats"]
    d_mm, d_bt = [], []
    for r in real["rows"]:
        db = real["motion_beats"][r["name"]]
        oth = [b for s, b in others.items() if s != r["song"]]
        d_mm.append(r["bas_mm"] - chance(db, oth, "mm"))
        d_bt.append(r["bas_beat"] - chance(db, oth, "beat"))
    rows["dbas_mm"], rows["dbas_beat"] = d_mm, d_bt
    held = rows[~rows.train_song]
    rows["style_acc"] = np.nan
    acc = np.mean([np.argmax(p) == g for p, g in zip(held.style_probs, held.genre_idx)])
    out = {c: (float(rows[c].mean()), None) for c in cols if c in rows and c != "style_acc"}
    out["style_acc"] = (float(acc), None)
    return out, len(rows), len(held)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["a", "b"], required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df, kin, jobs = load_rows(args.exp)
    real = pickle.load(open("runs/real_ref.pkl", "rb"))
    real_kin = real["kin"]

    motion_cols = ["bas_beat", "bas_onset", "hit5_onset", "bas_mm", "dbas_beat", "dbas_mm", "style_acc",
                   "style_p", "pfc", "jerk", "speed"]
    audio_cols = ["clap", "clap_sel", "gtzan_acc", "gtzan_p"] if args.exp == "b" else []
    cols = audio_cols + motion_cols
    base = select(df, "bas", 1)
    if args.exp == "a":
        methods = [("UniMuMo, 1 sample", base, "raw"),
                   ("+ retime (1 sample)", base, "retimed"),
                   ("+ retime to another song (control)", base, "retimed_ctrl"),
                   ("best-of-16 by BAS", select(df, "bas", 16), "raw"),
                   ("best-of-16 by BAS + retime", select(df, "bas", 16), "retimed")]
    else:
        methods = [("UniMuMo, 1 sample", base, "raw"),
                   ("best-of-16 by CLAP", select(df, "clap", 16), "raw"),
                   ("best-of-16 by BAS", select(df, "bas", 16), "raw"),
                   ("best-of-16 by CLAP+BAS", select(df, "clap+bas", 16), "raw"),
                   ("best-of-16 by CLAP+BAS + retime", select(df, "clap+bas", 16), "retimed")]

    table, tests = [], []
    base_sel = pick(df, base, "raw")
    for name, chosen, variant in methods:
        sel = pick(df, chosen, variant)
        row = {"method": name} | summarize(sel, cols)
        d_mean, d_sd, f = set_metrics(sel, kin, real_kin, variant)
        row["dist_k"], row["fid_k"] = (d_mean, d_sd), (f, None)
        table.append(row)
        if name != methods[0][0]:
            # paired test over inputs: the seeds of one input are averaged first (n = number of inputs)
            pair = sel.merge(base_sel, on=["job", "seed"], suffixes=("", "_base"))
            per_input = pair.groupby("job")[cols + [c + "_base" for c in cols]].mean()
            for c in cols:
                a, b = per_input[c].values, per_input[c + "_base"].values
                p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
                tests.append({"method": name, "metric": c, "mean_diff": float(np.mean(a - b)),
                              "n_inputs": len(a), "p_wilcoxon": p})
    if args.exp == "a":
        rr, n_real, n_held = real_row(real, cols)
        rr["dist_k"] = (avg_pairwise_distance(normalize_with(real_kin, real_kin)[0]), None)
        rr["fid_k"] = (np.nan, None)  # the real clips are the FID reference
        table.append({"method": f"real dancers (AIST++, {n_real} clips)"} | rr)

    # long csv
    flat = []
    for row in table:
        f = {"method": row["method"]}
        for c, v in row.items():
            if c != "method" and v is not None:
                f[c] = v[0]
                f[c + "_sd"] = v[1]
        flat.append(f)
    pd.DataFrame(flat).to_csv(os.path.join(args.out, f"exp_{args.exp}_table.csv"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(args.out, f"exp_{args.exp}_tests.csv"), index=False)
    write_md(table, args.exp, os.path.join(args.out, f"exp_{args.exp}_table.md"))

    # scaling with the number of candidates
    scal = []
    for selector in (["bas"] if args.exp == "a" else ["clap", "bas", "clap+bas"]):
        for k in KS:
            sel = pick(df, select(df, selector, k), "raw")
            scal.append({"selector": selector, "k": k} | {c: float(sel.groupby("seed")[c].mean().mean()) for c in cols})
    pd.DataFrame(scal).to_csv(os.path.join(args.out, f"exp_{args.exp}_scaling.csv"), index=False)

    # per-genre dance-style accuracy (baseline vs best-of-16)
    if args.exp == "a":
        g = []
        for name, chosen, variant in [methods[0], methods[3]]:
            sel = pick(df, chosen, variant)
            sel["genre"] = [jobs[j]["genre"] for j in sel.job]
            for genre, gg in sel.groupby("genre"):
                g.append({"method": name, "genre": genre, "style_acc": gg.style_acc.mean(), "n": len(gg)})
        pd.DataFrame(g).pivot(index="genre", columns="method", values="style_acc").to_csv(
            os.path.join(args.out, "exp_a_style_by_genre.csv"))
    print(pd.DataFrame(flat).to_string())
    print(pd.DataFrame(tests).to_string())


if __name__ == "__main__":
    main()
