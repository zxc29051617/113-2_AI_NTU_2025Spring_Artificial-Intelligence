"""Summarise the reproduction of the original project (runs/reproduce) into results/.

python reproduce_report.py
"""
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from judges import GTZAN  # noqa: E402
from metrics import bas  # noqa: E402
from motion_style import GENRES  # noqa: E402

RUN, OUT = "runs/reproduce", "results"
# Fig. 1 of the original report (CLAP score, rows = text, columns = music)
REPORT_CLAP = np.array([
    [0.44, 0.28, 0.16, 0.21, 0.27, 0.23, 0.29, 0.45, 0.23],
    [0.40, 0.15, 0.25, 0.31, 0.26, 0.32, 0.31, 0.25, 0.25],
    [0.32, 0.14, 0.28, 0.29, 0.17, 0.31, 0.28, 0.29, 0.14],
    [0.18, 0.033, 0.21, 0.14, 0.17, 0.15, 0.17, 0.14, 0.12],
    [0.31, 0.097, 0.26, 0.23, 0.18, 0.21, 0.22, 0.19, 0.15],
    [0.25, 0.08, 0.33, 0.24, 0.17, 0.28, 0.22, 0.18, 0.12],
    [0.24, 0.039, 0.45, 0.45, 0.18, 0.44, 0.42, 0.24, 0.25],
    [0.21, 0.0032, 0.34, 0.24, 0.17, 0.24, 0.25, 0.15, 0.25],
    [0.36, 0.28, 0.31, 0.38, 0.33, 0.43, 0.41, 0.30, 0.38]])
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def matrix_stats(A):
    n = len(A)
    off = A[~np.eye(n, dtype=bool)]
    return {"diag": float(np.diag(A).mean()), "off": float(off.mean()),
            "r1_text2music": float(np.mean(np.argmax(A, axis=1) == np.arange(n))),
            "r1_music2text": float(np.mean(np.argmax(A, axis=0) == np.arange(n)))}


def heatmaps(mats, titles, path):
    cmap = LinearSegmentedColormap.from_list("blue", BLUE)
    vmax = max(m.max() for m in mats)
    fig, axes = plt.subplots(1, len(mats), figsize=(5.2 * len(mats), 4.6), facecolor="#fcfcfb")
    for ax, A, t in zip(axes, mats, titles):
        im = ax.imshow(A, cmap=cmap, vmin=0, vmax=vmax)
        for i in range(len(A)):
            for j in range(len(A)):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="#0b0b0b" if A[i, j] < 0.6 * vmax else "#ffffff")
            j = int(np.argmax(A[i]))  # row maximum: the music CLAP finds closest to text i
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#eb6834", lw=2))
        ax.set_title(t, fontsize=10, color="#0b0b0b")
        ax.set_xlabel("music", color="#52514e")
        ax.set_ylabel("text", color="#52514e")
        ax.set_xticks(range(len(A)))
        ax.set_yticks(range(len(A)))
        ax.tick_params(colors="#52514e", labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="CLAP score")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())


def shift_curve(real, path):
    """BAS of real dancers when the music beats are shifted in time: a metric that measures
    synchronisation should peak at zero shift."""
    shifts = np.arange(-45, 46, 3)
    curves = {}
    for kind, label in [("onset", "onsets (original report)"), ("beat", "librosa beats"), ("mm", "madmom beats")]:
        curves[label] = np.mean([[bas(real["music_beats"][r["song"]][kind] + d, real["motion_beats"][r["name"]])
                                  for d in shifts] for r in real["rows"]], axis=0)
    fig, ax = plt.subplots(figsize=(5.6, 3.0), facecolor="#fcfcfb")
    for (label, c), color in zip(curves.items(), ["#2a78d6", "#eb6834", "#1baf7a"]):
        ax.plot(shifts / 60, c, color=color, lw=2, label=label)
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel("music shifted by (s)", color="#52514e", fontsize=8)
    ax.set_ylabel("BAS of real dancers", color="#52514e", fontsize=8)
    ax.tick_params(colors="#52514e", labelsize=7)
    ax.grid(axis="y", color="#e5e4e0", lw=0.8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    ax.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    return {k: (float(v[len(v) // 2]), float(v.min()), float(v.max())) for k, v in curves.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = {j["id"]: j for j in json.load(open("configs/reproduce.json"))["jobs"]}
    sc = pickle.load(open(os.path.join(RUN, "scores.pkl"), "rb"))
    au = pickle.load(open(os.path.join(RUN, "audio_scores.pkl"), "rb"))
    real = pickle.load(open("runs/real_ref.pkl", "rb"))
    lines = []

    # 1. CLAP similarity matrix (nine text -> music + motion prompts)
    t2mm = sorted([k for k in au["keys"] if k[0].startswith("t2mm_")], key=lambda k: int(k[0].split("_")[1]))
    tidx = {t: i for i, t in enumerate(au["texts"])}
    T = au["clap_orig_text"][[tidx[jobs[k[0]]["music_desc"]] for k in t2mm]]
    n_samples = len(au["clap_orig_audio"][t2mm[0]])
    mats = [np.maximum(0, T @ np.stack([au["clap_orig_audio"][k][s] for k in t2mm]).T) for s in range(n_samples)]
    rep, ours = matrix_stats(REPORT_CLAP), [matrix_stats(A) for A in mats]
    heatmaps([REPORT_CLAP, mats[0]], ["original report (Fig. 1)", "reproduction (sample 1 of 4)"],
             os.path.join(OUT, "reproduce_clap_matrix.png"))
    avg = {k: (np.mean([o[k] for o in ours]), np.std([o[k] for o in ours])) for k in ours[0]}
    lines += ["## CLAP 文字-音樂相似度（9 個描述）", "",
              "| | 對角線平均 | 非對角線平均 | 文字→音樂 R@1 | 音樂→文字 R@1 |", "|---|---|---|---|---|",
              f"| 原報告 Fig. 1 | {rep['diag']:.3f} | {rep['off']:.3f} | {rep['r1_text2music']:.2f} | {rep['r1_music2text']:.2f} |",
              f"| 重現（4 個樣本平均 ± 標準差） | {avg['diag'][0]:.3f} ± {avg['diag'][1]:.3f} | "
              f"{avg['off'][0]:.3f} ± {avg['off'][1]:.3f} | {avg['r1_text2music'][0]:.2f} ± {avg['r1_text2music'][1]:.2f} | "
              f"{avg['r1_music2text'][0]:.2f} ± {avg['r1_music2text'][1]:.2f} |",
              "", "R@1 的隨機水準是 1/9 = 0.11。", ""]

    # 2. GTZAN genre classifier
    lines += ["## 音樂曲風分類（GTZAN 分類器）", "", "| 描述 | 4 個樣本的預測 | 目標曲風機率（平均） |", "|---|---|---|"]
    for k in t2mm + [k for k in au["keys"] if k[0].startswith("mt2mu_")]:
        p = au["gtzan_probs"][k]
        desc = jobs[k[0]]["music_desc"]
        target = {"pop_genre": "pop", "disco_genre": "disco", "classical_piano": "classical",
                  "hiphop_beat": "hiphop", "heavy_metal": "metal", "blues_rock": "blues/rock",
                  "latinpop_lock": "pop", "dreamy_indiepop": "pop", "indie_house": "rock/pop"}
        key = "_".join(k[0].split("_")[2:]) if k[0].startswith("t2mm_") else "_".join(k[0].split("_")[1:])
        tg = target.get(key, "")
        tp = np.mean([p[:, GTZAN.index(t)].mean() for t in tg.split("/")]) if tg else np.nan
        lines.append(f"| {desc} ({'text+motion→music' if k[0].startswith('mt2mu') else 'text→music+motion'}) | "
                     f"{', '.join(GTZAN[i] for i in p.argmax(1))} | {tg} {tp:.2f} |")
    lines.append("")

    # 3. Beat Alignment Score of the nine text + music -> motion results
    tm = sorted([k for k in sc if k[0].startswith("tm2m_")])
    rows = [r for k in tm for r in sc[k]["rows"] if r["variant"] == "raw"]
    first = [r for r in rows if r["cand"] == 0]

    def chance_onset(k, r):
        db = sc[k]["motion_beats"][(r["cand"], "raw")]
        others = [sc[k2]["music_beats"][0]["onset"] for k2 in tm if k2 != k]
        return r["bas_onset"] - np.mean([bas(o, db) for o in others])

    d_on = [chance_onset(k, r) for k in tm for r in sc[k]["rows"] if r["variant"] == "raw"]
    real_rows = real["rows"]
    real_d = []
    for r in real_rows:
        db = real["motion_beats"][r["name"]]
        real_d.append(r["bas_onset"] - np.mean([bas(b["onset"], db) for s, b in real["music_beats"].items()
                                                if s != r["song"]]))

    def ms(x):
        return f"{np.mean(x):.3f} ± {np.std(x):.3f}"

    sc_real = shift_curve(real, os.path.join(OUT, "bas_vs_shift_real_dancers.png"))
    lines += ["## Beat Alignment Score（9 個 text + music → motion 結果）", "",
              "| | BAS（onset，Bailando 定義） | ±5 幀命中率 | BAS（librosa beat） | onset BAS 減去配錯音樂的 onset BAS |", "|---|---|---|---|---|",
              "| 原報告 | 0.262 | | | |",
              f"| 重現，每個描述第 1 個樣本 | {np.mean([r['bas_onset'] for r in first]):.3f} | "
              f"{np.mean([r['hit5_onset'] for r in first]):.3f} | {np.mean([r['bas_beat'] for r in first]):.3f} | |",
              f"| 重現，全部 36 個樣本 | {ms([r['bas_onset'] for r in rows])} | {ms([r['hit5_onset'] for r in rows])} | "
              f"{ms([r['bas_beat'] for r in rows])} | {ms(d_on)} |",
              f"| 真人舞者（AIST++，{len(real_rows)} 段） | {ms([r['bas_onset'] for r in real_rows])} | "
              f"{ms([r['hit5_onset'] for r in real_rows])} | {ms([r['bas_beat'] for r in real_rows])} | {ms(real_d)} |", ""]

    lines += ["把音樂相對真人舞者平移 -0.75 到 +0.75 秒時的 BAS（同步的指標應該在 0 秒最高）：", "",
              "| 音樂節拍 | 平移 0 秒 | 平移後最低 | 平移後最高 |", "|---|---|---|---|"]
    for k, (z, lo, hi) in sc_real.items():
        lines.append(f"| {k} | {z:.3f} | {lo:.3f} | {hi:.3f} |")
    lines.append("")

    # 4. dance style of the text + music -> motion results (judged by the AIST++ classifier)
    lines += ["## 舞蹈風格（AIST++ 舞風分類器）", "", "| 動作描述 | 4 個樣本的預測 | 命中數 |", "|---|---|---|"]
    for k in tm + [k for k in sc if k[0] in ("t2mm_0_latinpop_lock", "t2mm_1_indie_house")]:
        rr = [r for r in sc[k]["rows"] if r["variant"] == "raw"]
        tgt = jobs[k[0]].get("genre") or jobs[k[0]]["motion_desc"].split(" is ")[-1].rstrip(".")
        tgt = {"lock": "lock", "house": "house"}.get(tgt, tgt)
        pred = [GENRES[int(np.argmax(r["style_probs"]))] for r in rr]
        lines.append(f"| {jobs[k[0]]['motion_desc']} | {', '.join(pred)} | {sum(p == tgt for p in pred)}/{len(pred)} |")
    lines.append("")
    open(os.path.join(OUT, "reproduce.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
