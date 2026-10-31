"""Retiming strength: vary the largest allowed playback-rate change on the single-sample baseline
of experiment A and score the result against its own music and against the other songs.

python retime_sweep.py
"""
import json
import pickle

import numpy as np
import pandas as pd

from beats import motion_beats_speedmin
from metrics import bas, pfc
from motion_style import GENRES, Judge
from retime import retime

RATES = [1.0, 1.05, 1.1, 1.15, 1.25, 1.4]


def main():
    jobs = {j["id"]: j for j in json.load(open("configs/exp_a_music2dance.json"))["jobs"]}
    sc = pickle.load(open("runs/exp_a/scores.pkl", "rb"))
    judge = Judge("runs/motion_style.pt")
    rows = []
    for (jid, seed), out in sorted(sc.items()):
        joints = np.load(f"runs/exp_a/{jid}__s{seed}.npz")["joints"][0].astype(np.float32)
        mb = out["music_beats"][0]
        others = [sc[(j2, seed)]["music_beats"][0]["mm"] for (j2, s2) in sc if s2 == seed and j2 != jid]
        tgt = GENRES.index(jobs[jid]["genre"])
        for r in RATES:
            j = joints if r == 1.0 else retime(joints, mb["beat"], max_rate=r)[0]
            db = motion_beats_speedmin(j)
            own = bas(mb["mm"], db)
            rows.append({"job": jid, "seed": seed, "max_rate": r, "bas_beat": bas(mb["beat"], db), "bas_mm": own,
                         "dbas_mm": own - np.mean([bas(o, db) for o in others]), "pfc": pfc(j),
                         "style_acc": float(np.argmax(judge.probs([j])[0]) == tgt)})
    df = pd.DataFrame(rows)
    res = df.groupby("max_rate")[["bas_beat", "bas_mm", "dbas_mm", "pfc", "style_acc"]].mean()
    res.to_csv("results/exp_a_retime_strength.csv")
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
