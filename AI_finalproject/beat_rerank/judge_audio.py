"""Audio judges for generated music: CLAP text-music similarity (the checkpoint used in the
original report, and a second checkpoint used only for selection) and the GTZAN genre classifier.

python judge_audio.py --run runs/exp_b --jobs configs/exp_b_text2musicdance.json
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np

from judges import CLAP_ORIGINAL, CLAP_SELECT, ClapJudge, GenreJudge

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    jobs = {j["id"]: j for j in json.load(open(args.jobs))["jobs"]}
    files = sorted(glob.glob(os.path.join(args.run, "*__s*.npz")))
    files = [f for f in files if jobs[os.path.basename(f).split("__s")[0]]["task"] in ("mumo", "mu")]
    audio = {}
    for f in files:
        d = np.load(f)
        audio[(os.path.basename(f).split("__s")[0], int(d["seed"]))] = d["audio"].astype(np.float32) / 32767
    keys = sorted(audio)
    texts = sorted({jobs[k[0]]["music_desc"] for k in keys})
    out = {"keys": keys, "texts": texts}

    for tag, name in [("clap_orig", CLAP_ORIGINAL), ("clap_sel", CLAP_SELECT)]:
        judge = ClapJudge(name, device=args.device)
        out[f"{tag}_text"] = judge.text_emb(texts)
        out[f"{tag}_audio"] = {k: judge.audio_emb(list(audio[k]), sr=32000) for k in keys}
        del judge
        print("done", name, flush=True)
    genre = GenreJudge(device=args.device)
    out["gtzan_probs"] = {k: genre.probs(list(audio[k]), sr=32000) for k in keys}
    print("done genre classifier", flush=True)
    with open(os.path.join(args.run, "audio_scores.pkl"), "wb") as f:
        pickle.dump(out, f)
    print("saved", os.path.join(args.run, "audio_scores.pkl"))
