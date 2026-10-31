"""Score every generated candidate, plus beat-retimed copies of it, with the motion metrics.

Pass 1 finds the music beats of every clip:
    beat   librosa beat tracker (used for selection and for retiming)
    onset  librosa onsets (the "music beats" of the original report / Bailando)
    mm     madmom RNN + DBN beat tracker (held out: no method uses it)
Pass 2 scores three variants of every candidate motion:
    raw            the UniMuMo output
    retimed        retime.py aligned to the beats of its own music
    retimed_ctrl   control: retimed to the beats of a different clip's music
with
    bas_beat / bas_onset / bas_mm   Beat Alignment Score against the three beat sets
    hit5_onset                      share of onsets with a motion beat within +-5 frames
    pfc, jerk, speed                physical foot contact score, mean jerk, mean joint speed
    kinetic features, dance-genre probabilities (motion_style.py)
All scores are against the clip's own music.

python score_candidates.py --run runs/exp_a --jobs configs/exp_a_music2dance.json
"""
import argparse
import glob
import json
import os
import pickle
from multiprocessing import Pool

import librosa
import numpy as np

from beats import SR_BEAT, motion_beats_speedmin, music_beats_librosa, music_beats_madmom
from metrics import bas, hit_rate, kinetic_features, mean_jerk, mean_speed, pfc
from retime import retime

STYLE = {}


def music_beats(y32, n_frames):
    y = librosa.resample(y32, orig_sr=32000, target_sr=SR_BEAT)
    return {"beat": music_beats_librosa(y, mode="beat", n_frames=n_frames),
            "onset": music_beats_librosa(y, mode="onset", n_frames=n_frames),
            "mm": music_beats_madmom(y32, n_frames=n_frames, sr=32000)}


def pass1(task):
    """Music beats of every candidate's music (computed once when the music is the input)."""
    path, job = task
    d = np.load(path)
    audio = d["audio"].astype(np.float32) / 32767
    T = d["joints"].shape[1]
    if job["task"] == "mo":
        mb = music_beats(audio[0], T)
        beats = {c: mb for c in range(len(audio))}
    else:
        beats = {c: music_beats(audio[c], T) for c in range(len(audio))}
    return job["id"], int(d["seed"]), beats


def motion_metrics(j, mb):
    db = motion_beats_speedmin(j)
    return {"bas_beat": bas(mb["beat"], db), "bas_onset": bas(mb["onset"], db),
            "hit5_onset": hit_rate(mb["onset"], db), "bas_mm": bas(mb["mm"], db),
            "pfc": pfc(j), "jerk": mean_jerk(j), "speed": mean_speed(j), "n_motion_beats": len(db)}, db


def pass2(task):
    path, job, own, ctrl, style_ckpt = task
    from motion_style import Judge
    if "judge" not in STYLE:
        STYLE["judge"] = Judge(style_ckpt)
    d = np.load(path)
    joints = d["joints"].astype(np.float32)
    seed = int(d["seed"])
    out = {"rows": [], "motion_beats": {}, "kin": {}}
    for c in range(len(joints)):
        variants = {"raw": (joints[c], {})}
        if job["task"] != "mu":  # for motion -> music the motion is the input: nothing to retime
            variants["retimed"] = retime(joints[c], own[c]["beat"])
            if ctrl is not None:
                variants["retimed_ctrl"] = retime(joints[c], ctrl[c]["beat"])
        for v, (j, info) in variants.items():
            m, db = motion_metrics(j, own[c])
            m.update({"job": job["id"], "seed": seed, "cand": c, "variant": v})
            m.update({f"retime_{k}": val for k, val in info.items()})
            m["style_probs"] = STYLE["judge"].probs([j])[0]
            out["rows"].append(m)
            out["motion_beats"][(c, v)] = db
            out["kin"][(c, v)] = kinetic_features(j)
    return job["id"], seed, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--style_ckpt", default="runs/motion_style.pt")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    job_list = json.load(open(args.jobs))["jobs"]
    jobs = {j["id"]: j for j in job_list}
    order = [j["id"] for j in job_list]
    files = {}
    for p in sorted(glob.glob(os.path.join(args.run, "*__s*.npz"))):
        jid, seed = os.path.basename(p)[:-4].split("__s")
        files[(jid, int(seed))] = p

    with Pool(args.workers) as pool:
        beats = {}
        for jid, seed, b in pool.imap_unordered(pass1, [(p, jobs[k[0]]) for k, p in files.items()]):
            beats[(jid, seed)] = b
        print("music beats done", flush=True)
        tasks = []
        for (jid, seed), p in files.items():
            # control: the music of the next clip in the job list (same seed and candidate index)
            same_task = [o for o in order if jobs[o]["task"] == jobs[jid]["task"]]
            other = same_task[(same_task.index(jid) + 1) % len(same_task)]
            ctrl = beats.get((other, seed)) if len(same_task) > 1 else None
            tasks.append((p, jobs[jid], beats[(jid, seed)], ctrl, args.style_ckpt))
        results = {}
        for i, (jid, seed, out) in enumerate(pool.imap_unordered(pass2, tasks)):
            out["music_beats"] = beats[(jid, seed)]
            results[(jid, seed)] = out
            print(f"[{i + 1}/{len(tasks)}] {jid} seed {seed}", flush=True)
    with open(os.path.join(args.run, "scores.pkl"), "wb") as f:
        pickle.dump(results, f)
    print("saved", os.path.join(args.run, "scores.pkl"))
