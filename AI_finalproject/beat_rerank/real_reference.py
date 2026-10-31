"""Reference numbers from real dancers: AIST++ advanced dances (sFM, 28-48 s) cut to the same
8-18 s window as the generated clips, scored with exactly the same metrics.

python real_reference.py --data data/aist_motion --music data/aist_music --out runs/real_ref.pkl
"""
import argparse
import os
import pickle

import librosa
import numpy as np

from metrics import kinetic_features
from motion_style import CODE, Judge, aist_joints
from score_candidates import motion_metrics, music_beats

OFFSET, DUR, FPS = 8.0, 10.0, 60

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aist_motion")
    ap.add_argument("--music", default="data/aist_music")
    ap.add_argument("--unimumo_dir", default="UniMuMo")
    ap.add_argument("--style_ckpt", default="runs/motion_style.pt")
    ap.add_argument("--out", default="runs/real_ref.pkl")
    args = ap.parse_args()
    judge = Judge(args.style_ckpt)
    joints = aist_joints(args.data, args.unimumo_dir)
    a, b = int(OFFSET * FPS), int((OFFSET + DUR) * FPS)
    beats, rows, kin, mbeats = {}, [], [], {}
    for name, j in sorted(joints.items()):
        if "_sFM_" not in name or len(j) < b:
            continue
        song = name.split("_")[4]
        if song not in beats:
            y, _ = librosa.load(os.path.join(args.music, song + ".mp3"), sr=32000, offset=OFFSET, duration=DUR)
            beats[song] = music_beats(y, b - a)
        clip = j[a:b]
        m, db = motion_metrics(clip, beats[song])
        m.update({"name": name, "song": song, "genre_idx": CODE[name[:3]],
                  "style_probs": judge.probs([clip])[0], "train_song": int(song[-1]) <= 3})
        rows.append(m)
        kin.append(kinetic_features(clip))
        mbeats[name] = db
    pickle.dump({"rows": rows, "kin": np.stack(kin), "music_beats": beats, "motion_beats": mbeats},
                open(args.out, "wb"))
    print(len(rows), "real clips from", len(beats), "songs; saved", args.out)
