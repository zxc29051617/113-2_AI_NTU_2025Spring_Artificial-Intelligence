"""Run UniMuMo on a list of jobs, several seeds and N candidates per seed.

Each (job, seed) is saved as one .npz with all N candidates:
    joints   (N, T, 22, 3) float16   generated (or input) motion as joint positions, 60 fps
    feats    (N, T, 263)   float16   motion features (HumanML3D format)
    audio    (N, L)        int16     generated (or input) music, 32 kHz
Text is built exactly as in UniMuMo's generate.py: music_desc + ' <separation> ' + motion_desc.

python generate_batch.py --jobs configs/exp_a_music2dance.json --out runs/exp_a --seeds 0 1 2 --n_cand 16
"""
import argparse
import json
import os
import time

import librosa
import numpy as np
import torch

import fast_lm
from unimumo_loader import load_unimumo

SR = 32000


def build_text(job):
    mu = (job.get("music_desc") or "").capitalize()
    mo = (job.get("motion_desc") or "").capitalize()
    return f"{mu} <separation> {mo}"


def to_int16(x):
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def wait_for_shared_gpu(limit_mib=1024, poll=30):
    """On a shared GPU, pause while another process uses more than limit_mib of memory."""
    import subprocess
    while True:
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        except FileNotFoundError:
            return
        others = [l for l in out.strip().splitlines()
                  if l and int(l.split(",")[0]) != os.getpid() and int(l.split(",")[1]) > limit_mib]
        if not others:
            return
        print("GPU busy, waiting:", others, flush=True)
        time.sleep(poll)


def run_job(model, job, n_cand, cfg, temperature, duration):
    text = [build_text(job)]
    task = job["task"]
    if task == "mumo":
        wav, motion = model.generate_music_motion(
            text_description=text, batch_size=n_cand, duration=duration,
            conditional_guidance_scale=cfg, temperature=temperature)
        return wav, motion["joint"], motion["feature"]
    if task == "mo":
        y, _ = librosa.load(job["music_path"], sr=SR, offset=job.get("offset", 0.0), duration=duration)
        motion = model.generate_motion_from_music(
            waveform=y[None, None, :], text_description=text, batch_size=n_cand,
            conditional_guidance_scale=cfg, temperature=temperature)
        wav = np.tile(y[None, :], (n_cand, 1))
        return wav, motion["joint"], motion["feature"]
    if task == "mu":
        feat = np.load(job["motion_path"])
        start = int(job.get("offset", 0.0) * model.motion_fps)
        feat = feat[start:start + int(model.motion_fps * duration)][None]
        wav = model.generate_music_from_motion(
            motion_feature=feat, text_description=text, batch_size=n_cand,
            conditional_guidance_scale=cfg, temperature=temperature)
        joint = model.motion_vec_to_joint(torch.Tensor(model.normalize_motion(feat)))
        return wav, np.tile(joint, (n_cand, 1, 1, 1)), np.tile(feat, (n_cand, 1, 1))
    raise ValueError(task)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split_ckpt", default="pretrained/split")
    ap.add_argument("--unimumo_dir", default="UniMuMo")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_cand", type=int, default=1)
    ap.add_argument("--cfg", type=float, default=4.0)  # generate.py default
    ap.add_argument("--temperature", type=float, default=None)  # None -> 1.0 from the config
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--only", nargs="*", default=None, help="run only these job ids")
    ap.add_argument("--no_kv_cache", action="store_true", help="use UniMuMo's original decoder")
    ap.add_argument("--batch", type=int, default=8, help="candidates per forward pass")
    ap.add_argument("--max_vram_gb", type=float, default=9.5, help="hard cap for PyTorch's allocator")
    args = ap.parse_args()

    jobs = json.load(open(args.jobs))["jobs"]
    os.makedirs(args.out, exist_ok=True)
    json.dump({"args": vars(args), "jobs": jobs}, open(os.path.join(args.out, "run_config.json"), "w"), indent=1)

    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_vram_gb / total))
    model = load_unimumo(args.split_ckpt, args.unimumo_dir)
    if not args.no_kv_cache:  # same tokens as the original decoder, much faster (scripts/check_kv_cache.py)
        fast_lm.enable_kv_cache(model)
    log = open(os.path.join(args.out, "timing.log"), "a")
    for seed in args.seeds:
        for ji, job in enumerate(jobs):  # the seed of a job depends on its index in the full list
            path = os.path.join(args.out, f"{job['id']}__s{seed}.npz")
            if (args.only and job["id"] not in args.only) or os.path.exists(path):
                continue
            wait_for_shared_gpu()
            torch.manual_seed(seed * 100003 + ji)
            np.random.seed(seed * 100003 + ji)
            t0 = time.time()
            parts = []
            for b0 in range(0, args.n_cand, args.batch):
                parts.append(run_job(model, job, min(args.batch, args.n_cand - b0), args.cfg,
                                     args.temperature, args.duration))
                torch.cuda.empty_cache()
            wav, joint, feat = [np.concatenate([np.asarray(p[i]) for p in parts]) for i in range(3)]
            dt = time.time() - t0
            np.savez_compressed(path, joints=joint.astype(np.float16), feats=feat.astype(np.float16),
                                audio=to_int16(np.asarray(wav).reshape(args.n_cand, -1)),
                                text=build_text(job), seed=seed)
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            msg = f"{job['id']} seed={seed} n={args.n_cand} {dt:.1f}s peak_vram={peak:.2f}GB"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
