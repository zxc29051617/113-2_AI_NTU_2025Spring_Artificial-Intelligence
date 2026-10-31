"""Split UniMuMo's full.ckpt into a small metadata file and a safetensors file for the
music-motion LM, so the model can be loaded on a 16 GB RAM machine (the LM weights are
streamed to the GPU one tensor at a time instead of being held in RAM).

python split_checkpoint.py --ckpt pretrained/full.ckpt --out pretrained/split
"""
import argparse
import os

import torch
from safetensors.torch import save_file

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    print("keys:", list(ckpt.keys()))
    lm = ckpt.pop("music_motion_lm_weight")
    torch.save(ckpt, os.path.join(args.out, "meta.pt"))
    del ckpt
    seen = set()
    for k in list(lm.keys()):  # tied weights (T5 embeddings) share storage: store copies
        ptr = lm[k].data_ptr()
        lm[k] = lm[k].clone().contiguous() if ptr in seen else lm[k].contiguous()
        seen.add(ptr)
    n = sum(v.numel() for v in lm.values())
    print(f"LM tensors: {len(lm)}, parameters: {n / 1e6:.1f} M")
    save_file(lm, os.path.join(args.out, "music_motion_lm.safetensors"))
    print("saved to", args.out)
