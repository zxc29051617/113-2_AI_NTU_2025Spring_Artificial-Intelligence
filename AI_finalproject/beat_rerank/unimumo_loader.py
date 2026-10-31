"""Load UniMuMo from the split checkpoint (see split_checkpoint.py) with low CPU memory."""
import os
import sys

import torch


def add_unimumo_to_path(unimumo_dir):
    unimumo_dir = os.path.abspath(unimumo_dir)
    if unimumo_dir not in sys.path:
        sys.path.insert(0, unimumo_dir)


def load_unimumo(split_dir, unimumo_dir, device="cuda"):
    add_unimumo_to_path(unimumo_dir)
    from safetensors import safe_open
    from unimumo.models import UniMuMo

    meta = torch.load(os.path.join(split_dir, "meta.pt"), map_location="cpu")
    model = UniMuMo(**meta, music_motion_lm_weight=None)
    model = model.to(device)
    model.music_motion_lm = model.music_motion_lm.to(device)

    sd = model.music_motion_lm.state_dict()  # tensors share storage with the parameters
    path = os.path.join(split_dir, "music_motion_lm.safetensors")
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        missing = sorted(set(sd) - set(keys))
        unexpected = sorted(set(keys) - set(sd))
        if missing or unexpected:
            raise RuntimeError(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        with torch.no_grad():
            for k in keys:
                sd[k].copy_(f.get_tensor(k))
    model.eval()
    return model
