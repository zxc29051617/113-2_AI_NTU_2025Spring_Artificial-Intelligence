"""Check that the KV-cached decoder (fast_lm.py) matches UniMuMo's original decoder.

1. logits of the cached step vs the original full forward pass on random prefixes;
2. same seed, 2 s of generation with the original generate() and with the cached one:
   share of identical tokens and the time per step.
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from unimumo_loader import load_unimumo  # noqa: E402
import fast_lm  # noqa: E402

model = load_unimumo(os.path.join(HERE, "pretrained/split"), os.path.join(HERE, "UniMuMo"))
text = "The audio is a latin pop song. <separation> The style of the dance is lock."
for mode in ["music_motion", "music2motion"]:
    print(mode, "worst abs diff:", fast_lm.verify_cached_step(model, text, n_steps=60, mode=mode))

lm = model.music_motion_lm.model
orig_generate = lm.generate
mm = model.music_motion_lm
for name in ["original", "cached"]:
    if name == "cached":
        fast_lm.enable_kv_cache(model)
    torch.manual_seed(0)
    t0 = time.time()
    mu, mo = mm.generate_sample({"text": [text] * 2, "music_code": None, "motion_code": None},
                                duration=2.0, conditional_guidance_scale=4.0, return_result_only=True)
    torch.cuda.synchronize()
    print(f"{name}: {time.time() - t0:.1f}s for 2 s x 2 samples")
    if name == "original":
        ref = (mu.clone(), mo.clone())
    else:
        print("identical music tokens:", (mu == ref[0]).float().mean().item(),
              "identical motion tokens:", (mo == ref[1]).float().mean().item())
lm.generate = orig_generate
