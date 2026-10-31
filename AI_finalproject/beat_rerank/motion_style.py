"""Dance-genre classifier for motion, trained on AIST++ (10 genres). It is a held-out judge of
whether a generated dance follows the requested style (no method uses it for selection).

Input per frame (20 fps): joint positions relative to the root on the ground plane (66) and
joint velocities (66); 6 s windows; a small 1D CNN; at test time the window probabilities are
averaged. Split by song: songs 0-3 of every genre for training, songs 4-5 for testing, so the
reported accuracy is on music the classifier has not seen.

python motion_style.py --data data/aist_motion --unimumo_dir UniMuMo --out runs/motion_style.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

GENRES = ["break", "pop", "lock", "middle hip-hop", "LA style hip-hop", "house", "waack", "krump",
          "street jazz", "ballet jazz"]
CODE = {"gBR": 0, "gPO": 1, "gLO": 2, "gMH": 3, "gLH": 4, "gHO": 5, "gWA": 6, "gKR": 7, "gJS": 8, "gJB": 9}
WIN = 120  # 6 s at 20 fps


class Net(nn.Module):
    def __init__(self, d_in=132, h=128, n_cls=10):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(d_in, h, 5, padding=2), nn.GELU(), nn.Dropout(0.2),
            nn.Conv1d(h, h, 5, padding=2, stride=2), nn.GELU(), nn.Dropout(0.2),
            nn.Conv1d(h, h, 5, padding=2, stride=2), nn.GELU())
        self.head = nn.Linear(h, n_cls)

    def forward(self, x):  # x: (B, T, D)
        return self.head(self.f(x.transpose(1, 2)).mean(-1))


def joint_frames(joints, fps=60):
    """(T, 22, 3) joints at 60 fps -> (T', 132) per-frame features at 20 fps."""
    j = np.asarray(joints, dtype=np.float32)[:: fps // 20]
    rel = j.copy()
    rel[:, :, [0, 2]] -= j[:, :1, [0, 2]]
    vel = np.diff(j, axis=0, prepend=j[:1]) * 20
    return np.concatenate([rel.reshape(len(j), -1), vel.reshape(len(j), -1)], axis=1)


def windows(x, step=30):
    if len(x) <= WIN:
        return np.pad(x, ((0, WIN - len(x)), (0, 0)))[None]
    return np.stack([x[s:s + WIN] for s in range(0, len(x) - WIN + 1, step)])


def aist_joints(data_dir, unimumo_dir):
    """Real AIST++ dances (UniMuMo's processed release) as joint positions at 60 fps."""
    sys.path.insert(0, os.path.abspath(unimumo_dir))
    from unimumo.motion.motion_process import recover_from_ric
    out = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "*", "joint_vecs", "g*.npy"))):
        feat = torch.from_numpy(np.load(f)).float()
        out[os.path.basename(f)[:-4]] = recover_from_ric(feat, 22).numpy()
    return out


class Judge:
    def __init__(self, ckpt, device="cpu"):
        pkg = torch.load(ckpt, map_location="cpu")
        self.mean, self.std = pkg["mean"], pkg["std"]
        self.net = Net()
        self.net.load_state_dict(pkg["state_dict"])
        self.net.eval().to(device)
        self.device = device

    @torch.no_grad()
    def probs(self, joints_list):
        out = []
        for j in joints_list:
            x = (joint_frames(j) - self.mean) / self.std
            w = torch.from_numpy(windows(x).astype(np.float32)).to(self.device)
            out.append(torch.softmax(self.net(w), -1).mean(0).cpu().numpy())
        return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aist_motion")
    ap.add_argument("--unimumo_dir", default="UniMuMo")
    ap.add_argument("--out", default="runs/motion_style.pt")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    joints = aist_joints(args.data, args.unimumo_dir)
    items = [(joint_frames(j), CODE[n[:3]], int(n.split("_")[4][-1])) for n, j in joints.items()]
    tr = [it for it in items if it[2] <= 3]
    te = [it for it in items if it[2] >= 4]
    allx = np.concatenate([x for x, *_ in tr])
    mean, std = allx.mean(0), allx.std(0) + 1e-6
    tr = [((x - mean) / std, y, s) for x, y, s in tr]
    te = [((x - mean) / std, y, s) for x, y, s in te]
    print(f"train sequences {len(tr)}, test sequences {len(te)}")

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(args.epochs):
        net.train()
        idx = np.random.permutation(len(tr))
        tot = 0.0
        for b in range(0, len(idx), 32):
            xs, ys = [], []
            for i in idx[b:b + 32]:
                x, y, _ = tr[i]
                s = np.random.randint(0, max(1, len(x) - WIN + 1))
                xw = x[s:s + WIN]
                xs.append(np.pad(xw, ((0, WIN - len(xw)), (0, 0))))
                ys.append(y)
            xb = torch.from_numpy(np.stack(xs).astype(np.float32))
            loss = nn.functional.cross_entropy(net(xb + 0.05 * torch.randn_like(xb)), torch.tensor(ys))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xs)
        if (ep + 1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                pred = [net(torch.from_numpy(windows(x).astype(np.float32))).softmax(-1).mean(0).argmax().item()
                        for x, *_ in te]
            acc = np.mean(np.array(pred) == np.array([y for _, y, _ in te]))
            print(f"epoch {ep + 1} loss {tot / len(tr):.3f} held-out-song accuracy {acc:.3f}")
    torch.save({"state_dict": net.state_dict(), "mean": mean, "std": std, "test_acc": float(acc)}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
