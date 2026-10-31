"""Motion metrics: beat alignment, physical plausibility, smoothness, kinetic features.

Joints are (T, 22, 3) arrays in metres, y axis up, 60 fps (HumanML3D / UniMuMo skeleton).
"""
import numpy as np
from scipy import linalg

FPS = 60
FOOT_IDX = [7, 10, 8, 11]  # left ankle, left foot, right ankle, right foot (SMPL order)


def bas(music_beats, motion_beats, sigma=3.0):
    """Beat Alignment Score (Bailando, Li et al. 2022): for every music beat take the
    nearest motion beat and average exp(-d^2 / (2 sigma^2)); sigma in frames at 60 fps."""
    mb = np.asarray(music_beats, dtype=float)
    db = np.asarray(motion_beats, dtype=float)
    if len(mb) == 0:
        return np.nan
    if len(db) == 0:
        return 0.0
    d = np.min(np.abs(mb[:, None] - db[None, :]), axis=1)
    return float(np.mean(np.exp(-d ** 2 / (2 * sigma ** 2))))


def hit_rate(music_beats, motion_beats, tol=5):
    """Share of music beats with a motion beat within +-tol frames (the variant described
    in the original report)."""
    mb = np.asarray(music_beats, dtype=float)
    db = np.asarray(motion_beats, dtype=float)
    if len(mb) == 0:
        return np.nan
    if len(db) == 0:
        return 0.0
    d = np.min(np.abs(mb[:, None] - db[None, :]), axis=1)
    return float(np.mean(d <= tol))


def pfc(joints, fps=FPS):
    """Physical Foot Contact score (EDGE, Tseng et al. 2023), lower is better.
    Computed at 30 fps as in EDGE; the up axis here is y."""
    j = joints[:: max(1, fps // 30)]
    dt = 1.0 / 30
    up = 1
    flat = [0, 2]
    root_v = (j[1:, 0] - j[:-1, 0]) / dt
    root_a = (root_v[1:] - root_v[:-1]) / dt
    root_a[:, up] = np.maximum(root_a[:, up], 0)
    root_a = np.linalg.norm(root_a, axis=-1)
    root_a = root_a / max(root_a.max(), 1e-8)
    feet = j[:, FOOT_IDX]
    foot_v = np.linalg.norm(feet[2:][..., flat] - feet[1:-1][..., flat], axis=-1)
    left = np.minimum(foot_v[:, 0], foot_v[:, 1])
    right = np.minimum(foot_v[:, 2], foot_v[:, 3])
    return float(np.mean(left * right * root_a) * 10000)


def mean_jerk(joints, fps=FPS):
    """Mean magnitude of the third time derivative of joint positions (m/s^3)."""
    d3 = np.diff(joints, n=3, axis=0) * fps ** 3
    return float(np.linalg.norm(d3, axis=-1).mean())


def mean_speed(joints, fps=FPS):
    return float(np.linalg.norm(np.diff(joints, axis=0), axis=-1).mean() * fps)


def kinetic_features(joints, fps=FPS, window=2):
    """Kinetic feature vector of AIST++ (Li et al. 2021, FACT): for every joint the average
    horizontal and vertical kinetic energy and the average energy expenditure
    (acceleration magnitude), each averaged over a +-window frame neighbourhood."""
    dt = 1.0 / fps
    vel = np.diff(joints, axis=0) / dt  # (T-1, J, 3)
    acc = np.diff(vel, axis=0) / dt  # (T-2, J, 3)
    k = np.ones(2 * window + 1) / (2 * window + 1)

    def smooth(x):
        return np.apply_along_axis(lambda s: np.convolve(s, k, mode="same"), 0, x)

    v = smooth(vel.reshape(len(vel), -1)).reshape(vel.shape)
    a = smooth(acc.reshape(len(acc), -1)).reshape(acc.shape)
    ke_h = np.mean(v[..., 0] ** 2 + v[..., 2] ** 2, axis=0)
    ke_v = np.mean(v[..., 1] ** 2, axis=0)
    ee = np.mean(np.linalg.norm(a, axis=-1), axis=0)
    return np.stack([ke_h, ke_v, ee], axis=1).reshape(-1).astype(np.float64)


def normalize_with(ref, *xs):
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0) + 1e-10
    return [(x - mu) / sd for x in xs]


def avg_pairwise_distance(feats):
    """Dist_k (Bailando): mean pairwise Euclidean distance of normalised kinetic features."""
    f = np.asarray(feats)
    n = len(f)
    if n < 2:
        return np.nan
    d = np.linalg.norm(f[:, None, :] - f[None, :, :], axis=-1)
    return float(d[np.triu_indices(n, 1)].mean())


def fid(feat_a, feat_b):
    """Frechet distance between two feature sets (FID_k when fed kinetic features)."""
    mu1, mu2 = feat_a.mean(0), feat_b.mean(0)
    s1 = np.cov(feat_a, rowvar=False)
    s2 = np.cov(feat_b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1 + s2 - 2 * covmean))
