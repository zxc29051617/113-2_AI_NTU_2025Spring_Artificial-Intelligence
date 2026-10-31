"""Beat-aware motion retiming.

Given a motion (T, J, 3) and the music beats (frame indices), find a monotone time warp
w: output frame -> source frame so that motion beats land on music beats, while the local
playback rate stays within [1 / max_rate, max_rate] and the warp never drifts more than
max_shift frames from the original timing. The anchors are chosen by dynamic programming
over (music beat, motion beat) pairs; the warp through the anchors is a monotone cubic
(PCHIP), so the playback speed changes smoothly.

Related idea: Davis and Agrawala, "Visual Rhythm and Beat", SIGGRAPH 2018, which retimes
dance videos to music. UniMuMo uses a DTW variant of this to build its training pairs.
"""
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from beats import motion_beats_speedmin


def _anchor_dp(mb, db, n_frames, max_rate, max_shift, max_skip, lam):
    """Choose anchors (music beat mb[j] <- motion beat db[i]): monotone, rate within
    [1/max_rate, max_rate] between anchors and on the two boundary segments, |shift| <= max_shift.
    Score: +1 per anchor, minus lam * |log rate| per segment."""
    M, K = len(mb), len(db)
    last = n_frames - 1.0
    score = np.full((M, K), -np.inf)
    back = np.full((M, K, 2), -1, dtype=int)
    lo, hi = 1.0 / max_rate, max_rate
    for j in range(M):
        for i in range(K):
            if abs(db[i] - mb[j]) > max_shift:
                continue
            start_ok = db[i] >= mb[j] or db[i] >= mb[j] * lo  # first segment not slower than 1/max_rate
            best, arg = (1.0 if start_ok else -np.inf), (-1, -1)
            for jj in range(max(0, j - max_skip), j):
                for ii in range(max(0, i - max_skip), i):
                    if not np.isfinite(score[jj, ii]):
                        continue
                    rate = (db[i] - db[ii]) / max(mb[j] - mb[jj], 1e-6)
                    if rate < lo or rate > hi:
                        continue
                    s = score[jj, ii] + 1.0 - lam * abs(np.log(rate))
                    if s > best:
                        best, arg = s, (jj, ii)
            score[j, i] = best
            back[j, i] = arg
    end_ok = np.zeros((M, K), dtype=bool)
    for j in range(M):
        for i in range(K):
            end_ok[j, i] = db[i] <= mb[j] or (last - db[i]) >= (last - mb[j]) * lo
    score[~end_ok] = -np.inf
    if not np.isfinite(score).any():
        return []
    j, i = np.unravel_index(np.argmax(score), score.shape)
    path = []
    while j >= 0 and i >= 0:
        path.append((int(mb[j]), int(db[i])))
        j, i = back[j, i]
    return path[::-1]


def warp_function(n_frames, anchors):
    """Monotone warp through the anchors. Before the first and after the last anchor the warp
    is a shift (rate 1) as far as the clip allows, so the first and last frames are never
    frozen."""
    T = n_frames
    if len(anchors) == 0:
        return np.arange(T, dtype=float)
    out = [a[0] for a in anchors]
    src = [a[1] for a in anchors]
    knots_o = [0.0] + out + [T - 1.0]
    knots_s = [max(0.0, src[0] - out[0])] + src + [min(T - 1.0, src[-1] + (T - 1.0 - out[-1]))]
    ko, ks = [], []
    for o, s in zip(knots_o, knots_s):  # drop duplicate output knots (anchor at frame 0 / T-1)
        if ko and o <= ko[-1]:
            continue
        ko.append(o)
        ks.append(s)
    w = PchipInterpolator(ko, ks)(np.arange(T, dtype=float))
    return np.clip(w, 0, T - 1)


def resample(x, w):
    """Cubic-spline interpolation of x (T, ...) at fractional frame positions w (a cubic spline
    keeps the motion detail that linear interpolation would smooth away)."""
    return CubicSpline(np.arange(len(x)), x, axis=0)(w)


def retime(joints, music_beats, max_rate=1.25, max_shift=30, max_skip=4, lam=1.0):
    """Return (retimed joints, info). music_beats are frame indices on the motion grid."""
    T = len(joints)
    mb = np.asarray([b for b in music_beats if 0 < b < T - 1], dtype=float)
    db = motion_beats_speedmin(joints).astype(float)
    anchors = _anchor_dp(mb, db, T, max_rate, max_shift, max_skip, lam) if len(mb) and len(db) else []
    w = warp_function(T, anchors)
    rate = np.gradient(w)
    info = {
        "n_anchors": len(anchors),
        "n_music_beats": int(len(mb)),
        "rate_min": float(rate.min()),
        "rate_max": float(rate.max()),
        "mean_abs_shift": float(np.mean(np.abs(w - np.arange(T)))),
    }
    return resample(joints, w), info
