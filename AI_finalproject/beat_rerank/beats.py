"""Music and motion beat detection.

All beat positions are returned as frame indices on a 60 fps grid, the frame rate of
UniMuMo motion and of AIST++.
"""
import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

FPS = 60
HOP = 512
SR_BEAT = FPS * HOP  # 30720 Hz, so one onset-envelope frame = one motion frame


def load_audio(path, sr=SR_BEAT, duration=None):
    y, _ = librosa.load(path, sr=sr, duration=duration)
    return y


def music_beats_librosa(y, sr=SR_BEAT, mode="beat", n_frames=None):
    """Bailando-style music beats.

    mode="onset": onset peaks (librosa.onset.onset_detect), the "music beat" used by the
                  Bailando BAS code and described in the original report.
    mode="beat":  beat tracker (librosa.beat.beat_track, tightness=100).
    """
    if sr != SR_BEAT:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR_BEAT)
    env = librosa.onset.onset_strength(y=y, sr=SR_BEAT, hop_length=HOP)
    if mode == "onset":
        idx = librosa.onset.onset_detect(onset_envelope=env, sr=SR_BEAT, hop_length=HOP)
    elif mode == "beat":
        tempo = librosa.feature.tempo(onset_envelope=env, sr=SR_BEAT, hop_length=HOP)[0]
        _, idx = librosa.beat.beat_track(onset_envelope=env, sr=SR_BEAT, hop_length=HOP,
                                         start_bpm=float(tempo), tightness=100)
    else:
        raise ValueError(mode)
    idx = np.asarray(idx, dtype=int)
    if n_frames is not None:
        idx = idx[idx < n_frames]
    return idx


_MADMOM = {}


def music_beats_madmom(path_or_signal, n_frames=None, sr=44100):
    """RNN + DBN beat tracker from madmom (Boeck et al.), a tracker independent of librosa."""
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    if "rnn" not in _MADMOM:
        _MADMOM["rnn"] = RNNBeatProcessor()
        _MADMOM["dbn"] = DBNBeatTrackingProcessor(fps=100)
    from madmom.audio.signal import Signal
    if isinstance(path_or_signal, str):
        y, sr = librosa.load(path_or_signal, sr=44100)
    else:
        y = np.asarray(path_or_signal, dtype=np.float32)
        if sr != 44100:  # resample here; madmom would otherwise call an external ffmpeg
            y = librosa.resample(y, orig_sr=sr, target_sr=44100)
    act = _MADMOM["rnn"](Signal(y.astype(np.float32), sample_rate=44100, num_channels=1))
    times = _MADMOM["dbn"](act)
    idx = np.unique(np.rint(np.asarray(times) * FPS).astype(int))
    if n_frames is not None:
        idx = idx[idx < n_frames]
    return idx


def joint_speed(joints, fps=FPS):
    """Mean joint speed per frame (m/s); joints: (T, J, 3)."""
    d = np.linalg.norm(joints[1:] - joints[:-1], axis=-1)  # (T-1, J)
    return d.mean(axis=1) * fps


def motion_beats_speedmin(joints, sigma=5):
    """Bailando motion beats: local minima of the smoothed mean joint speed."""
    vel = gaussian_filter1d(joint_speed(joints), sigma)
    return argrelextrema(vel, np.less)[0].astype(int)
