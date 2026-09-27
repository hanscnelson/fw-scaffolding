"""Speaker embeddings: Kaldi-compatible 80-bin fbank (numpy) + 3D-Speaker CAM++ (onnxruntime).

numpy and onnxruntime both dispatch SIMD kernels at runtime (AVX on Sandy Bridge,
AVX2/AVX-512 where present), so nothing here needs an AVX2 host.
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

SR = 16000
FRAME = 400          # 25 ms
SHIFT = 160          # 10 ms
NFFT = 512
NMEL = 80

MODEL_NAME = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
MODEL_SHA256 = "357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b"
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/" + MODEL_NAME)


@lru_cache(maxsize=1)
def _mel_bank():
    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    lo, hi = mel(20.0), mel(SR / 2)
    centers = np.linspace(lo, hi, NMEL + 2)
    fft_mel = mel(np.arange(NFFT // 2) * SR / NFFT)
    bank = np.zeros((NMEL, NFFT // 2 + 1), np.float32)
    for m in range(NMEL):
        left, c, right = centers[m], centers[m + 1], centers[m + 2]
        up = (fft_mel - left) / (c - left)
        down = (right - fft_mel) / (right - c)
        bank[m, : NFFT // 2] = np.maximum(0.0, np.minimum(up, down))
    return bank


@lru_cache(maxsize=1)
def _povey():
    n = np.arange(FRAME)
    return (0.5 - 0.5 * np.cos(2 * np.pi * n / (FRAME - 1))) ** 0.85


def fbank(x: np.ndarray) -> np.ndarray:
    """Kaldi `compute-fbank-feats` defaults (dither 0, snip_edges) on float samples in [-1, 1]."""
    if len(x) < FRAME:
        return np.zeros((0, NMEL), np.float32)
    n = 1 + (len(x) - FRAME) // SHIFT
    idx = np.arange(FRAME)[None, :] + SHIFT * np.arange(n)[:, None]
    fr = x[idx].astype(np.float64) * 32768.0
    fr -= fr.mean(axis=1, keepdims=True)
    fr[:, 1:] -= 0.97 * fr[:, :-1]
    fr[:, 0] -= 0.97 * fr[:, 0]
    fr *= _povey()
    spec = np.abs(np.fft.rfft(fr, NFFT)) ** 2
    feats = np.log(np.maximum(spec @ _mel_bank().T.astype(np.float64), np.finfo(np.float32).eps))
    return feats.astype(np.float32)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resolve_model(model_dir: str | None) -> Path:
    d = Path(model_dir or os.environ.get("LONGDIAR_MODEL_DIR") or "/models/longdiar")
    p = d / MODEL_NAME
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing; fetch {MODEL_URL} (sha256 {MODEL_SHA256})")
    got = sha256(p)
    if got != MODEL_SHA256:
        raise RuntimeError(f"{p} sha256 {got} != pinned {MODEL_SHA256}")
    return p


class Embedder:
    def __init__(self, model_path: Path, threads: int = 0):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def embed(self, x: np.ndarray) -> np.ndarray:
        """One L2-normalised embedding for a float waveform (>= ~0.5 s)."""
        return self.embed_hb(x)[0]

    def embed_hb(self, x: np.ndarray):
        """(embedding, high-band energy share) from one fbank pass."""
        f = fbank(x)
        hb = _highband(f)
        f -= f.mean(axis=0, keepdims=True)
        e = self.sess.run(None, {self.inp: f[None]})[0][0]
        return e / (np.linalg.norm(e) + 1e-9), hb


def _highband(f: np.ndarray, cutoff_hz: float = 3800.0) -> float:
    if not len(f):
        return float("nan")
    p = np.exp(f)
    mel_hz = 700.0 * (np.exp(np.linspace(1127.0 * np.log(1 + 20 / 700.0),
                                         1127.0 * np.log(1 + SR / 2 / 700.0),
                                         NMEL + 2)[1:-1] / 1127.0) - 1)
    hi = mel_hz >= cutoff_hz
    return float(p[:, hi].sum() / (p.sum() + 1e-9))


def highband_ratio(x: np.ndarray, cutoff_hz: float = 3800.0) -> float:
    """Share of spectral energy above `cutoff_hz`; ~0 for phone/webcast-band audio."""
    return _highband(fbank(x), cutoff_hz)
