#!/usr/bin/env python3
"""Build synthetic long / many-speaker "podcast" test audio from LibriSpeech.

Test-only; not part of the pipeline. Produces a 16 kHz s16le mono WAV plus a
ground-truth RTTM and a JSON manifest of blocks and speaker roles.

Profiles:
  stress      ~2h35m, 17 speakers: 2 hosts throughout, 8 rotating commentators
              (two of them return after >60 min), and a ~35 min phone-band
              "earnings call" (moderator, 2 execs, 4 analysts, strictly sequential)
              with occasional studio-host interjections.
  regression  ~40 min, 2 speakers alternating (host asks short, guest answers long).

Usage:
  make_synthetic.py --libri /data/libri/LibriSpeech/test-clean --profile stress --out /data/synth/stress
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

SR = 16000


def load_speakers(root):
    spk = {}
    for f in sorted(Path(root).glob("*/*/*.flac")):
        spk.setdefault(f.parts[-3], []).append(f)
    info = {}
    for s, files in spk.items():
        durs = [sf.info(str(f)).duration for f in files]
        info[s] = dict(files=files, durs=durs, total=sum(durs))
    return info


class Picker:
    """Cycles each speaker's utterances in a shuffled order, reusing when exhausted."""

    def __init__(self, info, rng):
        self.info, self.rng, self.queues = info, rng, {}

    def next(self, s, max_s=None):
        for _ in range(50):
            q = self.queues.get(s)
            if not q:
                q = list(range(len(self.info[s]["files"])))
                self.rng.shuffle(q)
                self.queues[s] = q
            i = q.pop()
            if max_s is None or self.info[s]["durs"][i] <= max_s:
                break
        x, sr = sf.read(str(self.info[s]["files"][i]), dtype="float32")
        assert sr == SR
        return x


def level(x, rng, target_db=-23.0, jitter_db=3.0):
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    g = 10 ** ((target_db + rng.uniform(-jitter_db, jitter_db)) / 20) / rms
    return x * g


_PHONE = butter(6, [300, 3400], btype="bandpass", fs=SR, output="sos")


def phone(x):
    """Band-limit to 300-3400 Hz, 8 kHz round trip, 8-bit mu-law: a webcast/dial-in channel."""
    y = sosfilt(_PHONE, x)
    y = resample_poly(resample_poly(y, 1, 2), 2, 1)[: len(x)]
    y = np.clip(y / (np.max(np.abs(y)) + 1e-9), -1, 1)
    mu = 255.0
    c = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    c = np.round(c * 127) / 127
    y = np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu
    return y.astype(np.float32)


def build(profile, info, rng):
    by_total = sorted(info, key=lambda s: -info[s]["total"])
    if profile == "regression":
        h, g = by_total[0], by_total[1]
        roles = {h: "host", g: "guest"}
        blocks = [dict(name="interview", kind="interview", length_s=40 * 60, cast=[h, g])]
        return roles, blocks
    pool = by_total[:]
    rng.shuffle(pool[2:])
    hosts = pool[:2]
    rest = pool[2:]
    comm = rest[:8]
    call = rest[8:15]
    mod, e1, e2, *analysts = call
    roles = {hosts[0]: "host_1", hosts[1]: "host_2"}
    roles.update({c: f"commentator_{i + 1}" for i, c in enumerate(comm)})
    roles.update({mod: "call_moderator", e1: "call_exec_1", e2: "call_exec_2"})
    roles.update({a: f"call_analyst_{i + 1}" for i, a in enumerate(analysts)})
    H1, H2 = hosts
    C = comm
    blocks = [
        dict(name="pre_call_1", kind="studio", length_s=35 * 60, cast=[H1, H2, C[0], C[1]],
             weights=[3, 3, 2, 2]),
        dict(name="pre_call_2", kind="studio", length_s=25 * 60, cast=[H1, H2, C[2], C[3]],
             weights=[3, 2, 2, 2]),
        dict(name="earnings_call", kind="call", length_s=35 * 60, mod=mod, execs=[e1, e2],
             analysts=analysts, interjector=H1),
        dict(name="post_call_1", kind="studio", length_s=30 * 60,
             cast=[H1, H2, C[4], C[5], C[0]], weights=[3, 3, 2, 2, 2]),
        dict(name="post_call_2", kind="studio", length_s=30 * 60,
             cast=[H1, H2, C[6], C[7], C[2]], weights=[3, 3, 2, 2, 2]),
    ]
    return roles, blocks


def render(blocks, picker, rng):
    out, truth, t = [], [], 0.0

    def add(spk, x, channel="studio"):
        nonlocal t
        x = level(x, rng)
        if channel == "phone":
            x = level(phone(x), rng, jitter_db=1.0)
        gap = rng.uniform(0.15, 1.0)
        out.append(np.zeros(int(gap * SR), np.float32))
        t += gap
        out.append(x)
        truth.append((t, len(x) / SR, spk, channel))
        t += len(x) / SR

    for b in blocks:
        start = t
        b["start_s"] = round(t, 2)
        if b["kind"] == "interview":
            h, g = b["cast"]
            while t - start < b["length_s"]:
                add(h, picker.next(h, max_s=8))
                for _ in range(rng.randint(1, 4)):
                    add(g, picker.next(g))
        elif b["kind"] == "studio":
            prev = None
            while t - start < b["length_s"]:
                cands = [(s, w) for s, w in zip(b["cast"], b["weights"]) if s != prev]
                s = rng.choices([c for c, _ in cands], [w for _, w in cands])[0]
                for _ in range(rng.randint(1, 3)):
                    add(s, picker.next(s))
                prev = s
        else:
            add(b["mod"], picker.next(b["mod"], max_s=10), "phone")
            add(b["execs"][0], picker.next(b["execs"][0]), "phone")
            add(b["execs"][0], picker.next(b["execs"][0]), "phone")
            qi = 0
            while t - start < b["length_s"]:
                a = b["analysts"][qi % len(b["analysts"])]
                qi += 1
                add(b["mod"], picker.next(b["mod"], max_s=10), "phone")
                for _ in range(rng.randint(1, 2)):
                    add(a, picker.next(a), "phone")
                for _ in range(rng.randint(2, 4)):
                    ex = rng.choice(b["execs"])
                    add(ex, picker.next(ex), "phone")
                if rng.random() < 0.35:
                    add(b["interjector"], picker.next(b["interjector"], max_s=6))
        b["end_s"] = round(t, 2)
    out.append(np.zeros(SR, np.float32))
    audio = np.concatenate(out)
    audio += np.float32(10 ** (-55 / 20)) * np.random.default_rng(1).standard_normal(len(audio)).astype(np.float32)
    return np.clip(audio, -1, 1), truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri", required=True)
    ap.add_argument("--profile", choices=["stress", "regression"], default="stress")
    ap.add_argument("--out", required=True, help="output prefix, e.g. /data/synth/stress")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    info = load_speakers(a.libri)
    roles, blocks = build(a.profile, info, rng)
    audio, truth = render(blocks, Picker(info, rng), rng)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out) + ".wav", audio, SR, subtype="PCM_16")
    with open(str(out) + ".rttm", "w") as fh:
        for st, du, s, ch in truth:
            fh.write(f"SPEAKER synth 1 {st:.3f} {du:.3f} <NA> <NA> {s} <NA> <NA>\n")
    manifest = dict(profile=a.profile, seed=a.seed, duration_s=round(len(audio) / SR, 2),
                    roles=roles, blocks=[{k: v for k, v in b.items()} for b in blocks],
                    turns=len(truth),
                    talk_s={s: round(sum(d for _, d, x, _ in truth if x == s), 1) for s in roles})
    Path(str(out) + ".json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("duration_s", "turns", "talk_s")}, indent=1))


if __name__ == "__main__":
    main()
