"""Chunked Streaming-Sortformer diarization with cross-chunk speaker linking.

Runs AFTER fw: reads the job's audio.wav (16 kHz s16le mono) and fw.json, runs
`fw sortformer-diarize` on overlapping chunks, links chunk lanes into global
speakers with CAM++ embeddings, and returns a `result.diarization` block in the
shape fuse_diarization.py reads.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .embed import SR, Embedder, highband_ratio

IMPLEMENTATION = "longdiar-chunked-sortformer-v1"


@dataclass
class Params:
    chunk_s: float = 600.0          # nominal chunk length
    overlap_s: float = 60.0         # overlap between consecutive chunks
    snap_s: float = 15.0            # search radius for a quiet cut point
    min_piece_s: float = 1.0        # shortest clean span that gets an embedding
    max_piece_s: float = 4.0        # long clean spans are cut into pieces this long
    max_pieces_per_lane: int = 80
    link_min_overlap_s: float = 2.0  # must-link: shared activity in the chunk overlap
    link_min_iou: float = 0.4
    link_min_cos: float = 0.2       # must-link veto if embeddings clearly disagree
    merge_cos: float = 0.55         # global clustering threshold (wideband)
    merge_cos_narrow: float = 0.55  # global clustering threshold (phone/webcast band)
    split_cos: float = 0.45         # lane holds two voices if its sub-centroids are below this
    split_min_s: float = 8.0        # ...and each side has at least this much clean speech
    dup_cos: float = 0.70           # two lanes of one chunk above this (and not co-talking) are one voice
    narrowband_ratio: float = 0.02  # energy share above 3.8 kHz below this => phone band
    call_label_base: int = 40       # phone-band speakers are numbered from SPEAKER_40
    fw_bin: str = "fw"
    threads: int = 0
    chunk_timeout_s: float = 3600.0
    cache_dir: str | None = None    # per-chunk Sortformer results, reused on re-runs


# ---------------------------------------------------------------- audio helpers

class Wav:
    def __init__(self, path: Path):
        self.path = path
        with wave.open(str(path), "rb") as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, SR):
                raise ValueError(f"{path}: need 16 kHz s16le mono, got "
                                 f"{w.getnchannels()}ch/{8 * w.getsampwidth()}bit/{w.getframerate()}Hz")
            self.n = w.getnframes()
        self.duration = self.n / SR

    def read(self, a: float, b: float) -> np.ndarray:
        i, j = max(0, int(a * SR)), min(self.n, int(b * SR))
        if j <= i:
            return np.zeros(0, np.float32)
        with wave.open(str(self.path), "rb") as w:
            w.setpos(i)
            raw = w.readframes(j - i)
        return np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0

    def write_slice(self, a: float, b: float, out: Path):
        x = self.read(a, b)
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2").tobytes())


def quiet_point(wav: Wav, t: float, radius: float) -> float:
    """Centre of the lowest-energy 200 ms window within +-radius of t."""
    a, b = max(0.0, t - radius), min(wav.duration, t + radius)
    x = wav.read(a, b)
    hop = int(0.05 * SR)
    win = 4
    if len(x) < hop * win:
        return t
    e = np.add.reduceat(x[: len(x) // hop * hop] ** 2, np.arange(0, len(x) // hop * hop, hop))
    e = np.convolve(e, np.ones(win), "valid")
    dist = np.abs(a + (np.arange(len(e)) + win / 2) * hop / SR - t)
    k = int(np.argmin(e * (1.0 + 0.02 * dist)))
    return round(a + (k + win / 2) * hop / SR, 3)


def plan_chunks(wav: Wav, p: Params):
    """[(start, end)] windows with ~overlap_s overlap, cut at quiet points."""
    d = wav.duration
    if d <= p.chunk_s * 1.25:
        return [(0.0, d)]
    out, a = [], 0.0
    while True:
        if d - a <= p.chunk_s * 1.25:
            out.append((a, d))
            return out
        z = quiet_point(wav, a + p.chunk_s, p.snap_s)
        out.append((a, z))
        a = quiet_point(wav, z - p.overlap_s, p.snap_s)


# ---------------------------------------------------------------- sortformer per chunk

def sortformer_chunk(wav: Wav, a: float, b: float, p: Params, tmp: Path):
    cached = None
    if p.cache_dir:
        st = wav.path.stat()
        key = f"{st.st_size}_{int(st.st_mtime)}_{int(a * 1000)}_{int(b * 1000)}.json"
        cached = Path(p.cache_dir) / key
        if cached.is_file():
            r = json.loads(cached.read_text())
            r["turns"] = [tuple(t) for t in r["turns"]]
            return r
    r = _sortformer_chunk(wav, a, b, p, tmp)
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(r))
    return r


def _sortformer_chunk(wav: Wav, a: float, b: float, p: Params, tmp: Path):
    f = tmp / f"chunk_{int(a * 1000):09d}.wav"
    wav.write_slice(a, b, f)
    env = os.environ.copy()
    if p.threads:
        env["RAYON_NUM_THREADS"] = str(p.threads)
    t0 = time.monotonic()
    proc = subprocess.run([p.fw_bin, "sortformer-diarize", "--input", str(f)],
                          capture_output=True, text=True, env=env, timeout=p.chunk_timeout_s)
    f.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fw sortformer-diarize rc={proc.returncode} on [{a:.1f},{b:.1f}]: "
                           f"{proc.stderr.strip()[-400:]}")
    out = json.loads(proc.stdout)
    if out.get("status") != "ok":
        raise RuntimeError(f"sortformer status={out.get('status')} on [{a:.1f},{b:.1f}]")
    res = out["result"]
    turns = [(a + t["start_seconds"], a + t["end_seconds"], int(t["speaker_lane"]))
             for t in res.get("turns", []) if t["end_seconds"] > t["start_seconds"]]
    return dict(turns=turns, active_lanes=res.get("active_lane_count"),
                capacity=res.get("capacity", {}).get("status"),
                overlap_frames=res.get("overlap_frames"),
                seconds=round(time.monotonic() - t0, 2))


# ---------------------------------------------------------------- lane units

@dataclass
class Unit:
    chunk: int
    lane: str
    turns: list                              # [(start, end)]
    pieces: list = field(default_factory=list)   # [(start, end, emb)]
    hb: float = float("nan")
    centroid: np.ndarray | None = None
    label: int = -1
    conf: float = 0.7

    @property
    def talk(self):
        return sum(b - a for a, b in self.turns)

    @property
    def clean(self):
        return sum(b - a for a, b, _ in self.pieces)

    def update(self):
        if self.pieces:
            w = np.array([b - a for a, b, _ in self.pieces])
            c = (np.stack([e for _, _, e in self.pieces]) * w[:, None]).sum(0)
            self.centroid = c / (np.linalg.norm(c) + 1e-9)


def subtract(iv, others):
    """Interval iv minus a list of intervals."""
    out = [iv]
    for oa, ob in others:
        nxt = []
        for a, b in out:
            if ob <= a or oa >= b:
                nxt.append((a, b))
                continue
            if oa > a:
                nxt.append((a, oa))
            if ob < b:
                nxt.append((ob, b))
        out = nxt
    return out


def clean_spans(turns, lane):
    mine = [(a, b) for a, b, l in turns if l == lane]
    others = [(a, b) for a, b, l in turns if l != lane]
    out = []
    for iv in mine:
        out += subtract(iv, [o for o in others if o[1] > iv[0] and o[0] < iv[1]])
    return mine, out


def pieces_of(spans, p: Params):
    out = []
    for a, b in spans:
        a, b = a + 0.1, b - 0.1          # keep clear of lane-change edges
        if b - a < p.min_piece_s:
            continue
        n = max(1, int(np.ceil((b - a) / p.max_piece_s)))
        step = (b - a) / n
        out += [(a + i * step, a + (i + 1) * step) for i in range(n)]
    if len(out) > p.max_pieces_per_lane:
        out = sorted(sorted(out, key=lambda s: s[0] - s[1])[: p.max_pieces_per_lane])
    return out


def two_way_split(E: np.ndarray, w: np.ndarray, iters: int = 20):
    """Weighted spherical 2-means seeded from the farthest pair; returns labels."""
    S = E @ E.T
    i, j = np.unravel_index(np.argmin(S), S.shape)
    C = E[[i, j]].copy()
    lab = np.zeros(len(E), int)
    for _ in range(iters):
        new = np.argmax(E @ C.T, axis=1)
        if (new == lab).all() and _:
            break
        lab = new
        for k in (0, 1):
            if (lab == k).any():
                c = (E[lab == k] * w[lab == k, None]).sum(0)
                C[k] = c / (np.linalg.norm(c) + 1e-9)
    return lab, C


def dedupe_lanes(units, p: Params):
    """Merge lanes of one chunk that carry the same voice (Sortformer sometimes opens a
    second lane for a speaker). Same-voice lanes almost never talk at the same time."""
    units = list(units)
    while True:
        best = None
        for i in range(len(units)):
            for j in range(i + 1, len(units)):
                A, B = units[i], units[j]
                if A.centroid is None or B.centroid is None:
                    continue
                s = float(A.centroid @ B.centroid)
                co = inter(A.turns, B.turns)
                if s >= p.dup_cos and co < 0.1 * min(A.talk, B.talk) and (not best or s > best[0]):
                    best = (s, i, j)
        if not best:
            return units
        _, i, j = best
        keep, drop = (units[i], units[j]) if units[i].talk >= units[j].talk else (units[j], units[i])
        keep.turns = sorted(keep.turns + drop.turns)
        keep.pieces = sorted(keep.pieces + drop.pieces, key=lambda x: x[0])
        keep.lane += "+" + drop.lane
        keep.update()
        units.remove(drop)


def maybe_split(u: Unit, p: Params, depth=0):
    """Split a lane that carries two voices (more than 4 speakers inside one chunk)."""
    if depth >= 2 or len(u.pieces) < 6:
        return [u]
    E = np.stack([e for _, _, e in u.pieces])
    w = np.array([b - a for a, b, _ in u.pieces])
    lab, C = two_way_split(E, w)
    dur = [w[lab == k].sum() for k in (0, 1)]
    if float(C[0] @ C[1]) >= p.split_cos or min(dur) < p.split_min_s:
        return [u]
    halves = []
    for k in (0, 1):
        v = Unit(u.chunk, f"{u.lane}{'ab'[k]}", [], [pc for pc, l in zip(u.pieces, lab) if l == k],
                 u.hb)
        v.update()
        halves.append(v)
    # hand each turn to the half whose pieces it holds; piece-less turns go to the
    # half with the nearest piece in time
    for a, b in u.turns:
        votes = [sum(min(b, pb) - max(a, pa) for pa, pb, _ in h.pieces if pb > a and pa < b)
                 for h in halves]
        if max(votes) > 0:
            k = int(np.argmax(votes))
        else:
            mid = (a + b) / 2
            k = int(np.argmin([min(abs((pa + pb) / 2 - mid) for pa, pb, _ in h.pieces)
                               for h in halves]))
        halves[k].turns.append((a, b))
    out = []
    for h in halves:
        out += maybe_split(h, p, depth + 1)
    return out


# ---------------------------------------------------------------- linking

class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i


def activity(turns, a, b):
    return [(max(x, a), min(y, b)) for x, y in turns if y > a and x < b]


def inter(A, B):
    return sum(max(0.0, min(y1, y2) - max(x1, x2)) for x1, y1 in A for x2, y2 in B)


def must_links(units, chunks, p: Params):
    links = []
    for k in range(len(chunks) - 1):
        a, b = chunks[k + 1][0], chunks[k][1]
        L = [i for i, u in enumerate(units) if u.chunk == k]
        R = [i for i, u in enumerate(units) if u.chunk == k + 1]
        cand = []
        for i in L:
            Ai = activity(units[i].turns, a, b)
            si = sum(y - x for x, y in Ai)
            for j in R:
                Aj = activity(units[j].turns, a, b)
                sj = sum(y - x for x, y in Aj)
                ov = inter(Ai, Aj)
                if ov <= 0:
                    continue
                iou = ov / max(1e-9, si + sj - ov)
                cos = (float(units[i].centroid @ units[j].centroid)
                       if units[i].centroid is not None and units[j].centroid is not None else None)
                cand.append((ov, iou, cos, i, j))
        used_l, used_r = set(), set()
        for ov, iou, cos, i, j in sorted(cand, reverse=True):
            if i in used_l or j in used_r:
                continue
            ok = ov >= p.link_min_overlap_s and iou >= p.link_min_iou
            vetoed = ok and cos is not None and cos < p.link_min_cos
            links.append(dict(chunk=k, a=units[i].lane, b=units[j].lane, overlap_s=round(ov, 2),
                              iou=round(iou, 3), cos=None if cos is None else round(cos, 3),
                              accepted=ok and not vetoed, vetoed=vetoed))
            if ok and not vetoed:
                used_l.add(i)
                used_r.add(j)
                links[-1]["_ij"] = (i, j)
    return links


def cluster(units, links, p: Params):
    n = len(units)
    dsu = DSU(n)
    chunks_of = [{units[i].chunk} for i in range(n)]
    narrow = [bool(u.hb < p.narrowband_ratio) for u in units]

    def union(i, j):
        ri, rj = dsu.find(i), dsu.find(j)
        if ri == rj:
            return True
        if chunks_of[ri] & chunks_of[rj]:
            return False
        dsu.p[rj] = ri
        chunks_of[ri] |= chunks_of[rj]
        return True

    for l in sorted([l for l in links if l.get("_ij")], key=lambda l: -l["overlap_s"]):
        if not union(*l["_ij"]):
            l["accepted"] = False
            l["conflict"] = True

    def centroid(members):
        vs = [units[m].centroid * units[m].clean for m in members if units[m].centroid is not None]
        if not vs:
            return None
        c = np.sum(vs, axis=0)
        return c / (np.linalg.norm(c) + 1e-9)

    groups = {}
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(i)
    cl = [dict(members=m, chunks=set().union(*(chunks_of[dsu.find(x)] for x in m)),
               narrow=np.mean([narrow[x] for x in m]) >= 0.5, c=centroid(m)) for m in groups.values()]
    while True:
        best = None
        for x in range(len(cl)):
            for y in range(x + 1, len(cl)):
                A, B = cl[x], cl[y]
                if A["c"] is None or B["c"] is None or A["chunks"] & B["chunks"]:
                    continue
                if A["narrow"] != B["narrow"]:
                    continue
                s = float(A["c"] @ B["c"])
                thr = p.merge_cos_narrow if A["narrow"] else p.merge_cos
                if s >= thr and (best is None or s > best[0]):
                    best = (s, x, y)
        if not best:
            break
        _, x, y = best
        A, B = cl[x], cl.pop(y)
        A["members"] += B["members"]
        A["chunks"] |= B["chunks"]
        A["c"] = centroid(A["members"])
    return cl


# ---------------------------------------------------------------- main entry

def diarize(audio: Path, p: Params, embedder: Embedder, log=print):
    t_all = time.monotonic()
    wav = Wav(audio)
    chunks = plan_chunks(wav, p)
    log(f"[longdiar] {wav.duration / 3600:.2f} h -> {len(chunks)} chunk(s)")
    timings = dict(sortformer_s=0.0, embed_s=0.0)
    chunk_info, units = [], []
    with tempfile.TemporaryDirectory(prefix="longdiar_") as td:
        for k, (a, b) in enumerate(chunks):
            r = sortformer_chunk(wav, a, b, p, Path(td))
            timings["sortformer_s"] += r["seconds"]
            lanes = sorted({l for _, _, l in r["turns"]})
            t0 = time.monotonic()
            x = wav.read(a, b)
            chunk_units = []
            for lane in lanes:
                mine, clean = clean_spans(r["turns"], lane)
                u = Unit(k, f"c{k}l{lane}", mine)
                for pa, pb in pieces_of(clean, p):
                    seg = x[int((pa - a) * SR): int((pb - a) * SR)]
                    u.pieces.append((pa, pb, embedder.embed(seg)))
                if u.pieces:
                    u.hb = float(np.median([highband_ratio(x[int((pa - a) * SR): int((pb - a) * SR)])
                                            for pa, pb, _ in u.pieces[:8]]))
                u.update()
                chunk_units += maybe_split(u, p)
            n_split = len(chunk_units)
            chunk_units = dedupe_lanes(chunk_units, p)
            timings["embed_s"] += time.monotonic() - t0
            units += chunk_units
            chunk_info.append(dict(i=k, start_s=round(a, 3), end_s=round(b, 3),
                                   active_lanes=r["active_lanes"], capacity=r["capacity"],
                                   units=len(chunk_units),
                                   split_lanes=n_split - len(lanes),
                                   merged_duplicate_lanes=n_split - len(chunk_units),
                                   sortformer_s=r["seconds"]))
            log(f"[longdiar] chunk {k + 1}/{len(chunks)} [{a:.0f}-{b:.0f}s] lanes={len(lanes)} "
                f"units={len(chunk_units)} sortformer={r['seconds']:.1f}s")
    del x

    # tiny units with no clean speech cannot be embedded; attach them by raw-turn embedding
    for u in units:
        if u.centroid is None:
            segs = [wav.read(a, b) for a, b in u.turns if b - a >= 0.5]
            if segs:
                e = embedder.embed(np.concatenate(segs))
                u.centroid, u.conf = e, 0.5
                u.hb = highband_ratio(np.concatenate(segs))

    t0 = time.monotonic()
    links = must_links(units, chunks, p)
    cl = cluster(units, links, p)
    timings["link_s"] = round(time.monotonic() - t0, 3)

    # ownership cuts: middle of each overlap, snapped to a quiet point
    cuts = [0.0]
    for k in range(len(chunks) - 1):
        a, b = chunks[k + 1][0], chunks[k][1]
        cuts.append(quiet_point(wav, (a + b) / 2, min(p.snap_s, (b - a) / 2)))
    cuts.append(wav.duration)

    # labels: studio speakers by first appearance, phone-band from call_label_base
    for ci, c in enumerate(cl):
        own = [t for m in c["members"] for t in units[m].turns
               if cuts[units[m].chunk] <= t[0] < cuts[units[m].chunk + 1]]
        c["first"] = min((a for a, _ in own), default=float("inf"))
    wide = sorted([c for c in cl if not c["narrow"]], key=lambda c: c["first"])
    narrow = sorted([c for c in cl if c["narrow"]], key=lambda c: c["first"])
    base = p.call_label_base if p.call_label_base and len(wide) <= p.call_label_base else len(wide)
    for i, c in enumerate(wide):
        c["ref"] = f"SPEAKER_{i:02d}"
    for i, c in enumerate(narrow):
        c["ref"] = f"SPEAKER_{base + i:02d}"
    for c in cl:
        for m in c["members"]:
            u = units[m]
            u.label = c["ref"]
            if c["c"] is not None and u.centroid is not None and u.conf != 0.5:
                u.conf = float(np.clip(0.5 + 0.5 * float(u.centroid @ c["c"]), 0.5, 0.99))

    turns = []
    for u in units:
        lo, hi = cuts[u.chunk], cuts[u.chunk + 1]
        for a, b in u.turns:
            a, b = max(a, lo), min(b, hi)
            if b - a > 0.02:
                turns.append([a, b, u.label, u.conf])
    turns.sort()
    merged = []
    for t in turns:
        if merged and merged[-1][2] == t[2] and t[0] - merged[-1][1] < 0.05:
            merged[-1][1] = max(merged[-1][1], t[1])
            merged[-1][3] = max(merged[-1][3], t[3])
        else:
            merged.append(t)
    timings["total_s"] = round(time.monotonic() - t_all, 2)
    timings["sortformer_s"] = round(timings["sortformer_s"], 2)
    timings["embed_s"] = round(timings["embed_s"], 2)
    return dict(duration_s=wav.duration, chunks=chunk_info, cuts=cuts, units=units,
                links=links, clusters=cl, turns=merged, timings=timings)
