#!/usr/bin/env python3
"""Score diarization against a ground-truth RTTM (test harness).

  score.py truth.rttm HYP [--cuts longdiar.json] [--fused fused.json] [--manifest synth.json]

HYP is an fw-style JSON (result.diarization.turns) or an RTTM. Reports:
  der_1to1       DER with the optimal one-to-one speaker mapping (no collar)
  der_merge_ok   DER when every hyp label maps to its best true speaker (many-to-one);
                 this is what a flat speaker_map.json can repair by naming
  splits/merges  true speakers spread over >1 label / labels covering >1 true speaker
                 (a share counts when it is >= 10% of that speaker's/label's time)
  chunk consistency  per true speaker: same dominant label in every chunk it speaks in
  word_speaker_error on fuse_diarization.py --json output, many-to-one mapping
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

RES = 0.01


def read_rttm(path):
    out = []
    for line in open(path):
        f = line.split()
        if f and f[0] == "SPEAKER":
            a, d = float(f[3]), float(f[4])
            out.append((a, a + d, f[7]))
    return out


def read_hyp(path):
    if path.endswith(".rttm"):
        return read_rttm(path)
    d = json.load(open(path))
    turns = d["result"]["diarization"]["turns"]
    return [(t["start_ms"] / 1000, t["end_ms"] / 1000, t["speaker_ref"]) for t in turns
            if t.get("speaker_ref")]


def frames(turns, n):
    labs = sorted({s for *_, s in turns})
    idx = {s: i for i, s in enumerate(labs)}
    M = np.zeros((len(labs), n), bool)
    for a, b, s in turns:
        M[idx[s], int(a / RES): int(b / RES)] = True
    return labs, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("hyp")
    ap.add_argument("--cuts", help="longdiar.json whose cuts define chunks")
    ap.add_argument("--fused", help="fuse_diarization.py --json output built from HYP")
    ap.add_argument("--manifest", help="synthetic manifest (roles, blocks)")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    T, H = read_rttm(a.truth), read_hyp(a.hyp)
    n = int(max(b for _, b, _ in T + H) / RES) + 1
    tl, TM = frames(T, n)
    hl, HM = frames(H, n)
    ref_speech = TM.sum()
    C = (TM[:, None, :] & HM[None, :, :]).sum(-1).astype(float)   # [true, hyp] overlap frames
    r, c = linear_sum_assignment(-C)
    ref_n = TM.sum(0)
    hyp_n = HM.sum(0)
    miss = np.maximum(ref_n - hyp_n, 0).sum()
    fa = np.maximum(hyp_n - ref_n, 0).sum()

    def confusion(mapping):
        correct = np.zeros(n, int)
        for ti, hi in mapping:
            correct += (TM[ti] & HM[hi])
        return (np.minimum(ref_n, hyp_n) - correct).clip(0).sum()

    conf_1 = confusion(zip(r, c))
    best_true = C.argmax(0)
    conf_m = confusion([(best_true[h], h) for h in range(len(hl))])
    tt = C.sum(1)
    hh = C.sum(0)
    splits = {tl[i]: [hl[j] for j in range(len(hl)) if C[i, j] >= 0.1 * tt[i] and tt[i] > 0]
              for i in range(len(tl))}
    merges = {hl[j]: [tl[i] for i in range(len(tl)) if C[i, j] >= 0.1 * hh[j] and hh[j] > 0]
              for j in range(len(hl))}
    roles = json.load(open(a.manifest))["roles"] if a.manifest else {}
    out = dict(
        label=a.label, true_speakers=len(tl), hyp_speakers=len(hl),
        der_1to1=round(float(miss + fa + conf_1) / ref_speech, 4),
        der_merge_ok=round(float(miss + fa + conf_m) / ref_speech, 4),
        miss=round(float(miss) / ref_speech, 4), false_alarm=round(float(fa) / ref_speech, 4),
        confusion_1to1=round(float(conf_1) / ref_speech, 4),
        confusion_merge_ok=round(float(conf_m) / ref_speech, 4),
        split_true_speakers={f"{k} ({roles.get(k, '')})": v for k, v in splits.items() if len(v) > 1},
        merged_labels={k: [f"{x} ({roles.get(x, '')})" for x in v]
                       for k, v in merges.items() if len(v) > 1},
    )
    if a.cuts:
        cuts = json.load(open(a.cuts))["cuts"]
        per = defaultdict(dict)
        for k in range(len(cuts) - 1):
            lo, hi = int(cuts[k] / RES), int(cuts[k + 1] / RES)
            Ck = (TM[:, None, lo:hi] & HM[None, :, lo:hi]).sum(-1)
            for i in range(len(tl)):
                if TM[i, lo:hi].sum() * RES >= 5:
                    per[tl[i]][k] = hl[int(Ck[i].argmax())] if Ck[i].max() else None
        multi = {s: v for s, v in per.items() if len(v) >= 2}
        consistent = {s: len(set(v.values())) == 1 for s, v in multi.items()}
        bnd = total = 0
        for s, v in multi.items():
            ks = sorted(v)
            for x, y in zip(ks, ks[1:]):
                if y == x + 1:
                    total += 1
                    bnd += v[x] == v[y]
        out["chunk_consistency"] = dict(
            speakers_in_2plus_chunks=len(multi),
            same_label_in_every_chunk=sum(consistent.values()),
            adjacent_chunk_pairs_same_label=f"{bnd}/{total}",
            inconsistent={f"{s} ({roles.get(s, '')})": multi[s] for s, ok in consistent.items()
                          if not ok})
    if a.fused:
        F = json.load(open(a.fused))
        words = err = 0
        lab_true = defaultdict(lambda: defaultdict(float))
        rows = []
        for t in F:
            lo, hi = int(t["start"] / RES), int(t["end"] / RES) + 1
            cover = TM[:, lo:hi].sum(1)
            if cover.max() == 0:
                continue
            nw = len(t["text"].split())
            truth = tl[int(cover.argmax())]
            rows.append((t["speaker"], truth, nw))
            lab_true[t["speaker"]][truth] += nw
        m = {h: max(v, key=v.get) for h, v in lab_true.items()}
        for h, truth, nw in rows:
            words += nw
            err += nw * (m[h] != truth)
        out["word_speaker_error_merge_ok"] = round(err / max(1, words), 4)
        out["fused_words_scored"] = words
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
