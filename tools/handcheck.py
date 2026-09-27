#!/usr/bin/env python3
"""Pick short stretches of a longdiar run for a human to spot-check by ear.

  handcheck.py JOB_DIR [--truth truth.rttm] [--youtube VIDEO_ID]

Reads JOB_DIR/longdiar.json and JOB_DIR/fw.longdiar.json. Picks, when present:
  1. a speaker change right at a chunk-ownership cut (cross-chunk consistency)
  2. the start of the detected phone-band / earnings-call segment
  3. a call Q&A handoff: phone-band speaker A -> B -> C within ~60 s
  4. a stretch inside the chunk with the most voices (lane-split region)
  5. the lowest-confidence turn longer than 3 s
Each stretch is ~45 s. With --truth, the true speakers in the stretch are listed too.
"""
import argparse
import json
from pathlib import Path


def mmss(t):
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--truth")
    ap.add_argument("--youtube")
    a = ap.parse_args()
    job = Path(a.job_dir)
    rep = json.loads((job / "longdiar.json").read_text())
    turns = json.loads((job / "fw.longdiar.json").read_text())["result"]["diarization"]["turns"]
    T = [(t["start_ms"] / 1000, t["end_ms"] / 1000, t["speaker_ref"], t["speaker_confidence"])
         for t in turns]
    phone = {s["speaker_ref"] for s in rep.get("speakers", []) if s["channel"] == "phone_band"}
    picks = []

    def refs(lo, hi):
        seen = []
        for s, e, r, _ in T:
            if e > lo and s < hi and r not in seen:
                seen.append(r)
        return seen

    cuts = rep.get("cuts", [])[1:-1]
    best = None
    for c in cuts[len(cuts) // 2:] + cuts[: len(cuts) // 2]:
        near = [(s, r) for s, e, r, _ in T if abs(s - c) < 15]
        if len({r for _, r in near}) >= 2 and not (phone & {r for _, r in near}):
            best = c
            break
    if best is None and cuts:
        best = cuts[len(cuts) // 2]
    if best is not None:
        picks.append(("chunk-boundary speaker change (same people either side of the cut?)",
                      best - 22, best + 23))
    for seg in rep.get("call_segments", [])[:1]:
        picks.append(("start of detected earnings-call / phone-band segment", seg["start_s"] - 15,
                      seg["start_s"] + 30))
        ph = [(s, r) for s, e, r, _ in T if r in phone and seg["start_s"] + 300 < s < seg["end_s"]]
        for i in range(len(ph) - 2):
            (s0, r0), (s1, r1), (s2, r2) = ph[i: i + 3]
            if len({r0, r1, r2}) == 3 and s2 - s0 < 60:
                picks.append(("call Q&A handoff (moderator -> analyst -> exec?)", s0 - 5, s0 + 40))
                break
    ch = max(rep.get("chunks", []), key=lambda c: c.get("units", 0), default=None)
    if ch and ch.get("units", 0) > 4:
        mid = (ch["start_s"] + ch["end_s"]) / 2
        picks.append((f"busiest chunk ({ch['units']} voices in 4 Sortformer lanes)", mid - 22, mid + 23))
    for s, e, r, c in sorted([t for t in T if t[1] - t[0] > 3], key=lambda t: t[3])[:20]:
        picks.append((f"low-confidence turn ({r}, conf {c:.2f})", s - 10, s + 35))

    kept = []
    for why, lo, hi in picks:
        if all(hi <= klo - 60 or lo >= khi + 60 for _, klo, khi in kept):
            kept.append((why, lo, hi))
    kept = kept[:5]
    picks = kept
    truth = []
    if a.truth:
        for line in open(a.truth):
            f = line.split()
            truth.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    for why, lo, hi in picks:
        lo = max(0.0, lo)
        line = f"- {mmss(lo)}-{mmss(hi)}  {why}\n    labels: {' -> '.join(refs(lo, hi))}"
        if truth:
            seen = []
            for s, e, r in truth:
                if e > lo and s < hi and r not in seen:
                    seen.append(r)
            line += f"\n    truth:  {' -> '.join(seen)}"
        if a.youtube:
            line += f"\n    https://youtu.be/{a.youtube}?t={int(lo)}"
        print(line)


if __name__ == "__main__":
    main()
