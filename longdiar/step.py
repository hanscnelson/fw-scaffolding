"""Post-fw diarization step: decide, run, write outputs, never fail the video.

Outputs (in the job dir, fw.json is never modified unless --in-place):
  fw.longdiar.json   fw.json with result.diarization replaced (only when status == "ok")
  longdiar.json      step report: status ok|skipped|warning, reason, quality flags,
                     chunks, links, speakers, call segments, timings, peak RSS
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import __version__
from .chunked import IMPLEMENTATION, Params, diarize
from .embed import MODEL_NAME, MODEL_SHA256, Embedder, resolve_model

SORTFORMER_CEILING_S = 2 * 60 * 60


def peak_rss_mb():
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kids_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return dict(step_mb=round(self_kb / 1024, 1), largest_child_mb=round(kids_kb / 1024, 1))


def fw_health(diar: dict | None, duration_s: float):
    """Flags describing how fw's own diarization went, and whether to replace it."""
    flags, reasons = [], []
    if not diar:
        return ["fw_diarization_missing"], ["fw.json has no result.diarization"]
    impl = str(diar.get("implementation", ""))
    fb = diar.get("fallback_status")
    if fb not in (None, "not_needed"):
        flags.append(f"fw_diarization_fallback:{fb}")
        reasons.append(f"fw fallback_status={fb} (implementation={impl})")
    if "sortformer" not in impl.lower():
        flags.append(f"fw_diarization_engine:{impl or 'unknown'}")
        reasons.append(f"fw diarization implementation is {impl or 'unknown'}, not Sortformer")
    if duration_s > SORTFORMER_CEILING_S:
        flags.append("over_2h_sortformer_ceiling")
        reasons.append(f"audio is {duration_s / 3600:.2f} h (> 2 h whole-file Sortformer ceiling)")
    blob = json.dumps(diar)
    if "four_lane_capped" in blob or len(speaker_refs(diar)) >= 4:
        flags.append("fw_four_lane_capped")
        reasons.append("fw used all 4 Sortformer lanes; true speaker count may be higher")
    return flags, reasons


def speaker_refs(diar):
    return {t.get("speaker_ref") for t in diar.get("turns") or [] if t.get("speaker_ref")}


def build_block(r: dict, p: Params, fw_diar: dict | None):
    turns = [dict(start_ms=int(round(a * 1000)), end_ms=int(round(b * 1000)), speaker_ref=ref,
                  speaker_confidence=round(conf, 4), overlap_suspected=False)
             for a, b, ref, conf in r["turns"]]
    active = []
    for t in turns:                                  # mark cross-speaker overlap
        active = [x for x in active if x["end_ms"] > t["start_ms"]]
        for x in active:
            if x["speaker_ref"] != t["speaker_ref"]:
                x["overlap_suspected"] = t["overlap_suspected"] = True
        active.append(t)
    ev = {}
    for t in turns:
        e = ev.setdefault(t["speaker_ref"], dict(speaker_ref=t["speaker_ref"], voiced_duration_ms=0,
                                                  _c=0.0, turn_count=0))
        d = t["end_ms"] - t["start_ms"]
        e["voiced_duration_ms"] += d
        e["_c"] += d * t["speaker_confidence"]
        e["turn_count"] += 1
    evidence = []
    for e in sorted(ev.values(), key=lambda e: e["speaker_ref"]):
        e["mean_assignment_confidence"] = round(e.pop("_c") / max(1, e["voiced_duration_ms"]), 4)
        evidence.append(e)
    return dict(
        implementation=IMPLEMENTATION,
        fallback_status="not_needed",
        certification="prototype_uncertified",
        turns=turns,
        speaker_count=dict(
            status="estimated_by_global_linking",
            estimated_speaker_count=len(evidence),
            speaker_evidence=evidence,
            unknown_voiced_share=0.0,
        ),
        longdiar=dict(
            version=__version__,
            chunk_s=p.chunk_s, overlap_s=p.overlap_s, chunks=len(r["chunks"]),
            embedding_model=dict(name=MODEL_NAME, sha256=MODEL_SHA256),
            replaced=dict(implementation=(fw_diar or {}).get("implementation"),
                          fallback_status=(fw_diar or {}).get("fallback_status"),
                          speakers=sorted(speaker_refs(fw_diar or {}))),
        ),
    )


def call_segments(turns, min_len_s=300.0, win_s=120.0):
    """Spans where phone-band speakers (label >= call base) hold most of the speech."""
    if not turns:
        return []
    end = max(t["end_ms"] for t in turns) / 1000
    nb = np.zeros(int(end // win_s) + 1)
    tot = np.zeros_like(nb)
    for t in turns:
        k = int(t["start_ms"] / 1000 // win_s)
        d = (t["end_ms"] - t["start_ms"]) / 1000
        tot[k] += d
        if t.get("_narrow"):
            nb[k] += d
    on = (tot > 0) & (nb / np.maximum(tot, 1e-9) >= 0.5)
    out, k = [], 0
    while k < len(on):
        if on[k]:
            j = k
            while j + 1 < len(on) and (on[j + 1] or (j + 2 < len(on) and on[j + 2])):
                j += 1
            if (j - k + 1) * win_s >= min_len_s:
                out.append((k * win_s, (j + 1) * win_s))
            k = j + 1
        else:
            k += 1
    return out


def summarize(r, block, p: Params):
    cl = r["clusters"]
    units = r["units"]
    turns = block["turns"]
    narrow_refs = {c["ref"] for c in cl if c["narrow"]}
    for t in turns:
        t["_narrow"] = t["speaker_ref"] in narrow_refs
    segs = call_segments(turns)
    for t in turns:
        t.pop("_narrow")
    speakers = []
    for c in sorted(cl, key=lambda c: c["ref"]):
        own = [t for t in turns if t["speaker_ref"] == c["ref"]]
        talk = sum(t["end_ms"] - t["start_ms"] for t in own) / 1000
        speakers.append(dict(
            speaker_ref=c["ref"], talk_s=round(talk, 1),
            first_s=round(min((t["start_ms"] for t in own), default=0) / 1000, 1),
            last_s=round(max((t["end_ms"] for t in own), default=0) / 1000, 1),
            chunks=sorted(c["chunks"]), units=len(c["members"]),
            channel="phone_band" if c["narrow"] else "studio",
            linking_confidence=("low" if talk < 20 or all(units[m].conf <= 0.5 for m in c["members"])
                                else "normal")))
    acc = [l for l in r["links"] if l["accepted"]]
    flags = []
    if any(ch["split_lanes"] for ch in r["chunks"]):
        flags.append("lane_split:more_than_4_voices_in_a_chunk")
    if segs:
        flags.append("phone_band_segment_detected")
    if any(s["linking_confidence"] == "low" for s in speakers):
        flags.append("low_confidence_speakers")
    return dict(
        speakers=speakers,
        call_segments=[dict(start_s=a, end_s=b) for a, b in segs],
        links=dict(considered=len(r["links"]), accepted=len(acc),
                   vetoed=sum(1 for l in r["links"] if l.get("vetoed")),
                   conflicts=sum(1 for l in r["links"] if l.get("conflict")),
                   boundaries=max(0, len(r["chunks"]) - 1),
                   boundaries_with_link=len({l["chunk"] for l in acc})),
        chunks=r["chunks"], cuts=[round(c, 3) for c in r["cuts"]],
        timings=r["timings"], flags=flags,
    )


def run_step(job_dir, fw_json=None, audio=None, mode="auto", in_place=False, params=None,
             model_dir=None, log=None):
    """Run the step on a job dir. Always returns a report dict; never raises."""
    log = log or (lambda m: print(m, file=sys.stderr))
    job = Path(job_dir)
    fw_json = Path(fw_json or job / "fw.json")
    audio = Path(audio or job / "audio.wav")
    p = params or Params()
    rep = dict(step="longdiar", version=__version__, status="warning", reason=None, flags=[],
               fw_json=str(fw_json), audio=str(audio), mode=mode)
    t0 = time.monotonic()
    try:
        doc = json.loads(fw_json.read_text(encoding="utf-8"))
        res = doc.get("result")
        if not isinstance(res, dict):
            raise ValueError("fw.json has no 'result' block")
        from .chunked import Wav
        duration = Wav(audio).duration
        fw_diar = res.get("diarization")
        flags, reasons = fw_health(fw_diar, duration)
        rep["fw_diarization_flags"] = flags
        rep["duration_s"] = round(duration, 3)
        degraded = [f for f in flags if f.startswith(("fw_diarization", "over_2h"))]
        for f in degraded:
            log(f"[longdiar] DEGRADED fw diarization: {f}")
        if mode == "off" or (mode == "auto" and not flags):
            rep.update(status="skipped", reason="fw Sortformer output is within its limits"
                       if mode == "auto" else "mode=off")
            rep["flags"] = [f"fw:{f}" for f in flags]
            return rep
        rep["reason"] = "; ".join(reasons) or "mode=force"
        if p.cache_dir is None:
            p.cache_dir = str(job / "longdiar_cache")
        emb = Embedder(resolve_model(model_dir), threads=p.threads)
        r = diarize(audio, p, emb, log=log)
        block = build_block(r, p, fw_diar)
        rep.update(summarize(r, block, p))
        res["diarization"] = block
        out = job / "fw.longdiar.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, out)
        if in_place:
            backup = fw_json.with_name(fw_json.stem + ".fw-original.json")
            if not backup.exists():
                shutil.copy2(fw_json, backup)
            shutil.copy2(out, fw_json)
            rep["fw_json_backup"] = str(backup)
        rep["output"] = str(out)
        rep["status"] = "ok"
        rep["speaker_count"] = len(rep["speakers"])
    except Exception as exc:  # the step must never fail the video
        rep["status"] = "warning"
        rep["reason"] = f"{type(exc).__name__}: {exc}"
        rep["traceback"] = traceback.format_exc(limit=6)
        rep["flags"].append("longdiar_failed_fw_diarization_kept")
        if any(f.startswith(("fw_diarization", "over_2h"))
               for f in rep.get("fw_diarization_flags", [])):
            rep["flags"].append("DIARIZATION_DEGRADED")
        log(f"[longdiar] WARNING: step failed, keeping fw diarization: {rep['reason']}")
    finally:
        rep["wall_s"] = round(time.monotonic() - t0, 2)
        rep["peak_rss"] = peak_rss_mb()
        rep["flags"] = rep.get("flags", []) + [f"fw:{f}" for f in rep.get("fw_diarization_flags", [])
                                               if f"fw:{f}" not in rep.get("flags", [])]
        try:
            (job / "longdiar.json").write_text(json.dumps(rep, indent=1, default=str) + "\n",
                                               encoding="utf-8")
        except OSError as exc:
            log(f"[longdiar] WARNING: could not write longdiar.json: {exc}")
    return rep
