"""Fast contract tests with a fake `fw` binary (no models beyond CAM++ needed).

  LONGDIAR_MODEL_DIR=/models/longdiar FUSE=/path/to/fuse_diarization.py python -m pytest tests -q
"""
import hashlib
import json
import os
import stat
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from longdiar.chunked import Params  # noqa: E402
from longdiar.step import run_step  # noqa: E402

SR = 16000
FAKE_FW = r'''#!/usr/bin/env python3
import json, sys, wave
f = sys.argv[sys.argv.index("--input") + 1]
n = wave.open(f).getnframes() / 16000
if open(__file__).read().rstrip().endswith("FAIL"):
    sys.exit("boom")
turns, t, lane = [], 0.5, 0
while t < n - 3:
    turns.append(dict(start_seconds=t, end_seconds=min(n, t + 2.5), speaker_lane=lane))
    t += 3.0
    lane = 1 - lane
print(json.dumps(dict(status="ok", result=dict(turns=turns, active_lane_count=2,
      capacity=dict(status="within_capacity"), overlap_frames=0))))
#MODE OK
'''


def tone_voice(f0, n):
    t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    return (0.1 * x * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


@pytest.fixture
def job(tmp_path):
    dur = 90.0
    x = np.zeros(int(dur * SR), np.float32)
    t, lane = 0.5, 0
    while t < dur - 3:
        i = int(t * SR)
        x[i: i + int(2.5 * SR)] = tone_voice((120, 210)[lane], int(2.5 * SR))
        t += 3.0
        lane = 1 - lane
    with wave.open(str(tmp_path / "audio.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    segs, words = [], []
    t = 0.6
    while t < dur - 3:
        for k in range(4):
            segs.append(dict(start_sec=t + 0.5 * k, end_sec=t + 0.5 * k + 0.4, text="w."))
            words.append("word." if k == 3 else "word")
        t += 3.0
    fw = dict(result=dict(transcript=" ".join(words), segments=segs, diarization=dict(
        implementation="native-acoustic-v1", fallback_status="sortformer_ineligible",
        turns=[dict(start_ms=500, end_ms=int(dur * 1000), speaker_ref="SPEAKER_00",
                    speaker_confidence=0.5, overlap_suspected=False)])))
    (tmp_path / "fw.json").write_text(json.dumps(fw))
    fake = tmp_path / "fakefw"
    fake.write_text(FAKE_FW)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    return tmp_path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def params(job, **kw):
    return Params(chunk_s=40, overlap_s=10, snap_s=2, fw_bin=str(job / "fakefw"), **kw)


def test_ok_block_is_fuse_compatible(job):
    before = sha(job / "fw.json")
    rep = run_step(job, params=params(job), mode="auto")
    assert rep["status"] == "ok", rep
    assert sha(job / "fw.json") == before
    d = json.loads((job / "fw.longdiar.json").read_text())["result"]["diarization"]
    for t in d["turns"]:
        assert {"start_ms", "end_ms", "speaker_ref", "speaker_confidence",
                "overlap_suspected"} <= set(t)
        assert t["speaker_ref"].startswith("SPEAKER_")
    assert d["fallback_status"] == "not_needed"
    assert "fw:fw_diarization_fallback:sortformer_ineligible" in rep["flags"]
    fuse = os.environ.get("FUSE")
    if fuse:
        out = subprocess.run([sys.executable, fuse, str(job / "fw.longdiar.json")],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "SPEAKER_" in out.stdout


def test_fw_failure_is_warning_not_error(job):
    fake = job / "fakefw"
    fake.write_text(FAKE_FW.replace("#MODE OK", "#MODE FAIL"))
    before = sha(job / "fw.json")
    rep = run_step(job, params=params(job), mode="force")
    assert rep["status"] == "warning"
    assert "DIARIZATION_DEGRADED" in rep["flags"]
    assert sha(job / "fw.json") == before
    assert not (job / "fw.longdiar.json").exists()
    assert json.loads((job / "longdiar.json").read_text())["status"] == "warning"


def test_missing_model_is_warning(job, tmp_path_factory):
    rep = run_step(job, params=params(job), mode="force",
                   model_dir=str(tmp_path_factory.mktemp("nomodel")))
    assert rep["status"] == "warning" and "FileNotFoundError" in rep["reason"]


def test_in_place_keeps_backup(job):
    before = sha(job / "fw.json")
    rep = run_step(job, params=params(job), mode="force", in_place=True)
    assert rep["status"] == "ok"
    assert sha(job / "fw.fw-original.json") == before
    assert sha(job / "fw.json") == sha(job / "fw.longdiar.json")


def test_skip_when_fw_sortformer_is_healthy(job):
    d = json.loads((job / "fw.json").read_text())
    d["result"]["diarization"].update(implementation="native-sortformer-v1",
                                      fallback_status="not_needed")
    (job / "fw.json").write_text(json.dumps(d))
    rep = run_step(job, params=params(job), mode="auto")
    assert rep["status"] == "skipped"
