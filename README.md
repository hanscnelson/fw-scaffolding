# fw-scaffolding
A repo for tools that help to make fw more robust for use with audio longer than 2 hours, audio with more than 4 speakers, or both

## longdiar (prototype)

`longdiar/` is a diarization step that runs **after** `fw transcribe` (FrankenWhisper 0.9.3, Path B argv unchanged). It reads the job's `audio.wav` and `fw.json` and writes `fw.longdiar.json`: a copy of `fw.json` whose `result.diarization` is replaced by chunked Streaming Sortformer turns, with speakers linked across chunks. `fuse_diarization.py` and the 4 cleanup rounds read it unchanged.

How it works:

- ~10 min chunks with 60 s overlap, cut at quiet points; `fw sortformer-diarize` runs on each chunk.
- 3D-Speaker CAM++ embeddings (ONNX) of clean single-lane speech.
- Cleanup inside each chunk: lanes holding both studio and phone-band audio are split by channel, lanes holding two voices are split, and duplicate lanes are merged.
- Saturated studio chunks (>4 voices) are re-run in 3 min windows.
- Global speakers come from overlap must-links, within-chunk cannot-links, and centroid clustering, followed by a per-turn refinement pass.
- Phone-band (earnings-call) speakers are clustered separately and numbered from `SPEAKER_40`.

### Install (casarosie2 / any x86-64 with AVX; no AVX2 needed)

```bash
python3 -m venv /opt/longdiar && /opt/longdiar/bin/pip install -e /path/to/fw-scaffolding
mkdir -p /models/longdiar && curl -L -o /models/longdiar/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx
sha256sum /models/longdiar/*.onnx   # 357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b
```

### Run (after fw, on the job dir)

```bash
LONGDIAR_MODEL_DIR=/models/longdiar /opt/longdiar/bin/longdiar /artifacts/<job_id>        # auto mode
python3 fuse_diarization.py /artifacts/<job_id>/fw.longdiar.json > fused.txt               # if status == ok
```

- The exit code is always 0 (use `--strict` to exit 1 on warnings). Read `longdiar.json`:
  - `status: ok` means use `fw.longdiar.json`.
  - `status: skipped` means fw's Sortformer output was within limits.
  - `status: warning` means keep `fw.json` and raise the listed `flags` as QA flags (`DIARIZATION_DEGRADED` when fw itself had fallen back).
- `--mode auto` replaces fw's diarization when fw used the acoustic fallback, when `fallback_status != not_needed`, when audio is over 2 h, or when all 4 lanes are used. `--mode force` always replaces it.
- `--in-place` copies the result over `fw.json` and keeps `fw.fw-original.json`.
- Sortformer results are cached per chunk in `<job>/longdiar_cache/`, so a re-run only redoes embeddings and linking.

### Test harness (`tools/`, `tests/`)

- `tools/make_synthetic.py` builds LibriSpeech-based test audio with ground truth: a 2h35m, 17-speaker stress file with a phone-band "earnings call", or a 40 min 2-speaker regression clip.
- `tools/run_fw_pathb.sh` runs fw's exact Path B argv on a copy of the audio.
- `tools/score.py` computes DER, name-mergeable DER, chunk consistency, and word-level speaker error on fused output.
- `tools/handcheck.py` picks stretches to check by ear.
- `tests/test_step.py` holds contract tests with a fake `fw`: the failure path gives `warning`, fw.json is untouched, and the output is fuse-compatible.
- `tests/results/` holds the scores and step reports from the prototype runs.
