# Stack lock (Chester owns)

Parity target: live Path B on the Grok Bot box — `/workspace/ingest/batch52/run_batch.py`.
**Do not** substitute faster-whisper, pyannote, HuggingFace diarize, or a different Whisper binary.

## Binary + models

| Field | Value |
|---|---|
| Tool | `franken_whisper` (symlink/`PATH` alias `fw`) |
| Pin | **0.9.3** (ref box sha256 `3e449b1366fc5ea4cdc832e05ef2f1b78e0e4402b2a49904627575ec53104d08`, 2026-09-16) |
| Install | official install script from `Dicklesworthstone/franken_whisper`, then `fw pull all --json` |
| Models | Whisper `ggml-large-v3-turbo` + NVIDIA Streaming Sortformer (~2.1 GB). Prefer compose volume `/models` (`FRANKEN_WHISPER_MODEL_DIR=/models`) so images do not re-pull |
| Host/image deps | `ffmpeg` **pin n7.1.5** (BtbN static `ffmpeg-n7.1.5-12-g1fdbca85aa-linux64-gpl-7.1`, sha256 `c1e6caf4…f0e79`; Path B ref Debian `7.1.5-0+deb13u1`), `yt-dlp`. No Python ASR |

Verify: `fw --version` → 0.9.3; `fw robot triage`; `fw models --json`.

### Post-fw diarization step (`longdiar`, fw-scaffolding) — runs after `fw`, never inside it

| Field | Value |
|---|---|
| Tool | `python -m longdiar <job_dir>` from `hanscnelson/fw-scaffolding` (`longdiar/`), version 0.1.0 |
| Calls into fw | `fw sortformer-diarize --input <chunk.wav>` (same pinned 0.9.3 binary and verified Sortformer cache; one ~10 min chunk at a time) |
| Speaker-embedding model | 3D-Speaker CAM++ `3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx`, Apache-2.0, from `k2-fsa/sherpa-onnx` release `speaker-recongition-models` (GitHub, no HF token), sha256 `357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b`, 29.6 MB, at `/models/longdiar/` |
| Python | CPython 3.12 venv; `numpy==2.5.3`, `onnxruntime==1.30.0` (CPU EP). Runtime SIMD dispatch: AVX on the E5-2670, no AVX2/FMA needed |
| Not used | pyannote (code or weights), HF tokens, faster-whisper, GPUs |

Verify: `python -m longdiar --help`; `sha256sum /models/longdiar/*.onnx`.

## Fetch

1. `yt-dlp` bestaudio (m4a/webm/…)
2. Exact convert argv (Path B / bit-identical PCM):

```bash
ffmpeg -y -i <in> -ac 1 -ar 16000 <job>/audio.wav
```

- Output is PCM s16le mono 16 kHz WAV (ffmpeg wav default; do **not** add extra encode flags that diverge from Path B).
- Image must use **ffmpeg 7.1.5** (Path B). Ubuntu 24.04 apt ships 6.1.1 and is **not** bit-identical (±1 LSB on resample). Dockerfile installs BtbN static n7.1.5 (verified bit-identical vs Path B for the convert argv above).

## Exact `fw` argv (one job at a time)

```bash
export RAYON_NUM_THREADS=8
export FW_RETRY_FAILED_WINDOW=1
# budgets ≈ max(900_000, duration_sec * 3000) ms each
export FRANKEN_WHISPER_STAGE_BUDGET_BACKEND_MS=<budget_ms>
export FRANKEN_WHISPER_STAGE_BUDGET_DIARIZE_MS=<budget_ms>
export FRANKEN_WHISPER_MODEL_DIR=/models

taskset -c 0-7 fw transcribe \
  --input /artifacts/<job_id>/audio.wav \
  --backend whisper-cpp \
  --threads 8 \
  --processors 1 \
  --diarize \
  --diarization-engine sortformer \
  --language en \
  --no-persist \
  --max-segment-length 90 \
  --split-on-word \
  --json \
  --timeout <timeout_sec>
```

- stdout → `fw.json`  |  stderr → `fw.err`
- If stderr contains `FW-DTW-PROJECTION-PARENT-DURATION` and `--split-on-word` was used: **rerun identical argv without** `--split-on-word`
- Worker concurrency for ASR: **1**. Compose `cpus: 8` on casarosie2

## Diarization / naming

- Sortformer is **4-lane capped**. More speakers → QA flag in `quality.md`, do **not** hard-reject before ASR
- After `fw`, run `longdiar` (auto mode). It replaces `result.diarization` in a copy (`fw.longdiar.json`) when fw's diarization is degraded or capped: audio > 2 h, `fallback_status != not_needed` / acoustic engine, or all 4 lanes used. `status: warning` in `longdiar.json` → QA flag, keep fw's output, never fail the job. An fw acoustic fallback is always surfaced as a flag, never silent
- `speaker_map.json` **flat** shape only: `{"SPEAKER_00":"Alex","SPEAKER_01":"Hans C Nelson",…}` (cluster id → display_name string). No nested objects.
- Naming **prefers real names** from job/show priors + channel metadata (e.g. Hans C Nelson / Alex). Fall back to `Host` / leave `SPEAKER_XX` **only** when priors are missing. Do **not** invent surnames from thin air; **do** use explicit priors when provided.
- Prior sources (merged by `resolve_show_priors`):
  1. Job `show_priors` / `metadata` on `POST /jobs`
  2. Show config JSON under `$SHOW_CONFIG_DIR` (default `data/shows/*.json`)
  3. Built-in Path B `CHANNEL_HOST_PRIORS` (channel substring → host roster)
- Thin/low-confidence map → QA flag; still emit best-effort `cleaned.md`
- Deep links (`youtu.be/?t=`): **deferred** pending Chester confirmation — do not require in cleanup v1

### Show priors schema

Job body:

```json
{
  "source": "https://youtube.com/watch?v=…",
  "show_priors": {
    "channel": "My Show",
    "hosts": ["Hans C Nelson", "Alex"],
    "guests": [],
    "speaker_hints": {"SPEAKER_00": "Alex", "SPEAKER_01": "Hans C Nelson"}
  }
}
```

Show config file (`data/shows/<slug>.json`):

```json
{
  "channel_match": ["my show"],
  "hosts": ["Hans C Nelson", "Alex"],
  "guests": [],
  "speaker_hints": {}
}
```

## Cleanup (4-round, outside `fw`)

Required outputs:

| File | Role |
|---|---|
| `cleaned.md` | Named-speaker publishable transcript |
| `quality.md` | Quality / retention / QA flags |
| `speaker_map.json` | Cluster → display name |

v1 cleanup model env: `CLEANUP_MODEL` / `CLEANUP_EFFORT` (ref ingest: Grok Build `grok-4.6`). Notion is an optional sink later — **not** in the API.

## Artifact dir

```
/artifacts/<job_id>/
  meta.json
  audio.wav
  fw.json
  fw.err
  show_priors.json
  cleaned.md
  quality.md
  speaker_map.json
```

## Forbidden

- faster-whisper / whisper.cpp brew / pyannote / HF-token diarize
- Omarchy (or desktop) as container base — Ubuntu 24.04 like CTS
- Notion in the hot path

## Host note (casarosie2 / Xeon E5-2670)

Official Linux amd64 **prebuilts require x86-64-v3 (AVX2/BMI2/FMA)**. This VM CPU only exposes AVX (no AVX2) — prebuilts SIGILL. Install with:

```bash
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_whisper/main/install.sh | bash -s -- --from-source
```

Docker image for this host must likewise be **built from source on this CPU** (or a matching baseline), not the official release binary. Prefer baking that binary + mounting `/models`.
