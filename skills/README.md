# Skills

`jsm_transcript_skills_2026-09-27.zip` is the archive of skill files used to make speaker-diarized transcripts of YouTube podcasts with FrankenWhisper (`fw`) and Sortformer diarization, plus the 4-round cleanup. It is stored as a zip. `MANIFEST.md` inside the archive explains each skill and where it came from.

## Skills

- **transcribing-youtube-videos** — Download the YouTube source, run FrankenWhisper Path B, fuse Sortformer diarization into a speaker-labeled transcript, and audit cleanup quality.
- **transcribing-audio-from-calls-and-meetings** — Run the 4-round cleanup: clean and attribute speakers, verify what was kept, then a fresh-eyes pass for the speaker map and `[sic]` marks.
- **fw** — Operator reference for the FrankenWhisper binary: CLI flags, Sortformer diarization (including the 4-lane cap), and troubleshooting.
- **grok-build-cleanup-pipeline** — Orchestrate ASR through the Grok Build cleanup and the ready-ping queue into Notion Transcript Ingest.
- **choose-the-best-skills-for-me-to-run-in-this-project** — Rank and install skills from the `jsm` catalog. It sits beside the transcript flow as a skill selector.

## casarosie2

The `casarosie2/` folder inside the zip holds reference copies of the pipeline code (`cursor_backend.py`, `priors.py`, `stages/fw.py`, `stages/cleanup.py`, and `docs/STACK_LOCK.md`). Those copies contain no secrets.
