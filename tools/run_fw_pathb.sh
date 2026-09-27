#!/usr/bin/env bash
# Run fw 0.9.3 with the STACK_LOCK Path B argv on a COPY of an audio file (test harness).
#   run_fw_pathb.sh <audio.wav> <job_dir> [threads] [extra fw args...]
# Writes <job_dir>/{audio.wav,fw.json,fw.err,time.txt}. Includes the Path B
# FW-DTW-PROJECTION-PARENT-DURATION retry without --split-on-word.
set -euo pipefail
src=$1; job=$2; threads=${3:-8}; shift 3 || shift $#
mkdir -p "$job"
cp -f "$src" "$job/audio.wav"
dur=$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$job/audio.wav")
budget=$(python3 -c "print(max(900000, int(float('$dur')*3000)))")
export RAYON_NUM_THREADS=$threads FW_RETRY_FAILED_WINDOW=1
export FRANKEN_WHISPER_STAGE_BUDGET_BACKEND_MS=$budget FRANKEN_WHISPER_STAGE_BUDGET_DIARIZE_MS=$budget
last=$((threads - 1))
run() {
  /usr/bin/time -v -o "$job/time.txt" taskset -c 0-$last fw transcribe --input "$job/audio.wav" \
    --backend whisper-cpp --threads "$threads" --processors 1 "$@" --language en --no-persist \
    --max-segment-length 90 ${SPLIT:---split-on-word} --json > "$job/fw.json" 2> "$job/fw.err" || true
}
run --diarize --diarization-engine sortformer "$@"
if grep -q FW-DTW-PROJECTION-PARENT-DURATION "$job/fw.err"; then
  SPLIT=" " run --diarize --diarization-engine sortformer "$@"
fi
grep -E "Elapsed|Maximum resident" "$job/time.txt"
