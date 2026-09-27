"""CLI: python -m longdiar JOB_DIR [options]. Exit code is 0 unless --strict."""
import argparse
import json
import sys

from .chunked import Params
from .step import run_step


def main(argv=None):
    ap = argparse.ArgumentParser(prog="longdiar", description=__doc__)
    ap.add_argument("job_dir", help="dir holding audio.wav and fw.json")
    ap.add_argument("--fw-json")
    ap.add_argument("--audio")
    ap.add_argument("--mode", choices=["auto", "force", "off"], default="auto",
                    help="auto: replace fw diarization only when it is degraded or capped "
                         "(>2 h, acoustic fallback, 4 lanes used); force: always")
    ap.add_argument("--in-place", action="store_true",
                    help="also copy the result over fw.json (original kept as fw.fw-original.json)")
    ap.add_argument("--model-dir", help="dir with the CAM++ ONNX (default $LONGDIAR_MODEL_DIR "
                                        "or /models/longdiar)")
    ap.add_argument("--fw-bin", default="fw")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--chunk-s", type=float, default=Params.chunk_s)
    ap.add_argument("--overlap-s", type=float, default=Params.overlap_s)
    ap.add_argument("--merge-cos", type=float, default=Params.merge_cos)
    ap.add_argument("--strict", action="store_true", help="exit 1 when status is warning")
    a = ap.parse_args(argv)
    p = Params(chunk_s=a.chunk_s, overlap_s=a.overlap_s, merge_cos=a.merge_cos,
               merge_cos_narrow=a.merge_cos, fw_bin=a.fw_bin, threads=a.threads)
    rep = run_step(a.job_dir, a.fw_json, a.audio, a.mode, a.in_place, p, a.model_dir)
    brief = {k: rep.get(k) for k in ("status", "reason", "speaker_count", "flags", "call_segments",
                                     "wall_s", "peak_rss", "output")}
    print(json.dumps(brief, indent=1, default=str))
    return 1 if a.strict and rep["status"] == "warning" else 0


if __name__ == "__main__":
    sys.exit(main())
