"""
main.py

CLI entry point: video in -> annotated video + technique report out.

By default, output is organized into a folder named after the input clip:

    outputs/20260915_route1_attempt1_fall/
        annotated.mp4
        report.json

Usage:
    python main.py path/to/20260915_route1_attempt1_fall.mp4
    python main.py path/to/climb.mp4 --out custom_name.mp4 --report custom_report.json
"""

import argparse
import json
import sys
from pathlib import Path

from src.pose_extraction import extract_pose_sequence, smooth_sequence
from src.features import detect_com_over_base_violations
from src.overlay import render_annotated_video


def run(video_path: str, out_path: str, report_path: str, model_path: str):
    # Make sure the output folder exists (out_path and report_path share
    # a parent dir by default — see argument setup below)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Extracting pose from {video_path} ...")
    frames = extract_pose_sequence(video_path, model_path=model_path)
    detected_count = sum(1 for f in frames if f.detected)
    print(f"      {detected_count}/{len(frames)} frames had a detected pose")

    print("[2/4] Smoothing keypoint trajectories ...")
    frames = smooth_sequence(frames)

    print("[3/4] Running technique heuristics ...")
    flags = detect_com_over_base_violations(frames)
    print(f"      Found {len(flags)} COM-over-base flags")

    print(f"[4/4] Rendering annotated video to {out_path} ...")
    render_annotated_video(video_path, out_path, frames, flags)

    report = {
        "video": video_path,
        "total_frames": len(frames),
        "frames_with_detection": detected_count,
        "flags": [
            {
                "frame": fl.frame_idx,
                "timestamp_s": round(fl.timestamp_s, 2),
                "type": fl.flag_type,
                "severity": fl.severity,
                "message": fl.message,
            }
            for fl in flags
        ],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Done. Annotated video: {out_path} | Report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze climbing technique from video.")
    parser.add_argument("video", help="Path to input climbing video")
    parser.add_argument("--outdir", default="outputs",
                         help="Base folder for organized output (default: outputs/)")
    parser.add_argument("--out", default=None,
                         help="Override: exact path for annotated output video "
                              "(skips the auto folder-per-clip behavior)")
    parser.add_argument("--report", default=None,
                         help="Override: exact path for JSON flag report "
                              "(skips the auto folder-per-clip behavior)")
    parser.add_argument("--model", default="pose_landmarker.task",
                         help="Path to the MediaPipe pose_landmarker .task model file "
                              "(see src/pose_extraction.py header for download link)")
    args = parser.parse_args()

    if args.out and args.report:
        out_path, report_path = args.out, args.report
    else:
        # Default: outputs/<clip_name>/annotated.mp4 + report.json
        clip_name = Path(args.video).stem
        clip_dir = Path(args.outdir) / clip_name
        out_path = str(clip_dir / "annotated.mp4")
        report_path = str(clip_dir / "report.json")

    try:
        run(args.video, out_path, report_path, args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)