"""
main.py

CLI entry point: video in -> annotated video + technique report out.

By default, output is organized into a folder named after the input clip:

    outputs/20260915_route1_attempt1_fall/
        annotated.mp4
        report.json

The report covers when the climber was on the wall, how unstable they
were while there, any drops off the wall (fall vs. deliberate dismount),
and what the instability looked like in the lead-up to each drop.

Usage:
    python main.py path/to/20260915_route1_attempt1_fall.mp4
    python main.py path/to/climb.mp4 --out custom_name.mp4 --report custom_report.json
    python main.py path/to/climb.mp4 --no-video      # report only, much faster
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

from src.analysis import analyze
from src.overlay import render_annotated_video
from src.pose_extraction import extract_pose_sequence


def run(video_path: str, out_path: str, report_path: str, model_path: str, render: bool = True):
    # Make sure the output folder exists (out_path and report_path share
    # a parent dir by default — see argument setup below)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Extracting pose from {video_path} ...")
    raw = extract_pose_sequence(video_path, model_path=model_path)
    detected_count = sum(1 for f in raw if f.detected)
    print(f"      {detected_count}/{len(raw)} frames had a detected pose")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    print("[2/4] Finding the climber, scoring stability, detecting drops ...")
    result = analyze(raw, fps)
    s = result.summary
    print(f"      On the wall for {s['on_wall_s']}s, unstable for {s['unstable_s']}s "
          f"({s['unstable_pct_of_on_wall']}%), {len(result.events)} events")

    print("[3/4] Outcome ...")
    if not result.drops:
        print(f"      {s['outcome']}")
    for d in result.drops:
        print(f"      {d.kind.upper()} at {d.release_s}s: {'; '.join(d.reasons)}")

    if render:
        print(f"[4/4] Rendering annotated video to {out_path} ...")
        render_annotated_video(video_path, out_path, result)
    else:
        print("[4/4] Skipping annotated video (--no-video)")

    report = {
        "video": video_path,
        "total_frames": len(raw),
        "frames_with_detection": detected_count,
        **result.to_report(),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Done. Report: {report_path}" + (f" | Annotated video: {out_path}" if render else ""))


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
    parser.add_argument("--no-video", action="store_true",
                         help="Write the report only; skip rendering the annotated video")
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
        run(args.video, out_path, report_path, args.model, render=not args.no_video)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
