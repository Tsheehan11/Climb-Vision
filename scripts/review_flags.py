"""
review_flags.py

Non-interactive counterpart to review_flags.sh. For every COM-over-base
flag in a clip it writes:

    outputs/<clip>/flag_review/flagNN_frameFFFF.jpg   contact sheet of frames
                                                      around the flag, with the
                                                      skeleton, COM line and
                                                      base-of-support band drawn on
    outputs/<clip>/flag_review/diagnostics.csv        the numbers behind each flag

The contact sheets are what you (or Claude) look at to judge each flag;
the diagnostics explain *why* the heuristic fired (which side, how long,
how confident the foot/hip landmarks were, whether the climber was
already falling, etc).

Usage:
    python scripts/review_flags.py videos/20260915_route1_attempt1_fall.mp4
    python scripts/review_flags.py videos/*.mp4 --context 8 --step 4
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pose_extraction import extract_pose_sequence, smooth_sequence, FrameData  # noqa: E402
from src.features import (  # noqa: E402
    estimate_center_of_mass, base_of_support_x_range, _body_scale,
)
from src.overlay import SKELETON_EDGES  # noqa: E402
from src.analysis import analyze  # noqa: E402

MARGIN_RATIO = 0.15
MIN_DURATION = 3
FOOT_NAMES = ("LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX")
TORSO_NAMES = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP")


def frame_metrics(f: FrameData):
    """COM / base / scale for one frame, or None if the heuristic skips it."""
    if not f.detected:
        return None
    com = estimate_center_of_mass(f.landmarks)
    base = base_of_support_x_range(f.landmarks)
    scale = _body_scale(f.landmarks)
    if com is None or base is None or not scale:
        return None
    margin = MARGIN_RATIO * scale
    lo, hi = base[0] - margin, base[1] + margin
    return {"com": com, "base": base, "scale": scale, "lo": lo, "hi": hi,
            "violation": com[0] < lo or com[0] > hi}


def violation_runs(metrics):
    """Same run logic as features.detect_com_over_base_violations, but keeps
    the start/end of each run instead of only the peak frame."""
    runs, cur = [], []
    for i, m in enumerate(metrics):
        if m and m["violation"]:
            cur.append(i)
        else:
            if len(cur) >= MIN_DURATION:
                runs.append((cur[0], cur[-1]))
            cur = []
    if len(cur) >= MIN_DURATION:
        runs.append((cur[0], cur[-1]))
    return runs


def load_or_extract(video: Path, clip_dir: Path, model: str):
    cache = clip_dir / "keypoints.json"
    if cache.exists():
        raw = json.loads(cache.read_text())
        return [FrameData(r["i"], r["t"], {k: tuple(v) for k, v in r["lm"].items()}, r["d"])
                for r in raw]
    frames = extract_pose_sequence(str(video), model_path=model)
    cache.write_text(json.dumps(
        [{"i": f.frame_idx, "t": f.timestamp_s, "lm": f.landmarks, "d": f.detected}
         for f in frames]))
    return frames


def draw(frame, f: FrameData, m, label, flagged):
    h, w = frame.shape[:2]
    t = max(2, w // 300)
    if f.detected:
        pts = {n: (int(v[0] * w), int(v[1] * h)) for n, v in f.landmarks.items()}
        for a, b in SKELETON_EDGES:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), t)
        for n, p in pts.items():
            # red dot = low-confidence landmark
            col = (0, 0, 255) if f.landmarks[n][3] < 0.5 else (0, 200, 255)
            cv2.circle(frame, p, t * 2, col, -1)
    if m:
        # blue band = base of support (+margin); yellow line = COM x
        lo, hi = int(m["lo"] * w), int(m["hi"] * w)
        ov = frame.copy()
        cv2.rectangle(ov, (lo, 0), (hi, h), (255, 120, 0), -1)
        cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
        cv2.line(frame, (lo, 0), (lo, h), (255, 120, 0), t)
        cv2.line(frame, (hi, 0), (hi, h), (255, 120, 0), t)
        cx, cy = int(m["com"][0] * w), int(m["com"][1] * h)
        cv2.line(frame, (cx, 0), (cx, h), (0, 255, 255), t)
        cv2.circle(frame, (cx, cy), t * 4, (0, 255, 255), -1)
    return frame


def crop_box(fs, w, h, pad=0.12, min_frac=0.45):
    """Pixel box (x0, y0, x1, y1) around the skeleton across the given frames,
    so the climber fills the tile instead of being a speck in a wide shot."""
    xs = [v[0] for f in fs if f.detected for v in f.landmarks.values()]
    ys = [v[1] for f in fs if f.detected for v in f.landmarks.values()]
    if not xs:
        return (0, 0, w, h)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_w = max(min_frac, max(xs) - min(xs) + 2 * pad) / 2
    half_h = max(min_frac, max(ys) - min(ys) + 2 * pad) / 2
    # keep the box inside the frame without shrinking it
    cx = min(max(cx, half_w), 1 - half_w) if half_w < 0.5 else 0.5
    cy = min(max(cy, half_h), 1 - half_h) if half_h < 0.5 else 0.5
    return (int(max(0, cx - half_w) * w), int(max(0, cy - half_h) * h),
            int(min(1, cx + half_w) * w), int(min(1, cy + half_h) * h))


def review_clip(video: Path, outdir: Path, model: str, context: int, step: int, tile_h: int,
                include_off_wall: bool = False):
    clip_dir = outdir / video.stem
    review_dir = clip_dir / "flag_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    raw = load_or_extract(video, clip_dir, model)
    frames = smooth_sequence(raw)
    metrics = [frame_metrics(f) for f in frames]
    if not include_off_wall:
        # same gating main.py applies: only frames with a climber on the wall
        on_wall = analyze(raw, fps).kin.on_wall
        metrics = [m if on_wall[i] else None for i, m in enumerate(metrics)]
    runs = violation_runs(metrics)
    n = len(frames)

    # decode every frame we need in one sequential pass (seeking is unreliable)
    peaks = []
    for start, end in runs:
        def overshoot(i):
            m = metrics[i]
            return min(abs(m["com"][0] - m["lo"]), abs(m["com"][0] - m["hi"])) / (m["scale"] + 1e-6)
        peak = max(range(start, end + 1), key=overshoot)
        peaks.append((start, end, peak, min(1.0, overshoot(peak))))

    wanted = {}
    for k, (_, _, peak, _) in enumerate(peaks):
        for off in range(-context, context + 1, step):
            i = peak + off
            if 0 <= i < n:
                wanted.setdefault(i, None)
    idx = 0
    while wanted and idx <= max(wanted):
        ok, img = cap.read()
        if not ok:
            break
        if idx in wanted:
            wanted[idx] = img
        idx += 1
    cap.release()

    rows = []
    for k, (start, end, peak, sev) in enumerate(peaks, 1):
        tiles = []
        offs = [o for o in range(-context, context + 1, step)
                if wanted.get(peak + o) is not None]
        box = crop_box([frames[peak + o] for o in offs], *wanted[peak].shape[1::-1])
        for off in offs:
            i = peak + off
            img = draw(wanted[i].copy(), frames[i], metrics[i], "", off == 0)
            img = np.ascontiguousarray(img[box[1]:box[3], box[0]:box[2]])
            fs = img.shape[1] / 700
            cv2.rectangle(img, (0, 0), (img.shape[1], int(60 * fs)),
                          (0, 0, 200) if off == 0 else (40, 40, 40), -1)
            cv2.putText(img, f"f{i} {'FLAG' if off == 0 else f'{off:+d}'}", (10, int(42 * fs)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2 * fs, (255, 255, 255), max(2, int(3 * fs)))
            sc = tile_h / img.shape[0]
            tiles.append(cv2.resize(img, (int(img.shape[1] * sc), tile_h)))
        sheet = review_dir / f"flag{k:02d}_frame{peak:04d}.jpg"
        cv2.imwrite(str(sheet), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])

        m, f = metrics[peak], frames[peak]
        rf = raw[peak]
        vis = lambda names: min((rf.landmarks[x][3] for x in names if x in rf.landmarks), default=0.0)
        win = range(max(0, peak - context), min(n, peak + context + 1))
        # vertical COM speed in body-lengths/sec over +-5 frames (positive = moving down)
        a, b = max(0, peak - 5), min(n - 1, peak + 5)
        vy = None
        if metrics[a] and metrics[b] and b > a:
            vy = (metrics[b]["com"][1] - metrics[a]["com"][1]) / m["scale"] / ((b - a) / fps)
        hip_y = (f.landmarks["LEFT_HIP"][1] + f.landmarks["RIGHT_HIP"][1]) / 2
        sh_y = (f.landmarks["LEFT_SHOULDER"][1] + f.landmarks["RIGHT_SHOULDER"][1]) / 2
        foot_y = max(f.landmarks[x][1] for x in FOOT_NAMES if x in f.landmarks)
        rows.append({
            "clip": video.stem, "flag_no": k, "frame": peak,
            "timestamp_s": round(peak / fps, 2), "severity": round(sev, 3),
            "run_start": start, "run_end": end, "run_len_frames": end - start + 1,
            "run_len_s": round((end - start + 1) / fps, 2),
            "side": "left_of_base" if m["com"][0] < m["lo"] else "right_of_base",
            "com_x": round(m["com"][0], 3), "com_y": round(m["com"][1], 3),
            "base_lo_x": round(m["base"][0], 3), "base_hi_x": round(m["base"][1], 3),
            "base_width_body": round((m["base"][1] - m["base"][0]) / m["scale"], 2),
            "body_scale": round(m["scale"], 3),
            "torso_upright": sh_y < hip_y,
            "lowest_foot_y": round(foot_y, 3),
            "min_foot_visibility": round(vis(FOOT_NAMES), 2),
            "min_torso_visibility": round(vis(TORSO_NAMES), 2),
            "detected_frac_window": round(sum(frames[i].detected for i in win) / len(win), 2),
            "com_vy_body_per_s": None if vy is None else round(vy, 2),
            "sheet": sheet.name,
        })

    with open(review_dir / "diagnostics.csv", "w", newline="") as fh:
        if rows:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    det = sum(f.detected for f in frames)
    print(f"{video.stem}: {n} frames @ {fps:.1f}fps, {det} detected, {len(rows)} flags -> {review_dir}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build contact sheets + diagnostics for each flag.")
    ap.add_argument("videos", nargs="+", help="Input video(s); globs are expanded")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--model", default="pose_landmarker.task")
    ap.add_argument("--context", type=int, default=8, help="frames either side of the flag")
    ap.add_argument("--step", type=int, default=4, help="spacing between tiles, in frames")
    ap.add_argument("--tile-height", type=int, default=640)
    ap.add_argument("--include-off-wall", action="store_true",
                    help="review every raw flag, as v1 did, instead of on-wall frames only")
    args = ap.parse_args()

    paths = [Path(p) for v in args.videos for p in (glob.glob(v) or [v])]
    for p in paths:
        review_clip(p, Path(args.outdir), args.model, args.context, args.step, args.tile_height,
                    args.include_off_wall)
