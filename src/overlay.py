"""
overlay.py

Draws the extracted skeleton and the analysis back onto the source video,
so output is something you can actually watch and learn from, not just a
CSV of numbers.

On every frame: a status strip (is a climber being tracked, are they on
the wall, how unstable). While something is happening: a banner naming the
instability event, or the fall / dismount and why it was called that.
"""

import cv2
import numpy as np

from src.pose_extraction import RELEVANT_LANDMARKS

# Bone connections for the subset of landmarks we track
SKELETON_EDGES = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"), ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"), ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
]

EVENT_LABELS = {
    "off_base": "UNSTABLE: weight off the feet",
    "feet_cut": "UNSTABLE: feet cut loose",
    "jolt": "UNSTABLE: lurch",
    "stall": "HOLDING POSITION",
}
GREEN, AMBER, RED, GREY, WHITE = (0, 200, 0), (0, 170, 255), (0, 0, 220), (90, 90, 90), (255, 255, 255)


def _score_color(score: float):
    return GREEN if score < 0.35 else AMBER if score < 0.5 else RED


def render_annotated_video(video_path: str, out_path: str, analysis, drop_persist_s: float = 2.5):
    """`analysis` is the src.analysis.Analysis for this video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    kin, st, frames = analysis.kin, analysis.stability, analysis.frames
    fs = w / 1080            # layout was sized for 1080-wide video
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = max(1, int(round(2 * fs)))
    persist = int(drop_persist_s * fps)

    def banner(frame, text, color, row=0):
        y0 = int(row * 70 * fs)
        cv2.rectangle(frame, (0, y0), (w, y0 + int(70 * fs)), color, -1)
        cv2.putText(frame, text, (int(14 * fs), y0 + int(48 * fs)), font, 1.1 * fs, WHITE, thick + 1)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i = frame_idx
        in_range = i < kin.n
        tracked = in_range and kin.present[i]
        fdata = frames[i] if i < len(frames) else None

        # skeleton: full colour when it is a real tracked climber, grey when the
        # pose model fired on something the analysis ignored
        if fdata and fdata.detected:
            pts = {name: (int(fdata.landmarks[name][0] * w), int(fdata.landmarks[name][1] * h))
                   for name in RELEVANT_LANDMARKS if name in fdata.landmarks}
            bone, joint = ((0, 255, 0), (0, 200, 255)) if tracked else (GREY, GREY)
            for a, b in SKELETON_EDGES:
                if a in pts and b in pts:
                    cv2.line(frame, pts[a], pts[b], bone, thick)
            for p in pts.values():
                cv2.circle(frame, p, int(4 * fs) + 1, joint, -1)
            if tracked and not np.isnan(kin.com[i, 0]):
                cv2.circle(frame, (int(kin.com[i, 0] * w), int(kin.com[i, 1] * h)),
                           int(9 * fs) + 1, (0, 255, 255), -1)

        # banners: a drop outranks an instability event
        drop = next((d for d in analysis.drops
                     if d.release_frame <= i <= d.landing_frame + persist), None)
        events = [e for e in analysis.events if e.start_frame <= i <= e.end_frame]
        unstable = next((e for e in events if e.kind != "stall"), None)
        row = 0
        if drop:
            banner(frame, f"{drop.kind.upper()} at {drop.release_s:.1f}s",
                   RED if drop.kind == "fall" else GREEN)
            banner(frame, drop.reasons[0], (40, 40, 40), row=1)
            row = 2
        elif unstable:
            banner(frame, EVENT_LABELS[unstable.kind], RED)
            row = 1
        if not drop and any(e.kind == "stall" for e in events):
            banner(frame, f"{EVENT_LABELS['stall']} {st.stalled_s[i]:.0f}s", (120, 80, 0), row=row)

        # status strip along the bottom
        y1 = h - int(20 * fs)
        y0 = y1 - int(50 * fs)
        cv2.rectangle(frame, (0, y0 - int(14 * fs)), (w, h), (25, 25, 25), -1)
        on_wall = in_range and kin.on_wall[i]
        airborne = any(d.release_frame <= i <= d.landing_frame for d in analysis.drops)
        state = ("AIRBORNE" if airborne else "ON WALL" if on_wall
                 else "ON GROUND" if tracked else "NO CLIMBER")
        cv2.putText(frame, state, (int(14 * fs), y1 - int(8 * fs)), font, 1.0 * fs, WHITE, thick)
        if on_wall and not np.isnan(st.score[i]):
            x0, x1 = int(330 * fs), w - int(20 * fs)
            cv2.rectangle(frame, (x0, y0), (x1, y1), GREY, -1)
            cv2.rectangle(frame, (x0, y0), (x0 + int((x1 - x0) * st.score[i]), y1),
                          _score_color(st.score[i]), -1)
            cv2.putText(frame, f"instability {st.score[i]:.2f}", (x0 + int(10 * fs), y1 - int(12 * fs)),
                        font, 0.9 * fs, WHITE, thick)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
