"""
overlay.py

Draws the extracted skeleton and technique flags back onto the source
video, so output is something you can actually watch and learn from,
not just a CSV of numbers.
"""

import cv2
from src.pose_extraction import FrameData, RELEVANT_LANDMARKS
from src.features import TechniqueFlag

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


def render_annotated_video(
    video_path: str,
    out_path: str,
    frames: list[FrameData],
    flags: list[TechniqueFlag],
    flag_persist_s: float = 1.0,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    flags_by_frame = {f.frame_idx: f for f in flags}
    # Also mark a short window after each flag so it's visible/readable
    persist_frames = int(flag_persist_s * fps)
    flagged_ranges = []
    for fl in flags:
        flagged_ranges.append((fl.frame_idx, fl.frame_idx + persist_frames, fl))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        fdata = frames[frame_idx] if frame_idx < len(frames) else None

        if fdata and fdata.detected:
            pts = {}
            for name in RELEVANT_LANDMARKS:
                if name in fdata.landmarks:
                    x, y, z, v = fdata.landmarks[name]
                    pts[name] = (int(x * w), int(y * h))

            for a, b in SKELETON_EDGES:
                if a in pts and b in pts:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for p in pts.values():
                cv2.circle(frame, p, 4, (0, 200, 255), -1)

        active_flag = None
        for start, end, fl in flagged_ranges:
            if start <= frame_idx <= end:
                active_flag = fl
                break

        if active_flag:
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 180), -1)
            cv2.putText(frame, f"! {active_flag.message}", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
