"""
pose_extraction.py

Wraps MediaPipe's PoseLandmarker (Tasks API — the current API as of
mediapipe 0.10.x; the older `mp.solutions.pose` API is deprecated and
no longer available) to extract per-frame body keypoints from a climbing
video. Assumes a static/near-static camera (tripod or wall-mounted phone).

SETUP: this needs the pose_landmarker model file, which isn't bundled
with the pip package. Download once:

    curl -o pose_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

(There's also a `_full` and `_heavy` variant — lite is fine to start,
swap in `_full` later if you want more accuracy at the cost of speed.)

Output: a list of per-frame dicts, one per frame, each mapping landmark name
to (x, y, z, visibility) in normalized image coordinates (0-1).
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# 33-point MediaPipe pose landmark order (index position = enum value)
POSE_LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]
LANDMARK_INDEX = {name: i for i, name in enumerate(POSE_LANDMARK_NAMES)}

# The subset we actually care about for climbing technique analysis.
RELEVANT_LANDMARKS = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]


@dataclass
class FrameData:
    frame_idx: int
    timestamp_s: float
    landmarks: dict  # name -> (x, y, z, visibility), None if not detected
    detected: bool = True


def extract_pose_sequence(
    video_path: str,
    model_path: str = "pose_landmarker.task",
    min_confidence: float = 0.5,
) -> list[FrameData]:
    """
    Run MediaPipe's PoseLandmarker over every frame of the video.

    Returns a list of FrameData, one per frame, in order. Frames where no
    person was detected still get an entry (detected=False) so downstream
    code can keep frame indices aligned with timestamps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

    results_out: list[FrameData] = []

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp = frame_idx / fps

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]  # first detected person
                landmarks = {}
                for name in RELEVANT_LANDMARKS:
                    point = lm[LANDMARK_INDEX[name]]
                    visibility = getattr(point, "visibility", 1.0)
                    landmarks[name] = (point.x, point.y, point.z, visibility)
                results_out.append(FrameData(frame_idx, timestamp, landmarks, detected=True))
            else:
                results_out.append(FrameData(frame_idx, timestamp, {}, detected=False))

            frame_idx += 1

    cap.release()
    return results_out


def smooth_sequence(frames: list[FrameData], window: int = 5) -> list[FrameData]:
    """
    Simple moving-average smoothing over each landmark's (x, y) to reduce
    frame-to-frame jitter. Skips frames with no detection when averaging.
    Confidence-gating: landmarks with visibility below 0.3 are excluded
    from the average for that frame (helps with the motion-blur-near-falls
    problem).
    """
    half = window // 2
    smoothed = []

    for i, f in enumerate(frames):
        if not f.detected:
            smoothed.append(f)
            continue

        lo, hi = max(0, i - half), min(len(frames), i + half + 1)
        window_frames = [w for w in frames[lo:hi] if w.detected]

        new_landmarks = {}
        for name in RELEVANT_LANDMARKS:
            xs, ys, zs, vs = [], [], [], []
            for w in window_frames:
                if name in w.landmarks and w.landmarks[name][3] >= 0.3:
                    x, y, z, v = w.landmarks[name]
                    xs.append(x); ys.append(y); zs.append(z); vs.append(v)
            if xs:
                new_landmarks[name] = (float(np.mean(xs)), float(np.mean(ys)),
                                        float(np.mean(zs)), float(np.mean(vs)))
            elif name in f.landmarks:
                new_landmarks[name] = f.landmarks[name]

        smoothed.append(FrameData(f.frame_idx, f.timestamp_s, new_landmarks, detected=True))

    return smoothed
