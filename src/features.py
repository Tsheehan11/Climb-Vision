"""
features.py

Turns raw pose keypoints into climbing-specific technique metrics.
All measurements are normalized by body scale (shoulder-to-hip distance)
so they stay meaningful across different climbers, distances from camera,
and (later) handheld footage with some zoom drift.

v1 heuristic implemented: center-of-mass-over-base-of-support violation.
This is the single most common beginner fault — climbing "away from the
wall" or reaching with the COM outside the polygon formed by the limbs
in contact with holds — and it's a strong precursor to falls.
"""

from dataclasses import dataclass
from src.pose_extraction import FrameData


@dataclass
class TechniqueFlag:
    frame_idx: int
    timestamp_s: float
    flag_type: str
    severity: float  # 0-1, how far outside the threshold
    message: str


def _midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _body_scale(landmarks: dict) -> float | None:
    """Shoulder-to-hip distance in normalized coords, used to scale all
    other thresholds so they're consistent regardless of camera distance."""
    try:
        ls = landmarks["LEFT_SHOULDER"]
        rs = landmarks["RIGHT_SHOULDER"]
        lh = landmarks["LEFT_HIP"]
        rh = landmarks["RIGHT_HIP"]
    except KeyError:
        return None
    shoulder_mid = _midpoint(ls, rs)
    hip_mid = _midpoint(lh, rh)
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    return (dx ** 2 + dy ** 2) ** 0.5


def estimate_center_of_mass(landmarks: dict) -> tuple[float, float] | None:
    """
    Rough COM estimate: weighted average of hip midpoint (dominant mass)
    and shoulder midpoint. Good enough for flagging purposes — a full
    segmental COM model is a v2 upgrade, not needed to get useful signal.
    """
    try:
        lh, rh = landmarks["LEFT_HIP"], landmarks["RIGHT_HIP"]
        ls, rs = landmarks["LEFT_SHOULDER"], landmarks["RIGHT_SHOULDER"]
    except KeyError:
        return None
    hip_mid = _midpoint(lh, rh)
    shoulder_mid = _midpoint(ls, rs)
    # Hips carry roughly 60% of the COM weighting for a climber's typical
    # limb-splayed posture; this is a tunable approximation, not gospel.
    com_x = 0.6 * hip_mid[0] + 0.4 * shoulder_mid[0]
    com_y = 0.6 * hip_mid[1] + 0.4 * shoulder_mid[1]
    return (com_x, com_y)


def base_of_support_x_range(landmarks: dict) -> tuple[float, float] | None:
    """
    Horizontal span of whichever limbs are the likely points of contact.
    v1 simplification: use ankle/foot x-positions as the base (assumes
    feet are on holds, which is true most of the time in bouldering).
    """
    xs = []
    for name in ("LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"):
        if name in landmarks:
            xs.append(landmarks[name][0])
    if not xs:
        return None
    return (min(xs), max(xs))


def detect_com_over_base_violations(
    frames: list[FrameData],
    margin_ratio: float = 0.15,
    min_duration_frames: int = 3,
    frame_mask=None,
) -> list[TechniqueFlag]:
    """
    Flags stretches where the estimated COM x-position drifts outside the
    base-of-support x-range by more than `margin_ratio` * body_scale, for
    at least `min_duration_frames` consecutive frames (filters out
    single-frame noise rather than real weight-shift-away-from-wall events).

    `frame_mask` (one bool per frame, e.g. Kinematics.on_wall) restricts the
    check to frames where a climber is actually on the wall; without it the
    heuristic also fires on phantom poses and on the walk to and from the wall.
    """
    flags = []
    violation_run = []

    for f in frames:
        if not f.detected or (frame_mask is not None and not frame_mask[f.frame_idx]):
            _flush_run(violation_run, flags)
            violation_run = []
            continue

        com = estimate_center_of_mass(f.landmarks)
        base = base_of_support_x_range(f.landmarks)
        scale = _body_scale(f.landmarks)

        if com is None or base is None or scale is None or scale == 0:
            _flush_run(violation_run, flags)
            violation_run = []
            continue

        margin = margin_ratio * scale
        lo, hi = base[0] - margin, base[1] + margin
        com_x = com[0]

        if com_x < lo or com_x > hi:
            overshoot = min(abs(com_x - lo), abs(com_x - hi)) if com_x < lo or com_x > hi else 0
            severity = min(1.0, overshoot / (scale + 1e-6))
            violation_run.append((f, severity))
        else:
            _flush_run(violation_run, flags, min_duration_frames)
            violation_run = []

    _flush_run(violation_run, flags, min_duration_frames)
    return flags


def _flush_run(run, flags_out, min_duration=3):
    if len(run) >= min_duration:
        # Report the peak-severity frame in the run as the flag point
        worst = max(run, key=lambda r: r[1])
        f, severity = worst
        flags_out.append(TechniqueFlag(
            frame_idx=f.frame_idx,
            timestamp_s=f.timestamp_s,
            flag_type="com_over_base",
            severity=round(severity, 3),
            message="Center of mass drifted outside base of support "
                    "(reaching/leaning away from the wall)",
        ))
