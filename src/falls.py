"""
falls.py

Finds the moments the climber leaves the wall and drops to the mat,
decides whether each was a fall or a deliberate jump-off, and ties each
one back to the instability that came before it.

A "drop" is any airborne descent. Fall vs. dismount is a judgement on top
of that — see classify_drop(). The thresholds were set against four clips
(two falls, one jump-off after a send, one send that left the frame), so
treat the label as a first guess until more footage has been reviewed.
"""

from dataclasses import dataclass, field

import numpy as np

from src.kinematics import Kinematics, runs_of
from src.stability import StabilityEvent, StabilityTrack

# --- drop detection (body-scales, body-scales/sec) --------------------------
DESCENT_SPEED = 1.0         # COM moving down at least this fast counts as "descending"
MIN_PEAK_SPEED = 4.0        # ...and must peak at least this fast (sitting down peaks ~1-2)
MIN_DROP_HEIGHT = 1.0       # ...and cover at least this much height
MAX_DESCENT_GAP_S = 0.15

# --- fall vs. dismount ------------------------------------------------------
PRE_RELEASE_S = 0.5         # window before release examined for "was mid-move"
MOVING_LATERAL_SPEED = 1.5  # mean sideways COM speed before release above this = mid-move
ROTATION_DEG = 60.0         # torso tilt range during the drop above this = tumbling
LATERAL_DRIFT = 1.0         # sideways travel during the drop above this = thrown off
POST_LANDING_S = 0.2        # rotation is measured through the landing

# --- instability -> fall ----------------------------------------------------
UNSTABLE_BEFORE_S = 3.0     # an instability event this close to release counts as a precursor
LOOKBACK_S = 5.0            # events this far back are listed in the lead-up


@dataclass
class Drop:
    release_frame: int
    landing_frame: int
    release_s: float
    duration_s: float
    height: float                     # body-scales descended
    peak_speed: float                 # body-scales/sec
    lateral_drift: float              # |sideways COM travel| during the drop
    torso_rotation_deg: float         # range of torso tilt during the drop
    pre_release_lateral_speed: float  # mean sideways COM speed just before release
    kind: str = "drop"                # "fall" | "dismount"
    reasons: list[str] = field(default_factory=list)
    lead_up: dict = field(default_factory=dict)


def _torso_tilt_deg(k: Kinematics) -> np.ndarray:
    """Angle of the hip->shoulder line from vertical (0 = upright)."""
    d = k.shoulder_mid - k.hip_mid
    return np.degrees(np.arctan2(d[:, 0], -d[:, 1]))


def _merge(runs: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def detect_drops(k: Kinematics) -> list[Drop]:
    vel = k.velocity(k.com)
    vy = np.where(k.present, vel[:, 1], np.nan)
    tilt = _torso_tilt_deg(k)

    descending = np.nan_to_num(vy, nan=0.0) > DESCENT_SPEED
    drops = []
    for start, end in _merge(runs_of(descending), int(MAX_DESCENT_GAP_S * k.fps)):
        seg = slice(start, end + 1)
        scale = np.nanmedian(k.scale[seg])
        height = (np.nanmax(k.com[seg, 1]) - np.nanmin(k.com[seg, 1])) / scale
        peak = float(np.nanmax(vy[seg]))
        if peak < MIN_PEAK_SPEED or height < MIN_DROP_HEIGHT:
            continue

        pre = np.abs(vel[max(0, start - int(PRE_RELEASE_S * k.fps)):start, 0])
        through = tilt[start:end + 1 + int(POST_LANDING_S * k.fps)]
        through = through[~np.isnan(through)]
        drops.append(Drop(
            release_frame=start,
            landing_frame=end,
            release_s=round(start / k.fps, 2),
            duration_s=round((end - start + 1) / k.fps, 2),
            height=round(float(height), 2),
            peak_speed=round(peak, 2),
            lateral_drift=round(float(abs(k.com[end, 0] - k.com[start, 0]) / scale), 2),
            torso_rotation_deg=round(float(np.ptp(through)), 1) if len(through) else 0.0,
            pre_release_lateral_speed=round(float(np.nanmean(pre)), 2) if np.any(~np.isnan(pre)) else 0.0,
        ))
    return drops


def describe_lead_up(drop: Drop, k: Kinematics, st: StabilityTrack,
                     events: list[StabilityEvent]) -> None:
    """Fills drop.lead_up: what the stability signals were doing before release."""
    fps, rel = k.fps, drop.release_frame

    def mean_score(seconds):
        seg = st.score[max(0, rel - int(seconds * fps)):rel]
        return round(float(np.nanmean(seg)), 2) if np.any(~np.isnan(seg)) else None

    # baseline = the same attempt, excluding the final stretch being compared against it
    attempt_start = rel
    while attempt_start > 0 and not k.on_wall[attempt_start - 1] and rel - attempt_start < fps:
        attempt_start -= 1
    while attempt_start > 0 and k.on_wall[attempt_start - 1]:
        attempt_start -= 1
    base = st.score[attempt_start:max(attempt_start, rel - int(UNSTABLE_BEFORE_S * fps))]
    baseline = round(float(np.nanmean(base)), 2) if np.any(~np.isnan(base)) else None

    recent = []
    for e in events:
        gap_s = (rel - e.end_frame) / fps
        if e.end_frame < rel and gap_s <= LOOKBACK_S:
            recent.append({"kind": e.kind, "start_s": e.start_s, "duration_s": e.duration_s,
                           "ended_before_release_s": round(gap_s, 2), "detail": e.detail})

    unstable = [e for e in recent if e["kind"] != "stall" and e["ended_before_release_s"] <= UNSTABLE_BEFORE_S]
    window = st.score[max(0, rel - int(LOOKBACK_S * fps)):rel]
    drop.lead_up = {
        "unstable_before_release": bool(unstable),
        "last_instability_ended_before_release_s":
            min((e["ended_before_release_s"] for e in unstable), default=None),
        "unstable_s_in_last_5s": round(float(np.nansum(window >= 0.5)) / fps, 2),
        "mean_instability_last_1s": mean_score(1.0),
        "mean_instability_last_3s": mean_score(3.0),
        "mean_instability_rest_of_attempt": baseline,
        "held_position_before_release_s":
            round(float(np.max(st.stalled_s[max(0, rel - int(UNSTABLE_BEFORE_S * fps)):rel], initial=0.0)), 1),
        "events": recent,
    }


def classify_drop(drop: Drop) -> None:
    """
    Rule of thumb, not ground truth: a deliberate dismount starts from a
    settled position and comes straight down upright; a fall starts from an
    unstable position or mid-move, and/or the body tumbles or is thrown
    sideways on the way down. Any one sign of lost control => "fall".
    Call describe_lead_up() first.
    """
    reasons = []
    if drop.lead_up.get("unstable_before_release"):
        reasons.append(f"unstable until {drop.lead_up['last_instability_ended_before_release_s']}s before release")
    if drop.pre_release_lateral_speed > MOVING_LATERAL_SPEED:
        reasons.append(f"moving sideways at {drop.pre_release_lateral_speed} body-lengths/s at release (mid-move)")
    if drop.torso_rotation_deg > ROTATION_DEG:
        reasons.append(f"torso swung through {drop.torso_rotation_deg:.0f} degrees on the way down")
    if drop.lateral_drift > LATERAL_DRIFT:
        reasons.append(f"travelled {drop.lateral_drift} body-lengths sideways in the air")

    drop.kind = "fall" if reasons else "dismount"
    drop.reasons = reasons or ["settled before release and came straight down upright"]
