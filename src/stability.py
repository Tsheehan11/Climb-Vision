"""
stability.py

Scores how unstable the climber is on every on-wall frame, and groups the
unstable stretches into events.

Three signals feed the score, each normalized so 1.0 means "clearly
unstable" (all in body-scale units, so they hold across camera distances):

    off_base   COM sideways of both feet. From a camera behind the climber
               this is lateral offset, not distance from the wall: the feet
               are not under the weight, so the arms are holding it.
    feet_cut   both feet moving fast at once — the feet have come off.
               (One foot moving is just a step.)
    jolt       COM acceleration — lurches, catches and swings, as opposed
               to smooth weight shifts.

A fourth, `stalled_s`, tracks how long the COM has stayed put. It is not
instability, but long stalls are where strength runs out, so it is
reported alongside as its own event kind.
"""

from dataclasses import dataclass

import numpy as np

from src.kinematics import Kinematics, clean_runs, runs_of

# value at which each signal alone scores 1.0
OFF_BASE_FULL = 1.0       # COM a full torso-length sideways of the feet
FEET_CUT_FULL = 5.0       # slower of the two ankles moving at 5 body/s
JOLT_FULL = 25.0          # COM acceleration, body/s^2

EVENT_SCORE = 0.5
EVENT_MIN_S = 0.3
EVENT_MAX_GAP_S = 0.2

STALL_RADIUS = 0.35       # COM staying within this radius counts as not moving
STALL_MIN_S = 3.0


@dataclass
class StabilityTrack:
    score: np.ndarray       # (n,) 0-1, NaN off the wall
    off_base: np.ndarray    # raw signals, same units as the *_FULL constants
    feet_cut: np.ndarray
    jolt: np.ndarray
    stalled_s: np.ndarray


@dataclass
class StabilityEvent:
    kind: str               # "off_base" | "feet_cut" | "jolt" | "stall"
    start_frame: int
    end_frame: int
    peak_frame: int
    start_s: float
    duration_s: float
    peak_score: float       # for stalls: 0
    detail: str


def compute_stability(k: Kinematics) -> StabilityTrack:
    n = k.n
    wall = k.on_wall

    with np.errstate(invalid="ignore"):
        left_of = k.feet_x[:, 0] - k.com[:, 0]
        right_of = k.com[:, 0] - k.feet_x[:, 1]
        off_base = np.maximum(0, np.maximum(left_of, right_of)) / k.scale

    ankle_speed = np.stack([np.linalg.norm(k.velocity(k.ankles[:, j]), axis=1) for j in (0, 1)])
    feet_cut = np.min(ankle_speed, axis=0)

    vel = k.velocity(k.com, half_window=3)
    acc = np.full_like(vel, np.nan)
    h = 3
    acc[h:-h] = (vel[2 * h:] - vel[:-2 * h]) / (2 * h / k.fps)
    jolt = np.linalg.norm(acc, axis=1)

    parts = np.stack([off_base / OFF_BASE_FULL, feet_cut / FEET_CUT_FULL, jolt / JOLT_FULL])
    score = np.clip(np.nanmax(np.nan_to_num(parts, nan=0.0), axis=0), 0, 1)

    stalled = np.zeros(n)
    anchor = None
    for i in range(n):
        if not wall[i]:
            anchor = None
            continue
        if anchor is None or np.linalg.norm(k.com[i] - k.com[anchor]) / k.scale[i] > STALL_RADIUS:
            anchor = i
        stalled[i] = (i - anchor) / k.fps

    for arr in (score, off_base, feet_cut, jolt):
        arr[~wall] = np.nan
    return StabilityTrack(score=score, off_base=off_base, feet_cut=feet_cut,
                          jolt=jolt, stalled_s=stalled)


def detect_events(k: Kinematics, st: StabilityTrack) -> list[StabilityEvent]:
    events = []
    unstable = clean_runs(np.nan_to_num(st.score, nan=0.0) >= EVENT_SCORE,
                          int(EVENT_MIN_S * k.fps), int(EVENT_MAX_GAP_S * k.fps)) & k.on_wall
    signals = {"off_base": st.off_base / OFF_BASE_FULL,
               "feet_cut": st.feet_cut / FEET_CUT_FULL,
               "jolt": st.jolt / JOLT_FULL}
    units = {"off_base": ("COM {:.1f} body-lengths sideways of the feet", st.off_base),
             "feet_cut": ("both feet moving at {:.1f} body-lengths/s", st.feet_cut),
             "jolt": ("COM lurched at {:.0f} body-lengths/s^2", st.jolt)}

    for start, end in runs_of(unstable):
        seg = slice(start, end + 1)
        peak = start + int(np.nanargmax(st.score[seg]))
        # label by whichever signal was above the bar for most of the event
        kind = max(signals, key=lambda name: np.nansum(signals[name][seg] >= EVENT_SCORE))
        text, raw = units[kind]
        events.append(StabilityEvent(
            kind=kind, start_frame=start, end_frame=end, peak_frame=peak,
            start_s=round(start / k.fps, 2), duration_s=round((end - start + 1) / k.fps, 2),
            peak_score=round(float(np.nanmax(st.score[seg])), 2),
            detail=text.format(float(np.nanmax(raw[seg])))))

    for start, end in runs_of(st.stalled_s >= STALL_MIN_S):
        dur = float(st.stalled_s[end])
        begin = end - int(dur * k.fps)
        events.append(StabilityEvent(
            kind="stall", start_frame=begin, end_frame=end, peak_frame=end,
            start_s=round(begin / k.fps, 2), duration_s=round(dur, 2), peak_score=0.0,
            detail=f"held one position for {dur:.1f}s"))

    return sorted(events, key=lambda e: e.start_frame)
