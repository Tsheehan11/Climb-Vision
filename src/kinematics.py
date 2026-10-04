"""
kinematics.py

Turns the per-frame landmark dicts into aligned numpy tracks (COM, body
scale, feet, wrists, velocities) and decides, frame by frame, whether we
are actually looking at a climber on the wall.

That second part exists because of the v1 real-footage review
(analysis/flag_review.csv): 36 of 50 flags fired on frames with no climber
on the wall at all — phantom skeletons on holds, the climber half out of
frame, walking to/from the wall, or a hand over the lens. Every heuristic
downstream should only look at frames where `on_wall` is True.

All distances are in body-scale units (shoulder-to-hip length) and all
speeds in body-scales per second, so thresholds hold across camera
distances and frame rates.
"""

from dataclasses import dataclass

import numpy as np

from src.pose_extraction import FrameData

TORSO = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP")
ANKLES = ("LEFT_ANKLE", "RIGHT_ANKLE")
WRISTS = ("LEFT_WRIST", "RIGHT_WRIST")
FEET = ("LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX")

# --- presence gating -------------------------------------------------------
TRACK_WINDOW_S = 0.5        # pose must be detected almost continuously this far either side
TRACK_MIN_FRAC = 0.9        # ...phantom skeletons on holds flicker in and out
MIN_TORSO_VISIBILITY = 0.5
FRAME_EDGE = 0.01           # torso must be this far inside the left/right edges
FOOT_MAX_Y = 1.1            # feet may sit slightly below the frame, but no further
SCALE_BAND = (0.65, 1.4)    # vs. the clip's median body scale; rejects walking up to the lens

# --- on-wall state ---------------------------------------------------------
FLOOR_PERCENTILE = 97       # lowest-foot height the climber reaches = the mat
MIN_FOOT_CLEARANCE = 0.4    # lowest foot this far above the mat (body-scales)
MIN_COM_CLEARANCE = 2.2     # COM higher than it would be standing on the mat
REENTRY_FOOT_CLEARANCE = 1.0  # after a drop, feet must clearly leave the mat again
MIN_STATE_RUN_S = 0.5
MIN_WALL_RUN_S = 1.0        # shorter "on wall" blips are sit-start shuffles, not climbing
MAX_STATE_GAP_S = 0.3


@dataclass
class Kinematics:
    fps: float
    n: int
    com: np.ndarray          # (n, 2) normalized image coords, NaN where undetected
    scale: np.ndarray        # (n,) smoothed body scale
    feet_x: np.ndarray       # (n, 2) min / max foot x
    lowest_foot_y: np.ndarray
    ankles: np.ndarray       # (n, 2, 2) left/right ankle xy
    wrists: np.ndarray       # (n, 2, 2)
    shoulder_mid: np.ndarray  # (n, 2)
    hip_mid: np.ndarray       # (n, 2)
    foot_visibility: np.ndarray
    present: np.ndarray      # bool: a real, fully framed climber is being tracked
    on_wall: np.ndarray      # bool: present and off the mat (filled in by label_on_wall)
    floor_y: float

    def velocity(self, track: np.ndarray, half_window: int = 2) -> np.ndarray:
        """Central-difference velocity of an (n, 2) track in body-scales/sec."""
        v = np.full_like(track, np.nan)
        h = half_window
        v[h:-h] = (track[2 * h:] - track[:-2 * h]) / (2 * h / self.fps)
        return v / self.scale[:, None]


def _rolling_nanmedian(x: np.ndarray, half: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        w = x[max(0, i - half):i + half + 1]
        if np.any(~np.isnan(w)):
            out[i] = np.nanmedian(w)
    return out


def clean_runs(mask: np.ndarray, min_run: int, max_gap: int) -> np.ndarray:
    """Fill False gaps shorter than max_gap, then drop True runs shorter than min_run."""
    m = mask.copy()
    for value, limit in ((False, max_gap), (True, min_run)):
        i = 0
        while i < len(m):
            if m[i] == value:
                j = i
                while j < len(m) and m[j] == value:
                    j += 1
                interior = i > 0 and j < len(m)
                if j - i < limit and (value or interior):
                    m[i:j] = not value
                i = j
            else:
                i += 1
    return m


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of each True run."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def build_kinematics(raw: list[FrameData], smoothed: list[FrameData], fps: float) -> Kinematics:
    """
    `raw` supplies visibility and detection (smoothing blurs both);
    `smoothed` supplies positions.
    """
    n = len(smoothed)
    nan2 = lambda: np.full((n, 2), np.nan)
    com, sh, hip, feet_x = nan2(), nan2(), nan2(), nan2()
    ankles, wrists = np.full((n, 2, 2), np.nan), np.full((n, 2, 2), np.nan)
    scale, lowest_foot = np.full(n, np.nan), np.full(n, np.nan)
    torso_vis, foot_vis = np.zeros(n), np.zeros(n)
    detected = np.array([f.detected for f in raw])
    framed = np.zeros(n, dtype=bool)

    for i, (r, s) in enumerate(zip(raw, smoothed)):
        lm = s.landmarks
        if not s.detected or any(k not in lm for k in TORSO + FEET + WRISTS):
            continue
        pts = {k: np.array(lm[k][:2]) for k in lm}
        sh[i] = (pts["LEFT_SHOULDER"] + pts["RIGHT_SHOULDER"]) / 2
        hip[i] = (pts["LEFT_HIP"] + pts["RIGHT_HIP"]) / 2
        com[i] = 0.6 * hip[i] + 0.4 * sh[i]
        scale[i] = np.linalg.norm(sh[i] - hip[i])
        fx = [pts[k][0] for k in FEET]
        feet_x[i] = (min(fx), max(fx))
        lowest_foot[i] = max(pts[k][1] for k in FEET)
        ankles[i] = [pts[k] for k in ANKLES]
        wrists[i] = [pts[k] for k in WRISTS]
        torso_vis[i] = min(r.landmarks[k][3] for k in TORSO) if r.detected else 0.0
        foot_vis[i] = min(r.landmarks[k][3] for k in ANKLES) if r.detected else 0.0
        torso_x = [pts[k][0] for k in TORSO]
        framed[i] = (min(torso_x) > FRAME_EDGE and max(torso_x) < 1 - FRAME_EDGE
                     and 0 <= min(fx) and max(fx) <= 1 and lowest_foot[i] <= FOOT_MAX_Y
                     and 0 < sh[i][1] < 1)

    scale = _rolling_nanmedian(scale, int(fps * 0.5))

    half = int(round(TRACK_WINDOW_S * fps))
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    tracked = np.convolve(detected.astype(float), kernel, mode="same") >= TRACK_MIN_FRAC
    present = tracked & detected & framed & (torso_vis >= MIN_TORSO_VISIBILITY) & ~np.isnan(scale)

    if present.any():
        ref = np.nanmedian(scale[present])
        present &= (scale >= SCALE_BAND[0] * ref) & (scale <= SCALE_BAND[1] * ref)
    present = clean_runs(present, int(MIN_STATE_RUN_S * fps), 0)

    floor_y = float(np.percentile(lowest_foot[present], FLOOR_PERCENTILE)) if present.any() else 1.0

    return Kinematics(fps=fps, n=n, com=com, scale=scale, feet_x=feet_x,
                      lowest_foot_y=lowest_foot, ankles=ankles, wrists=wrists,
                      shoulder_mid=sh, hip_mid=hip, foot_visibility=foot_vis,
                      present=present, on_wall=np.zeros(n, dtype=bool), floor_y=floor_y)


def label_on_wall(k: Kinematics, drops: list | None = None) -> None:
    """
    Sets k.on_wall: present, feet off the mat, COM higher than standing height.
    `drops` (from falls.detect_drops) end an attempt: from release until the
    feet clearly leave the mat again, the climber is on the ground.
    """
    with np.errstate(invalid="ignore"):
        foot_clear = (k.floor_y - k.lowest_foot_y) / k.scale
        com_clear = (k.floor_y - k.com[:, 1]) / k.scale
        on = k.present & (foot_clear > MIN_FOOT_CLEARANCE) & (com_clear > MIN_COM_CLEARANCE)

    for d in drops or []:
        i = d.release_frame
        while i < k.n and not (k.present[i] and i > d.landing_frame
                               and foot_clear[i] > REENTRY_FOOT_CLEARANCE):
            on[i] = False
            i += 1

    k.on_wall = clean_runs(on, int(MIN_WALL_RUN_S * k.fps), int(MAX_STATE_GAP_S * k.fps))
    for d in drops or []:
        k.on_wall[d.release_frame:d.landing_frame + 1] = False
