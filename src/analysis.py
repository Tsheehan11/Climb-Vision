"""
analysis.py

Runs the whole post-pose pipeline in order and packages the result:

    keypoints -> kinematics + on-wall gating -> drops -> stability score
              -> instability events -> fall/dismount + lead-up -> summary
"""

from dataclasses import asdict, dataclass

import numpy as np

from src.falls import Drop, classify_drop, describe_lead_up, detect_drops
from src.features import TechniqueFlag, detect_com_over_base_violations
from src.kinematics import Kinematics, build_kinematics, label_on_wall, runs_of
from src.pose_extraction import FrameData, smooth_sequence
from src.stability import (EVENT_SCORE, StabilityEvent, StabilityTrack,
                           compute_stability, detect_events)


@dataclass
class Analysis:
    frames: list[FrameData]          # smoothed keypoints
    kin: Kinematics
    stability: StabilityTrack
    events: list[StabilityEvent]
    drops: list[Drop]
    flags: list[TechniqueFlag]       # v1 COM-over-base flags, on-wall frames only
    summary: dict

    def to_report(self) -> dict:
        return {
            "summary": self.summary,
            "drops": [asdict(d) for d in self.drops],
            "instability_events": [asdict(e) for e in self.events],
            "flags": [{"frame": f.frame_idx, "timestamp_s": round(f.timestamp_s, 2),
                       "type": f.flag_type, "severity": f.severity, "message": f.message}
                      for f in self.flags],
        }


def analyze(raw: list[FrameData], fps: float) -> Analysis:
    frames = smooth_sequence(raw)
    kin = build_kinematics(raw, frames, fps)
    drops = detect_drops(kin)
    label_on_wall(kin, drops)

    stability = compute_stability(kin)
    events = detect_events(kin, stability)
    for d in drops:
        describe_lead_up(d, kin, stability, events)
        classify_drop(d)

    flags = detect_com_over_base_violations(frames, frame_mask=kin.on_wall)
    return Analysis(frames, kin, stability, events, drops, flags,
                    _summarize(kin, stability, events, drops))


def _summarize(kin: Kinematics, st: StabilityTrack, events, drops) -> dict:
    fps = kin.fps
    on_wall_s = float(kin.on_wall.sum()) / fps
    unstable_s = float(np.nansum(st.score >= EVENT_SCORE)) / fps
    falls = [d for d in drops if d.kind == "fall"]
    if falls:
        outcome = "fall"
    elif drops:
        outcome = "dismount"
    elif on_wall_s > 0:
        outcome = "no_drop_seen"  # still on the wall, or climbed out of frame
    else:
        outcome = "no_climbing_found"

    kinds = {}
    for e in events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    return {
        "outcome": outcome,
        "fps": round(fps, 2),
        "frames_with_climber": int(kin.present.sum()),
        "on_wall_s": round(on_wall_s, 2),
        "on_wall_spans_s": [[round(a / fps, 2), round(b / fps, 2)] for a, b in runs_of(kin.on_wall)],
        "unstable_s": round(unstable_s, 2),
        "unstable_pct_of_on_wall": round(100 * unstable_s / on_wall_s, 1) if on_wall_s else 0.0,
        "mean_instability": round(float(np.nanmean(st.score)), 2) if on_wall_s else None,
        "longest_hold_s": round(float(st.stalled_s.max()), 1) if on_wall_s else 0.0,
        "event_counts": kinds,
        "falls": len(falls),
        "dismounts": len(drops) - len(falls),
    }
