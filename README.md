# climb-vision

A climbing technique analyzer: video in -> annotated video + JSON report out.

For each clip it works out:

- **when you are actually on the wall** — frames with no climber, a
  half-framed climber, or you walking to and from the wall are ignored
- **how unstable you are** — a 0-1 instability score per frame, built from
  weight off the feet, both feet cutting loose, and lurches of the COM,
  grouped into timestamped events (plus long holds in one position)
- **when you come off** — every drop to the mat, labelled as a *fall* or a
  deliberate *dismount*, with the reasons
- **how the two relate** — for each drop, what the instability looked like
  in the seconds before release, compared with the rest of the attempt

v1 (tagged `v1`) had a single heuristic, COM-over-base-of-support. Reviewing
it on real footage (`analysis/flag_review.csv`) showed 36 of 50 flags fired
on frames with no climber on the wall, which is what the on-wall gating fixes.

## Setup

```bash
pip install -r requirements.txt

# Download the pose model (not bundled with the pip package)
curl -o pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

## Usage

```bash
python main.py path/to/climb.mp4
```

Outputs `outputs/<clip>/annotated.mp4` (skeleton, status strip, event and
fall banners) and `outputs/<clip>/report.json` (`summary`, `drops`,
`instability_events`, and the v1 `flags`). Add `--no-video` for the report only.

```bash
python scripts/summarize_stability.py    # one row per clip -> analysis/stability_summary.csv
python scripts/review_flags.py "videos/*.mp4"   # contact sheets for each COM-over-base flag
```

Name clips `..._fall.mp4` / `..._send.mp4` and the summary also checks the
detected outcome against the real one.

**Filming tip:** static camera (tripod or clip mount), far enough back to
keep your whole body in frame for the whole climb, including the top and
the landing. Start and stop recording away from the lens if you can.
Handheld support (motion compensation) is a planned v2.

**Camera angle matters.** From behind, "weight off the feet" means your
hips are sideways of your feet. From side-on, the same measurement means
your hips are away from the wall. The numbers are comparable within one
angle, not across them.

**What the fall / dismount label is based on.** A drop is called a fall if
you were unstable just before letting go, were moving sideways at release,
tumbled, or were thrown sideways in the air; otherwise it is a dismount.
These rules were set on four clips, so treat the label as a first guess.

## Project structure

```
climb-vision/
├── main.py                  # CLI entry point
├── src/
│   ├── pose_extraction.py   # MediaPipe wrapper: video -> per-frame keypoints
│   ├── kinematics.py        # keypoints -> COM/limb tracks; is a climber on the wall?
│   ├── stability.py         # per-frame instability score + events
│   ├── falls.py             # drops, fall vs. dismount, lead-up to each drop
│   ├── features.py          # v1 COM-over-base heuristic
│   ├── analysis.py          # runs the above in order, builds the report
│   └── overlay.py           # draws skeleton + status + banners onto the video
├── scripts/
│   ├── review_flags.py      # contact sheets + diagnostics for each flag
│   └── summarize_stability.py   # per-clip comparison table
├── analysis/                # review logs and summaries from real footage
└── requirements.txt
```

## Roadmap

**v1** — static camera, rule-based COM-over-base flagging, skeleton
overlay + JSON report.

**v1.1 (this branch)** — on-wall gating, instability score and events,
fall/dismount detection, instability-before-fall lead-up.

**v1.x** — more heuristics:
- hip rotation angle (flagging / twist-lock detection)
- dead-point / contact-time analysis (how long a hand hovers before committing)
- use the detected drops to auto-segment a longer session into attempts

**v2** — handheld camera support via optical-flow-based motion compensation,
so metrics stay valid without a tripod.

**v3** — ML layer once there's enough footage:
- small classifier (MLP / 1D-CNN over joint-angle time series) trained on
  your own labeled clips to replace/augment the hand-written heuristics
- movement embeddings for similarity search across your climbing history
  ("find the move most similar to this one")

## Notes

- All feature thresholds are normalized by body scale (shoulder-to-hip
  distance) rather than raw pixels, so they hold up across different
  distances from the camera.
- The COM estimate (60% hip midpoint / 40% shoulder midpoint) is a
  reasonable approximation, not a full segmental biomechanical model —
  good enough to get useful signal, worth revisiting if flags feel
  systematically off in either direction once you test on real footage.
- Reference: Jerry Qu (Stanford CS231n), "Using Pose Estimation to Analyze
  Rock Climbing Technique" — used ViTPose + YOLOv8, worth a skim for what
  metrics correlated with skill level in their data.
