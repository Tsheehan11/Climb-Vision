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

## Progress

| Stage | Status | Where |
|---|---|---|
| v1: pose extraction, one heuristic, overlay, report | Done, tagged `v1` | `main` |
| v1 tested on real footage | Done: 4 clips, all 50 flags reviewed | `analysis/flag_review.csv` |
| v1.1: on-wall gating, instability, falls | Built, checked on the same 4 clips | branch `v1.1-stability-and-falls` |
| v1.1 tested on new footage | Not started | — |

**What testing v1 showed.** v1 had a single heuristic: flag any stretch
where the centre of mass (COM) sits sideways of both feet. On four real
clips (two falls, two sends, filmed 2026-09-15) it raised 50 flags:

| Verdict | Flags | What it was |
|---|---|---|
| False positive | 36 | No climber on the wall: climber cut off by the frame edge (11), skeleton drawn on holds with nobody there (10), walking to or from the wall (8), hand over the lens (6), bad leg tracking (1) |
| True but not a fault | 6 | Real position, but normal for the move (sit start, high step, mid foot-swap) |
| Plausibly correct | 6 | Hips held well to one side of the feet for a second or more |
| Other | 2 | One was the fall itself, one had unreliable leg tracking |

So v1's geometry was fine; its problem was not knowing when to look. It
also could not tell a wobble from a fall.

**What v1.1 changed, measured on the same four clips.**

- The 50 flags drop to 13. None of the 36 false positives remain and all 6
  plausibly-correct ones are kept.
- The outcome matches the filename label on 4 of 4 clips: both falls are
  called falls, the jump-off after a send is called a dismount, and the
  send that climbs out of frame reports no drop.

| Clip | Detected | On wall | Unstable | Lead-up to the drop |
|---|---|---|---|---|
| route1 (fall) | fall at 25.5s | 15.0s | 3.8% | Stable beforehand; came off mid-move, travelling sideways |
| route2 (send) | no drop seen | 4.8s | 4.9% | Climbs out of the top of the frame |
| route3 (fall) | fall at 39.8s | 25.0s | 57.5% | Unstable most of the climb, then a 10s hold, then off |
| route4 (send) | dismount at 18.1s | 10.2s | 2.3% | Settled at the top, dropped straight down |

**What is not proven yet.** Four clips built the rules and the same four
checked them, so the numbers above are a best case. The fall-vs-dismount
rules rest on three drops. Nothing has been checked for *missed* events:
the review only judged flags that fired.

## How it works

Each step feeds the next. All distances are measured in "body-lengths"
(shoulder-to-hip distance), so results do not depend on how far away the
camera is.

1. **Find the body** (`pose_extraction.py`). MediaPipe looks at every frame
   and returns 14 joint positions (shoulders, elbows, wrists, hips, knees,
   ankles, toes), each with a confidence. A 5-frame average removes jitter.

2. **Decide whether to trust the frame** (`kinematics.py`). A frame counts as
   "climber present" only if the pose is detected almost continuously for
   half a second either side (phantom skeletons on holds flicker), the torso
   is inside the frame, and the body is a normal size for the clip (rules
   out walking up to the lens). The COM is estimated as 60% hips, 40%
   shoulders.

3. **Decide whether you are on the wall** (`kinematics.py`). The lowest point
   your feet reach in the clip is taken as the mat. You are "on the wall"
   when your feet are clear of the mat and your COM is higher than it would
   be standing. Everything after this step only looks at on-wall frames.

4. **Find drops** (`falls.py`). A drop is the COM moving down fast (peaking
   above 4 body-lengths/s) for at least one body-length. Sitting down or
   lowering onto a hold is too slow to count.

5. **Score instability** (`stability.py`). Each on-wall frame gets a score
   from 0 to 1, the highest of three signals:
   - *weight off the feet*: how far the COM is sideways of both feet
     (1.0 = a full body-length)
   - *feet cut loose*: both feet moving fast at once (one foot moving is
     just a step)
   - *lurch*: sudden acceleration of the COM

   A score of 0.5 or more held for 0.3s becomes an **instability event**.
   Separately, staying within about a third of a body-length for 3s or more
   is logged as a **stall**.

6. **Label each drop and explain it** (`falls.py`). A drop is a **fall** if
   any one of these is true, otherwise a **dismount**:
   - an instability event ended within 3s of letting go
   - you were moving sideways at release (mid-move)
   - your torso swung through more than 60 degrees on the way down
   - you travelled more than a body-length sideways in the air

   Each drop also records the lead-up: average instability in the last 1s
   and 3s against the rest of the attempt, and how long you had been holding
   one position.

7. **Write the outputs** (`analysis.py`, `overlay.py`). `report.json` and the
   annotated video.

**Reading `report.json`:**

- `summary`: outcome, seconds on the wall, seconds and percent unstable,
  longest hold, event counts
- `drops`: one entry per drop, with `kind` (fall or dismount), `reasons`,
  and `lead_up`
- `instability_events`: each event's kind, start time, duration and detail
- `flags`: the original v1 COM flags, on-wall frames only

**Reading the annotated video:** the strip along the bottom shows the state
(NO CLIMBER, ON GROUND, ON WALL, AIRBORNE) and the instability bar (green
below 0.35, amber to 0.5, red above). Banners at the top name the current
instability event, stall, or drop. A grey skeleton means the pose model
fired on something the analysis ignored.

**To change how sensitive it is,** every threshold is a named constant at
the top of `kinematics.py`, `stability.py` and `falls.py`.

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
