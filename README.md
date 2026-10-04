# climb-vision

v1 of a climbing technique analyzer: video in -> annotated video + JSON
report of technique flags out.

Currently implements one heuristic: **center-of-mass-over-base-of-support
violation** — flags stretches where your COM drifts outside your foot
placements (the classic "reaching/leaning away from the wall" fault, and
a strong precursor to falls).

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

Outputs `annotated_output.mp4` (skeleton overlay + on-screen flags) and
`technique_report.json` (structured flag list with timestamps).

**Filming tip:** static camera (tripod or clip mount), side-on, hip-height,
far enough back to catch your full body through the moves. v1 assumes a
static camera — handheld support (motion compensation) is a planned v2.

## Project structure

```
climb-vision/
├── main.py                  # CLI entry point
├── src/
│   ├── pose_extraction.py   # MediaPipe wrapper: video -> per-frame keypoints
│   ├── features.py          # keypoints -> climbing-specific technique metrics
│   └── overlay.py           # draws skeleton + flags back onto the video
└── requirements.txt
```

## Roadmap

**v1 (this)** — static camera, rule-based COM-over-base flagging, skeleton
overlay + JSON report.

**v1.x** — more heuristics:
- hip rotation angle (flagging / twist-lock detection)
- dead-point / contact-time analysis (how long a hand hovers before committing)
- fall/dismount detection (sudden vertical hip velocity spike) to
  auto-segment attempts from a longer recording session

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

## License

MIT. See [LICENSE](LICENSE).
