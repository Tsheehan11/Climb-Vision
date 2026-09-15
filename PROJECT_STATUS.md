# climb-vision — Project Status Report

**Date:** 2026-09-15
**Stage:** v1 complete, untested on real footage

## Summary

Built the full v1 pipeline: video in → pose extraction → smoothing →
technique-fault detection → annotated video + JSON report out. Core
logic (COM-over-base heuristic) has been unit-tested against synthetic
keypoint data and behaves correctly. Not yet run against real climbing
footage.

## What's done

| Component | Status | Notes |
|---|---|---|
| Pose extraction (`pose_extraction.py`) | Built | Uses MediaPipe Tasks API (`PoseLandmarker`, VIDEO mode). Requires separate `.task` model download — not bundled in pip package. |
| Frame smoothing | Built | Moving-average window (default 5 frames), confidence-gated at 0.3 visibility. |
| COM-over-base heuristic (`features.py`) | Built + tested | Verified against synthetic "reaching away from wall" scenario — flagged the correct window, no false positives on balanced frames. |
| Skeleton overlay + flag banner (`overlay.py`) | Built | Not visually verified against real footage yet. |
| CLI (`main.py`) | Built | End-to-end run smoke-tested (synthetic clip, no real pose data — confirms no crashes / wiring errors). |
| Repo setup | Done | Pushed to GitHub (`Tsheehan11/Climb-Vision`), `.gitignore` excludes model file and generated video/report artifacts. |

## Known gaps / risks

- **No real-footage validation yet.** All testing so far is synthetic
  (fabricated keypoints, blank test video). The COM heuristic's
  thresholds (`margin_ratio=0.15`, `min_duration_frames=3`) are
  reasonable starting guesses, not tuned against actual climbing.
- **MediaPipe API surface changed recently** — the older `mp.solutions.pose`
  API used in most tutorials/examples online is deprecated in the
  installed version (0.10.33); this project uses the current Tasks API.
  Worth double-checking this stays current if revisiting after a gap.
- **Single-person assumption** — pipeline takes the first detected person
  per frame; not an issue solo-bouldering with a clean background, but
  could misfire if other people are visible in frame.
- **COM model is an approximation** (60% hip / 40% shoulder midpoint),
  not a full biomechanical model. Fine for v1 signal; may need revisiting
  if flags feel systematically biased once tested on real clips.

## Immediate next steps

See `V1.X_TASKS.md` for the full checklist. Priority order:
1. Film real footage on a tripod and run it through the pipeline
2. Visually verify skeleton tracking quality before trusting any flags
3. Log flag accuracy (correct / false-positive / missed) per clip
4. Tune heuristic thresholds based on that log
