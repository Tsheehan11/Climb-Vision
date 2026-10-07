# climb-vision — Project Status Report

**Date:** 2026-10-07
**Stage:** v1 tested on real footage; v1.1 built and checked on the same four clips, not yet on new footage

## Summary

v1 (tagged `v1` on `main`) was run on four real clips filmed 2026-09-15 and
every one of its 50 flags was reviewed. 36 were false positives, almost all
because the heuristic ran on frames with no climber on the wall.

v1.1 (branch `v1.1-stability-and-falls`) fixes that and adds an instability
score, fall detection, and a description of the instability leading up to
each fall. On the same four clips it removes all 36 false positives and
gets the outcome right on 4 of 4. It has not been run on footage it was not
built against.

## What's done

| Component | Status | Notes |
|---|---|---|
| Pose extraction (`pose_extraction.py`) | Built, used on real footage | MediaPipe Tasks API (`PoseLandmarker`, VIDEO mode), lite model. Draws phantom skeletons on holds when nobody is in frame; handled downstream. |
| Presence + on-wall gating (`kinematics.py`) | Built, checked on 4 clips | Requires a steadily tracked, fully framed, normal-sized climber who is off the mat. |
| Instability score + events (`stability.py`) | Built, checked on 4 clips | Weight off the feet, feet cut loose, COM lurch; stalls logged separately. |
| Drop detection, fall vs. dismount (`falls.py`) | Built, checked on 3 drops | Found both falls and the one jump-off, labelled all three correctly. |
| Lead-up to each drop (`falls.py`) | Built | Instability in the last 1s / 3s vs. the rest of the attempt. |
| v1 COM-over-base heuristic (`features.py`) | Kept, now gated to on-wall frames | 50 flags -> 13 on the four clips. |
| Overlay (`overlay.py`) | Built, spot-checked visually | Status strip, instability bar, event and fall banners. |
| Review tooling (`scripts/`) | Built | `review_flags.py` contact sheets; `summarize_stability.py` per-clip table checked against `_fall` / `_send` filenames. |

## Results so far

- `analysis/flag_review.csv`: verdict for each of the 50 v1 flags
  (36 false positive, 6 true but not a fault, 6 plausibly correct, 1 fall, 1 uncertain).
- `analysis/stability_summary.csv`: one row per clip under v1.1.

| Clip | Detected | On wall | Unstable |
|---|---|---|---|
| route1 (fall) | fall at 25.5s | 15.0s | 3.8% |
| route2 (send) | no drop seen (climbs out of frame) | 4.8s | 4.9% |
| route3 (fall) | fall at 39.8s | 25.0s | 57.5% |
| route4 (send) | dismount at 18.1s | 10.2s | 2.3% |

## Known gaps / risks

- **Built and checked on the same four clips.** The results above are a best
  case. Every threshold in `kinematics.py`, `stability.py` and `falls.py`
  was chosen while looking at these clips.
- **Fall vs. dismount rests on three drops.** Margins are thin: torso
  rotation was 44 degrees on a fall and 37 on the dismount, against a 60
  degree threshold. Other rules carried those two cases.
- **Missed events have never been checked.** The review judged flags that
  fired, not moments that should have been flagged and were not.
- **Camera angle.** All four clips are filmed from behind, so "weight off
  the feet" measures sideways offset. Side-on footage would measure distance
  from the wall instead and has not been tried.
- **On-wall detection uses fixed thresholds** relative to the lowest point
  the feet reach. A clip where the climber never stands on the mat in frame
  would get the mat height wrong.
- **Holding the finish counts as a stall**, the same as hanging on before a fall.
- **Single-person assumption.** The pipeline takes the first detected person
  per frame; untested with anyone else in shot.
- **COM model is an approximation** (60% hip / 40% shoulder midpoint).

## Footage needed before the next stage

The next stage is the remaining v1.x heuristics (hip rotation, dead-point
timing, splitting a session into attempts). Before that, v1.1 needs to be
checked on clips it was not built on. Target: **about 20 new clips**.

| Clips | What | Why |
|---|---|---|
| 8 | Falls, as varied as possible: foot slip, missed dynamic move, strength giving out, falling from low and from high | Two falls is too few to trust the fall rules; the two so far already look completely different |
| 6 | Sends that end with a jump-off | Only one dismount so far, and it is the case most likely to be mislabelled as a fall |
| 3 | Sends that felt shaky but stayed on | Tests whether instability is flagged without a fall to explain it |
| 2 | The same route filmed once from behind and once side-on | Shows how much the camera angle changes the scores |
| 1 | One longer recording with 3 or more attempts in it | Needed for splitting a session into attempts |

For every clip:

- Tripod or fixed mount; whole body in frame for the whole climb, including
  the top of the route and the landing.
- Include a second or two of standing on the mat in frame before starting.
- Start and stop the recording without covering the lens if possible.
- Name files `YYYYMMDD_routeN_attemptN_fall.mp4` or `..._send.mp4`.
- For falls, a few words on what went wrong (in the filename or a note)
  makes the review far more useful.

If 20 is too many for one session, the falls and jump-offs (14 clips)
matter most.

## Immediate next steps

See `V1.X_TASKS.md` for the full checklist. Priority order:
1. Film the clips above
2. Run them through `main.py` and `scripts/summarize_stability.py` with no threshold changes; record how many outcomes are right
3. Review instability events on the new clips, including moments that were missed
4. Only then tune thresholds, and start the next heuristics
