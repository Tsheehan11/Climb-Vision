"""
summarize_stability.py

Collects every outputs/<clip>/report.json into one row per clip so falls
and sends can be compared side by side:

    analysis/stability_summary.csv

Clips named like 20260915_route1_attempt1_fall.mp4 carry their real
outcome in the filename (_fall / _send); that is compared against what
the pipeline detected, so this doubles as an accuracy check.

Usage:
    python scripts/summarize_stability.py
    python scripts/summarize_stability.py --outputs outputs --csv analysis/stability_summary.csv
"""

import argparse
import csv
import json
from pathlib import Path

# what the pipeline may report for each real outcome
AGREES = {"fall": {"fall"}, "send": {"dismount", "no_drop_seen"}}


def summarize(outputs: Path) -> list[dict]:
    rows = []
    for report_path in sorted(outputs.glob("*/report.json")):
        report = json.loads(report_path.read_text())
        if "summary" not in report:  # a v1 report: re-run main.py on that clip
            continue
        clip, s = report_path.parent.name, report["summary"]
        labelled = clip.rsplit("_", 1)[-1] if clip.rsplit("_", 1)[-1] in AGREES else ""
        drop = (report["drops"] or [None])[-1]
        lead = drop["lead_up"] if drop else {}
        rows.append({
            "clip": clip,
            "labelled_outcome": labelled,
            "detected_outcome": s["outcome"],
            "agrees": "" if not labelled else s["outcome"] in AGREES[labelled],
            "on_wall_s": s["on_wall_s"],
            "unstable_s": s["unstable_s"],
            "unstable_pct_of_on_wall": s["unstable_pct_of_on_wall"],
            "mean_instability": s["mean_instability"],
            "longest_hold_s": s["longest_hold_s"],
            "off_base_events": s["event_counts"].get("off_base", 0),
            "feet_cut_events": s["event_counts"].get("feet_cut", 0),
            "jolt_events": s["event_counts"].get("jolt", 0),
            "stall_events": s["event_counts"].get("stall", 0),
            "drop_at_s": drop["release_s"] if drop else "",
            "drop_height_body": drop["height"] if drop else "",
            "unstable_before_release": lead.get("unstable_before_release", ""),
            "unstable_s_in_last_5s": lead.get("unstable_s_in_last_5s", ""),
            "mean_instability_last_3s": lead.get("mean_instability_last_3s", ""),
            "mean_instability_rest_of_attempt": lead.get("mean_instability_rest_of_attempt", ""),
            "held_position_before_release_s": lead.get("held_position_before_release_s", ""),
            "why": "; ".join(drop["reasons"]) if drop else "",
        })
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="One row per clip: stability and outcome.")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--csv", default="analysis/stability_summary.csv")
    args = ap.parse_args()

    rows = summarize(Path(args.outputs))
    if not rows:
        raise SystemExit(f"No reports found under {args.outputs}/ - run main.py first.")
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        print(f"{r['clip']}: labelled {r['labelled_outcome'] or '?'}, detected {r['detected_outcome']}, "
              f"unstable {r['unstable_pct_of_on_wall']}% of {r['on_wall_s']}s on the wall")
    print(f"Wrote {args.csv}")
