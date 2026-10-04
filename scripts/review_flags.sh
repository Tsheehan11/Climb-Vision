#!/usr/bin/env bash
#
# review_flags.sh
#
# For every flag in a report.json, extracts a short burst of individual
# frames (as numbered PNGs) centered on the flagged frame, then opens
# the folder in File Explorer. No video playback involved — you just
# browse the stills, and the flagged frame is labeled so there's no
# guessing which one it is.
#
# Usage:
#   ./scripts/review_flags.sh outputs/20260915_route1_attempt1_fall/annotated.mp4 \
#                              outputs/20260915_route1_attempt1_fall/report.json \
#                              [context_frames]
#
# context_frames = how many frames before/after the flag to extract
# (default 8 — at ~15fps that's roughly half a second either side)

set -e

VIDEO="$1"
REPORT="$2"
CONTEXT_FRAMES="${3:-8}"

if [[ -z "$VIDEO" || -z "$REPORT" ]]; then
    echo "Usage: $0 <video.mp4> <report.json> [context_frames]"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg not found — make sure it's on your PATH."
    exit 1
fi

# Extract to a temp folder (not your repo) so nothing lingers on disk
TMP_ROOT="$(mktemp -d)"
BASE_OUT="$TMP_ROOT/flag_review"
mkdir -p "$BASE_OUT"

FLAGS=$(python3 -c "
import json
with open('$REPORT') as f:
    data = json.load(f)
for fl in data['flags']:
    print(f\"{fl['frame']}|{fl['timestamp_s']}|{fl['message']}\")
")

if [[ -z "$FLAGS" ]]; then
    echo "No flags found in $REPORT"
    exit 0
fi

COUNT=$(echo "$FLAGS" | wc -l)
echo "Found $COUNT flag(s). Extracting frames..."
echo "---"

i=1
while IFS='|' read -r FRAME TS MSG; do
    OUTDIR="$BASE_OUT/flag${i}_frame${FRAME}"
    mkdir -p "$OUTDIR"
    echo "[$i/$COUNT] Frame $FRAME (${TS}s): $MSG"

    START=$((FRAME - CONTEXT_FRAMES))
    if [[ $START -lt 0 ]]; then START=0; fi
    END=$((FRAME + CONTEXT_FRAMES))

    for ((f=START; f<=END; f++)); do
        if [[ $f -eq $FRAME ]]; then
            NAME="frame${f}_FLAGGED.png"
        else
            NAME="frame${f}.png"
        fi
        ffmpeg -y -nostdin -i "$VIDEO" -vf "select=eq(n\,$f)" -vframes 1 -loglevel error "$OUTDIR/$NAME" < /dev/null
    done

    i=$((i + 1))
done <<< "$FLAGS"

echo "---"
echo "Done. Frames saved to a temp folder: $BASE_OUT"
echo "(each flag gets its own folder; the exact flagged frame is named *_FLAGGED.png)"

# Open the folder in File Explorer so you can browse straight away
if command -v explorer.exe &> /dev/null; then
    if command -v cygpath &> /dev/null; then
        WIN_PATH="$(cygpath -w "$BASE_OUT")"
    else
        WIN_PATH="$BASE_OUT"
    fi
    echo "Opening: $WIN_PATH"
    explorer.exe "$WIN_PATH" || true
else
    echo "(explorer.exe not found on PATH — open this folder manually: $BASE_OUT)"
fi

echo ""
read -p "Press Enter once you're done reviewing to delete these temp frames (or Ctrl+C to keep them)... "
rm -rf "$TMP_ROOT"
echo "Cleaned up. Nothing left on disk."