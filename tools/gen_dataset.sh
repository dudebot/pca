#!/bin/bash
# gen_dataset.sh — Run solver + recorder on synthetic tasks to generate training data
#
# Usage: ./tools/gen_dataset.sh <tasks_dir> <output_dir> [max_depth] [neg_stride]

set -e

TASKS_DIR="${1:?Usage: gen_dataset.sh <tasks_dir> <output_dir> [max_depth] [neg_stride]}"
OUTPUT_DIR="${2:?Usage: gen_dataset.sh <tasks_dir> <output_dir> [max_depth] [neg_stride]}"
MAX_DEPTH="${3:-6}"
NEG_STRIDE="${4:-4}"

mkdir -p "$OUTPUT_DIR"

TASKS=($(ls "$TASKS_DIR"/*.json 2>/dev/null))
TOTAL=${#TASKS[@]}

echo "Processing $TOTAL tasks (max_depth=$MAX_DEPTH, neg_stride=$NEG_STRIDE)..."

SOLVED=0
FAILED=0
TASK_ID=0

for task in "${TASKS[@]}"; do
    NAME=$(basename "$task" .json)

    # Run solver with recording, timeout at 30 seconds per task
    if timeout 30 ./enumerate "$task" -d "$MAX_DEPTH" --record "$OUTPUT_DIR" \
        --task-id "$TASK_ID" --neg-stride "$NEG_STRIDE" > /dev/null 2>&1; then
        SOLVED=$((SOLVED + 1))
    else
        FAILED=$((FAILED + 1))
    fi

    TASK_ID=$((TASK_ID + 1))

    # Progress
    DONE=$((SOLVED + FAILED))
    if [ $((DONE % 50)) -eq 0 ]; then
        echo "  $DONE/$TOTAL done ($SOLVED solved, $FAILED failed)"
    fi
done

echo ""
echo "Done: $SOLVED solved, $FAILED failed out of $TOTAL tasks"
echo "Output: $OUTPUT_DIR"
ls "$OUTPUT_DIR"/states_*.bin 2>/dev/null | wc -l | xargs echo "  state files:"
ls "$OUTPUT_DIR"/edges_*.bin 2>/dev/null | wc -l | xargs echo "  edge files:"
du -sh "$OUTPUT_DIR" | cut -f1 | xargs echo "  total size:"
