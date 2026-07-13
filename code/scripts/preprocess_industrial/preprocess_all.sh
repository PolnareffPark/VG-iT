#!/bin/bash
# Run all industrial dataset preprocessing scripts
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Industrial Dataset Preprocessing"
echo "=========================================="
echo "Working directory: $PROJECT_ROOT"
echo ""

datasets=(basf kamp bdg2 ashrae sdwpf care_wind)
success=0
failed=0

for ds in "${datasets[@]}"; do
    echo "--- Preprocessing: $ds ---"
    if python "$SCRIPT_DIR/preprocess_${ds}.py"; then
        success=$((success + 1))
    else
        echo "[ERROR] $ds preprocessing failed!"
        failed=$((failed + 1))
    fi
    echo ""
done

echo "=========================================="
echo "Preprocessing Complete"
echo "  Success: $success / ${#datasets[@]}"
echo "  Failed:  $failed / ${#datasets[@]}"
echo "=========================================="

if [ "$failed" -gt 0 ]; then
    exit 1
fi
