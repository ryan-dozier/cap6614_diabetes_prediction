#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
OUTPUT_DIR="$(python3 -c 'from src.paths import NHANES_RAW_DIR; print(NHANES_RAW_DIR)')"

BASE_URL="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles"

mkdir -p "$OUTPUT_DIR"

FILES=(
    # Demographics
    DEMO_I

    # Examination
    BMX_I
    BPX_I

    # Laboratory
    TCHOL_I
    HDL_I
    TRIGLY_I
    GHB_I
    GLU_I
    INS_I
    OGTT_I

    # Questionnaires
    DIQ_I
    PAQ_I
    SLQ_I
    SMQ_I
    ALQ_I
    OCQ_I
    BPQ_I
    MCQ_I
)

echo "========================================"
echo "Downloading NHANES 2015-2016"
echo "========================================"
echo
echo "Destination:"
echo "  $OUTPUT_DIR"
echo

for FILE in "${FILES[@]}"; do
    DEST="$OUTPUT_DIR/${FILE}.XPT"
    URL="$BASE_URL/${FILE}.XPT"

    if [ -f "$DEST" ]; then
        echo "[skip] ${FILE}.XPT already exists"
        continue
    fi

    echo "[download] ${FILE}.XPT"

    curl \
        --fail \
        --location \
        --retry 3 \
        --output "$DEST" \
        "$URL"
done

echo
echo "========================================"
echo "NHANES download complete."
echo "========================================"