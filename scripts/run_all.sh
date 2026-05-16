#!/usr/bin/env bash
# runs the full pipeline in order from the scripts directory

set -e  # stop immediately if any script fails

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPTS_DIR"

echo "=== PHASE 1 — COCO128 / yolo11n ==="
echo ""

echo "=== 1/6  explore dataset ==="
python explore_dataset.py

echo "=== 2/6  split dataset ==="
python split_dataset.py

echo "=== 3/6  analyse dataset ==="
python analyse_dataset.py

echo "=== 4/6  create config ==="
python create_config.py

echo "=== 5/6  pre-training check ==="
python pre_training_check.py

echo "=== 6/6  train ==="
python train.py

echo ""
echo "=== PHASE 2 — COCO val2017 / yolo11s ==="
echo ""

echo "=== 1/4  download coco val2017 ==="
python download_coco_val.py

echo "=== 2/4  split coco val2017 ==="
python split_coco_val.py

echo "=== 3/4  create config ==="
python create_config.py

echo "=== 4/4  train ==="
python train.py

echo ""
echo "pipeline complete."
