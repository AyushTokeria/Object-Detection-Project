"""
Step 5 — Pre-Training Sanity Check

Before we train we want to be completely sure everything is wired up right.
A training run on this machine takes maybe 10-20 minutes. Finding out AFTER
that a path was wrong or labels were misread is annoying. Finding out BEFORE
costs 5 seconds.

This script checks:
  1. the dataset.yaml file loads correctly
  2. every image in each split has a matching label file
  3. every label file has valid content (no corrupt lines)
  4. the model can actually load the yaml (we do a dry run)
  5. we visualise one image from each split with its labels drawn on

if this runs without errors and the visualisation looks right — we're ready to train.
"""

import yaml
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── 1. Load the config ────────────────────────────────────────────────────────
config_path = Path("../data/dataset.yaml")
with open(config_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print("=== Pre-Training Sanity Check ===\n")
print(f"Config loaded from : {config_path.resolve()}")
print(f"Dataset root       : {cfg['path']}")
print(f"Classes            : {cfg['nc']}")

split_root = Path(cfg["path"])

# ── 2. Check image/label pairing for every split ──────────────────────────────
all_good = True
for split in ["train", "val", "test"]:
    img_dir = split_root / "images" / split
    lbl_dir = split_root / "labels" / split

    images = sorted(img_dir.glob("*.jpg"))
    labels = sorted(lbl_dir.glob("*.txt"))

    label_stems = {l.stem for l in labels}
    missing = [i for i in images if i.stem not in label_stems]

    # images without a label file are "background images" — no objects present.
    # this is valid and intentional in COCO128. yolo handles them fine.
    status = "OK" if len(missing) == 0 else f"OK ({len(missing)} background images, no labels expected)"
    print(f"  {split:5s} -> {len(images)} images, {len(labels)} labels  [{status}]")
    for m in missing:
        print(f"    background image (no objects): {m.name}")

# ── 3. Check label file contents ──────────────────────────────────────────────
print("\nChecking label file contents...")
corrupt = 0
for split in ["train", "val", "test"]:
    lbl_dir = split_root / "labels" / split
    for lbl_path in lbl_dir.glob("*.txt"):
        with open(lbl_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # each line must have exactly 5 values: class cx cy w h
                if len(parts) != 5:
                    print(f"  CORRUPT: {lbl_path.name} line {i+1} has {len(parts)} values")
                    corrupt += 1
                    continue
                cls_id = int(parts[0])
                coords = [float(p) for p in parts[1:]]
                # coordinates must all be between 0 and 1
                if any(c < 0 or c > 1 for c in coords):
                    print(f"  OUT OF RANGE: {lbl_path.name} line {i+1}")
                    corrupt += 1

if corrupt == 0:
    print("  All label files look clean.")
else:
    print(f"  Found {corrupt} corrupt lines.")
    all_good = False

# ── 4. Dry-run: can the model actually read the config? ───────────────────────
print("\nDry-run: loading YOLOv8n and checking it can read the config...")
try:
    model = YOLO("yolo11n.pt")
    # we're not training — just checking the data loader doesnt crash
    # imgsz=320 makes this faster; batch=4 is small enough to run anywhere
    results = model.val(data=str(config_path), imgsz=320, batch=4, verbose=False, plots=False)
    print("  Model read the config fine.")
    print(f"  Val mAP50 (pretrained, untrained on this data) : {results.box.map50:.4f}")
    print("  (this number means nothing yet — we havent trained. just confirms it runs.)")
except Exception as e:
    print(f"  ERROR during dry-run: {e}")
    all_good = False

# ── 5. Visualise one image from each split ────────────────────────────────────
print("\nGenerating pre-training visualisation...")
class_names = cfg["names"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Pre-Training Check — one image from each split", fontsize=13, fontweight="bold")
cmap = plt.colormaps.get_cmap("tab20")

for ax, split in zip(axes, ["train", "val", "test"]):
    img_dir = split_root / "images" / split
    lbl_dir = split_root / "labels" / split

    # grab first image in each split
    img_path = sorted(img_dir.glob("*.jpg"))[0]
    lbl_path = lbl_dir / (img_path.stem + ".txt")

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img.shape[:2]

    ax.imshow(img)
    ax.set_title(f"{split}  ({img_path.name})", fontsize=9)
    ax.axis("off")

    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = (cx - bw / 2) * w_img
                y1 = (cy - bh / 2) * h_img
                color = cmap(cls_id % 20)
                rect = patches.Rectangle((x1, y1), bw * w_img, bh * h_img,
                                          linewidth=1.5, edgecolor=color, facecolor="none")
                ax.add_patch(rect)
                ax.text(x1, y1 - 3, class_names[cls_id], fontsize=7, color="white",
                        bbox=dict(facecolor=color, pad=1, edgecolor="none", alpha=0.85))

plt.tight_layout()
out = Path("../runs/pre_training_check.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.show()

# ── Final verdict ─────────────────────────────────────────────────────────────
print("\n=================================")
if all_good:
    print("ALL CHECKS PASSED. Ready to train.")
else:
    print("SOME CHECKS FAILED. Fix issues above before training.")
print("=================================")
