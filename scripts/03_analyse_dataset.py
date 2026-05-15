"""
Step 3 — Analyse the Dataset

Before training you want to understand your data properly.
Surprises at this stage are cheap. Surprises after a 3-hour training run are not.

We're going to look at:
  1. class distribution  — are some classes way more common than others?
  2. objects per image   — how crowded are these images?
  3. bounding box sizes  — are objects mostly big or small?

why does this matter?

  if 90% of your labels are "person" and only 1% are "toothbrush", the model
  is going to be much better at finding people than toothbrushes. knowing this
  upfront helps you interpret your results later — if toothbrush detection is bad
  you'll know it's partly a data problem, not just a model problem.

  this is the same intuition as class imbalance in tabular ML (like HouseIQ),
  just applied to object categories instead of target values.
"""

from pathlib import Path
from collections import Counter
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
import ultralytics as _ult

# ── Load class names from the coco128 yaml ────────────────────────────────────
yaml_path = Path(_ult.__file__).parent / "cfg" / "datasets" / "coco128.yaml"
with open(yaml_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
class_names = cfg["names"]  # dict like {0: 'person', 1: 'bicycle', ...}

# ── We'll analyse the full dataset (all splits combined) ─────────────────────
# for a dataset this small it makes sense to look at everything together.
LABELS_DIR = Path("../data/split")
all_label_files = (
    list((LABELS_DIR / "labels" / "train").glob("*.txt")) +
    list((LABELS_DIR / "labels" / "val").glob("*.txt")) +
    list((LABELS_DIR / "labels" / "test").glob("*.txt"))
)
all_image_files = (
    list((LABELS_DIR / "images" / "train").glob("*.jpg")) +
    list((LABELS_DIR / "images" / "val").glob("*.jpg")) +
    list((LABELS_DIR / "images" / "test").glob("*.jpg"))
)

print(f"Analysing {len(all_label_files)} label files across all splits...")

# ── Parse every label file ────────────────────────────────────────────────────
class_counts  = Counter()   # how many times each class appears total
objects_per_image = []      # list of object counts per image
box_widths  = []            # normalised widths of all boxes
box_heights = []            # normalised heights of all boxes

for lbl_path in all_label_files:
    with open(lbl_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    objects_per_image.append(len(lines))

    for line in lines:
        parts = line.split()
        cls_id = int(parts[0])
        w, h   = float(parts[3]), float(parts[4])
        class_counts[cls_id] += 1
        box_widths.append(w)
        box_heights.append(h)

total_objects = sum(class_counts.values())
print(f"Total objects annotated : {total_objects}")
print(f"Unique classes present  : {len(class_counts)} out of 80")
print(f"Avg objects per image   : {np.mean(objects_per_image):.1f}")
print(f"Max objects in one image: {max(objects_per_image)}")

# top 10 most common classes
print("\nTop 10 most common classes:")
for cls_id, count in class_counts.most_common(10):
    bar = "#" * int(count / total_objects * 100)
    print(f"  {class_names[cls_id]:20s} {count:4d}  {bar}")

# ── Build the charts ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("COCO128 Dataset Analysis", fontsize=15, fontweight="bold")

# ── Chart 1: Top 20 classes by count ─────────────────────────────────────────
ax = axes[0, 0]
top20 = class_counts.most_common(20)
labels_20  = [class_names[cid] for cid, _ in top20]
counts_20  = [cnt for _, cnt in top20]
colors_20  = plt.colormaps.get_cmap("tab20")(np.linspace(0, 1, 20))
ax.barh(labels_20[::-1], counts_20[::-1], color=colors_20)
ax.set_xlabel("Number of instances")
ax.set_title("Top 20 classes by instance count")
ax.axvline(np.mean(counts_20), color="red", linestyle="--", linewidth=1, label=f"mean={np.mean(counts_20):.0f}")
ax.legend(fontsize=8)

# ── Chart 2: Objects per image distribution ───────────────────────────────────
ax = axes[0, 1]
ax.hist(objects_per_image, bins=20, color="steelblue", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Objects in image")
ax.set_ylabel("Number of images")
ax.set_title("How many objects are in each image?")
ax.axvline(np.mean(objects_per_image), color="red", linestyle="--", linewidth=1.5,
           label=f"mean = {np.mean(objects_per_image):.1f}")
ax.legend()

# ── Chart 3: Bounding box size scatter ───────────────────────────────────────
# plotting width vs height of every bounding box.
# boxes clustered near (0,0) = lots of small objects (hard to detect).
# boxes near (1,1) = objects that fill the frame (easy to detect).
ax = axes[1, 0]
ax.scatter(box_widths, box_heights, alpha=0.15, s=8, color="darkorange")
ax.set_xlabel("Box width (normalised 0-1)")
ax.set_ylabel("Box height (normalised 0-1)")
ax.set_title("Bounding box size distribution\n(each dot = one annotated object)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# ── Chart 4: Box area histogram ───────────────────────────────────────────────
# area = width * height. gives us a single number for "how big is this box".
# area < 0.01 = very small objects (less than 10% x 10% of the image)
box_areas = [w * h for w, h in zip(box_widths, box_heights)]
ax = axes[1, 1]
ax.hist(box_areas, bins=40, color="mediumseagreen", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Box area (normalised, 0-1)")
ax.set_ylabel("Count")
ax.set_title("Bounding box area distribution\n(how big are the objects?)")
small = sum(1 for a in box_areas if a < 0.01)
print(f"\nBoxes with area < 1% of image (small objects): {small} / {len(box_areas)} ({small/len(box_areas):.0%})")
ax.axvline(0.01, color="red", linestyle="--", linewidth=1.5, label="1% area threshold (small object)")
ax.legend(fontsize=8)

plt.tight_layout()
out = Path("../runs/dataset_analysis.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.show()
