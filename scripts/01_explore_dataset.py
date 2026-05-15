"""
Step 1 — Download and Explore the Dataset

Before we train anything, we want to actually LOOK at the data.
This is a habit every good ML engineer has — never train blind.

We're using COCO128: 128 images from the COCO dataset, already
formatted for YOLOv8. Think of it as the "Hello World" of object detection.
"""

from ultralytics import settings
from ultralytics.utils import downloads
from pathlib import Path
import yaml
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random

# ── Where to save the dataset ────────────────────────────────────────────────
# We want it inside our project folder so everything stays organized.
# By default ultralytics dumps datasets into C:/Users/You/datasets — we override that.
DATA_DIR = Path("../data")
DATA_DIR.mkdir(exist_ok=True)

# ── Download COCO128 ─────────────────────────────────────────────────────────
# ultralytics has a helper that downloads datasets by name.
# coco128.yaml is a config file that tells YOLO where the images/labels live
# and what the 80 class names are. We'll look at it in a second.
print("Downloading COCO128 dataset (this is small, ~7MB)...")
downloads.download(
    "https://ultralytics.com/assets/coco128.zip",
    dir=DATA_DIR,
    unzip=True,
)
print("Done.")

# ── Look at the config file ──────────────────────────────────────────────────
# Every YOLO dataset ships with a .yaml file that acts like a "readme" for the data.
# It tells the model: where are the images, how many classes, what are they called.
# The zip doesn't include the yaml, but ultralytics bundles it with its own install.
import ultralytics as _ult
yaml_path = Path(_ult.__file__).parent / "cfg" / "datasets" / "coco128.yaml"
with open(yaml_path, encoding="utf-8") as f:
    dataset_config = yaml.safe_load(f)

print("\n── Dataset Config (coco128.yaml) ──────────────────────────")
# 'names' is a dict like {0: 'person', 1: 'bicycle', ...}
class_names = dataset_config["names"]
num_classes = len(class_names)
first_10 = [class_names[i] for i in range(min(10, num_classes))]
print(f"  Number of classes : {num_classes}")
print(f"  First 10 classes  : {first_10}")
print(f"  ...and {num_classes - 10} more")

# ── Count what we have ───────────────────────────────────────────────────────
images_dir = DATA_DIR / "coco128" / "images" / "train2017"
labels_dir = DATA_DIR / "coco128" / "labels" / "train2017"

image_files = sorted(images_dir.glob("*.jpg"))
label_files = sorted(labels_dir.glob("*.txt"))

print(f"\n── Files on Disk ───────────────────────────────────────────")
print(f"  Images : {len(image_files)}")
print(f"  Labels : {len(label_files)}")
# These should match — every image needs exactly one label file.
# (Some label files can be empty if the image has no objects — that's valid.)

# ── Peek at a single label file ──────────────────────────────────────────────
# Let's open one label file and decode what's inside.
# This is the raw format YOLO trains on — worth seeing at least once.
sample_label = label_files[0]
print(f"\n── Sample label file: {sample_label.name} ──────────────────")
with open(sample_label) as f:
    lines = f.readlines()

print(f"  {len(lines)} object(s) annotated in this image:")
for line in lines:
    parts = line.strip().split()
    cls_id = int(parts[0])
    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    print(f"    class={cls_id} ({class_names[cls_id]:15s})  "
          f"center=({cx:.3f}, {cy:.3f})  size=({w:.3f} x {h:.3f})")

# ── Visualize a few images with their ground-truth boxes ────────────────────
# "Ground truth" = the human-drawn boxes, NOT model predictions.
# This is what the model is trying to learn to reproduce.
print("\nGenerating visualization of 6 random training images...")

random.seed(42)  # same seed = same 6 images every time, reproducible
sample_images = random.sample(image_files, 6)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("COCO128 — Ground Truth Labels (before any training)", fontsize=14, fontweight="bold")

# one color per class — we have 80 classes so we spread across a colormap
cmap = plt.colormaps.get_cmap("tab20")

for ax, img_path in zip(axes.flat, sample_images):
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_bgr.shape[:2]  # actual pixel dimensions

    ax.imshow(img_rgb)
    ax.set_title(img_path.stem, fontsize=8)
    ax.axis("off")

    # load the matching label file (same name, different folder + extension)
    label_path = labels_dir / (img_path.stem + ".txt")
    if not label_path.exists():
        continue  # some images have no annotations — totally fine

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # convert from normalized (0–1) back to pixel coordinates for drawing
            # YOLO stores center + size; matplotlib needs top-left corner + size
            x1 = (cx - bw / 2) * w_img
            y1 = (cy - bh / 2) * h_img
            box_w = bw * w_img
            box_h = bh * h_img

            color = cmap(cls_id % 20)
            rect = patches.Rectangle((x1, y1), box_w, box_h,
                                      linewidth=1.5, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1 - 2, class_names[cls_id],
                    fontsize=6, color="white",
                    bbox=dict(facecolor=color, pad=1, edgecolor="none", alpha=0.8))

plt.tight_layout()
output_path = Path("../runs/coco128_ground_truth.png")
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved → {output_path}")
plt.show()
