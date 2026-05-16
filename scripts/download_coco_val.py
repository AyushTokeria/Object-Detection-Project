"""
Phase 2 — Step 1: Download and Convert COCO val2017

Downloads the official COCO val2017 dataset (5,000 images) and converts
the annotations from COCO JSON format to YOLO txt format.

why val2017 and not train2017?
  train2017 has 118,000 images — would take days on a consumer GPU.
  val2017 has 5,000 images — 40x more than phase 1, trains in under 2 hours,
  and all 80 classes are properly represented (not starved like in coco128).

COCO annotation format (what we download):
  bbox = [x_min, y_min, width, height]  pixel coords, top-left corner

YOLO label format (what we need):
  class_id  cx  cy  w  h               normalised 0-1, centre of box

one tricky thing: COCO category IDs go from 1 to 90 but with gaps
(some numbers are unused). YOLO needs contiguous 0-indexed class IDs.
so we use a fixed mapping table — the same one ultralytics uses internally.

saves to:
  data/coco_val2017/val2017/   <- the 5000 jpg images (raw download)
  data/coco_val2017/labels/    <- one .txt per image, YOLO format
"""

import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "coco_val2017"
DATA_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_URL      = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# COCO category IDs are 1-90 with gaps. YOLO needs 0-79, contiguous.
# this is the standard ultralytics mapping — do not change the order.
COCO_TO_YOLO = {
     1:  0,  2:  1,  3:  2,  4:  3,  5:  4,  6:  5,  7:  6,  8:  7,  9:  8, 10:  9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
    22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
    35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
    46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
    67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
    80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}


def download_with_progress(url, dest):
    def _report(count, block, total):
        done = count * block / 1024 / 1024
        total_mb = total / 1024 / 1024
        print(f"\r  {done:.1f} / {total_mb:.1f} MB", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_report)
    print()


# ── 1. Download ────────────────────────────────────────────────────────────────
print("=== Phase 2 — Download COCO val2017 ===\n")

img_zip = DATA_DIR / "val2017.zip"
ann_zip = DATA_DIR / "annotations_trainval2017.zip"

if not img_zip.exists():
    print("Downloading images (~1 GB)...")
    download_with_progress(IMAGES_URL, img_zip)
else:
    print("Images zip already present, skipping download.")

if not ann_zip.exists():
    print("Downloading annotations (~241 MB)...")
    download_with_progress(ANNOTATIONS_URL, ann_zip)
else:
    print("Annotations zip already present, skipping download.")

# ── 2. Extract ─────────────────────────────────────────────────────────────────
img_dir  = DATA_DIR / "val2017"
ann_path = DATA_DIR / "annotations" / "instances_val2017.json"

if not img_dir.exists():
    print("\nExtracting images...")
    with zipfile.ZipFile(img_zip) as z:
        z.extractall(DATA_DIR)
    print(f"  -> {img_dir}")
else:
    print(f"Images already extracted ({len(list(img_dir.glob('*.jpg')))} files).")

if not ann_path.exists():
    print("Extracting annotations...")
    with zipfile.ZipFile(ann_zip) as z:
        z.extractall(DATA_DIR)
    print(f"  -> {ann_path}")
else:
    print("Annotations already extracted.")

# ── 3. Load annotations JSON ───────────────────────────────────────────────────
print(f"\nLoading {ann_path.name}...")
with open(ann_path, encoding="utf-8") as f:
    coco = json.load(f)

images_info = {img["id"]: img for img in coco["images"]}

annotations_by_image = defaultdict(list)
for ann in coco["annotations"]:
    annotations_by_image[ann["image_id"]].append(ann)

print(f"  images      : {len(images_info)}")
print(f"  annotations : {len(coco['annotations'])}")
print(f"  categories  : {len(coco['categories'])}")

# ── 4. Convert to YOLO format ──────────────────────────────────────────────────
# COCO bbox is [x_min, y_min, width, height] in pixels.
# YOLO needs [class_id, cx, cy, w, h] all normalised to image dimensions.
out_labels = DATA_DIR / "labels"
out_labels.mkdir(exist_ok=True)

print(f"\nConverting to YOLO format...")
skipped_crowd = 0
skipped_cat   = 0

for img_id, img_info in images_info.items():
    img_w = img_info["width"]
    img_h = img_info["height"]
    stem  = Path(img_info["file_name"]).stem

    lines = []
    for ann in annotations_by_image[img_id]:
        if ann.get("iscrowd", 0):
            # crowd annotations mark groups of objects — not suitable for
            # per-instance detection. yolo ignores them, so we skip them.
            skipped_crowd += 1
            continue

        cat_id = ann["category_id"]
        if cat_id not in COCO_TO_YOLO:
            skipped_cat += 1
            continue

        yolo_cls = COCO_TO_YOLO[cat_id]
        x_min, y_min, bw, bh = ann["bbox"]

        cx = (x_min + bw / 2) / img_w
        cy = (y_min + bh / 2) / img_h
        w  = bw / img_w
        h  = bh / img_h

        # a handful of COCO annotations are very slightly out of bounds — clamp them
        cx, cy, w, h = [max(0.0, min(1.0, v)) for v in (cx, cy, w, h)]
        lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(out_labels / f"{stem}.txt", "w") as f:
        f.write("\n".join(lines))

total_labels = len(list(out_labels.glob("*.txt")))
print(f"  Label files written : {total_labels}")
print(f"  Crowd annotations skipped : {skipped_crowd}")
print(f"  Unknown category IDs skipped : {skipped_cat}")
print(f"\nDone. Images at {img_dir}")
print(f"      Labels at {out_labels}")
print(f"\nNext -> python split_coco_val.py")
