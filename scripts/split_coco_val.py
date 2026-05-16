"""
Phase 2 — Step 2: Split COCO val2017 into Train / Val / Test

takes the 5,000 images (and their YOLO labels) from download_coco_val.py
and splits them into train / val / test sets.

split ratio: 80 / 15 / 5
  train : 4,000 images  <- model learns from these
  val   :   750 images  <- checked after each epoch
  test  :   250 images  <- touched only at the very end

same logic as phase 1 (split_dataset.py) just pointed at the new data.
images without any annotations are kept as background images — yolo handles them fine.

saves to:
  data/coco_val2017/split/images/train/   val/   test/
  data/coco_val2017/split/labels/train/   val/   test/
"""

import random
import shutil
from pathlib import Path

random.seed(42)

SRC_IMAGES = Path(__file__).parent.parent / "data" / "coco_val2017" / "val2017"
SRC_LABELS = Path(__file__).parent.parent / "data" / "coco_val2017" / "labels"
SPLIT_DIR  = Path(__file__).parent.parent / "data" / "coco_val2017" / "split"

# ── Create output folders ─────────────────────────────────────────────────────
for split in ("train", "val", "test"):
    (SPLIT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# ── Gather all images and shuffle ─────────────────────────────────────────────
all_images = sorted(SRC_IMAGES.glob("*.jpg"))
random.shuffle(all_images)

total = len(all_images)
n_train = int(total * 0.80)
n_val   = int(total * 0.15)
# test gets the remainder so nothing is lost to rounding

train_imgs = all_images[:n_train]
val_imgs   = all_images[n_train : n_train + n_val]
test_imgs  = all_images[n_train + n_val:]

print("=== Phase 2 — Split COCO val2017 ===\n")
print(f"Total images  : {total}")
print(f"  train       : {len(train_imgs)}")
print(f"  val         : {len(val_imgs)}")
print(f"  test        : {len(test_imgs)}")

# ── Copy files into split folders ──────────────────────────────────────────────
def copy_split(images, split_name):
    background = 0
    for img_path in images:
        dst_img = SPLIT_DIR / "images" / split_name / img_path.name
        shutil.copy2(img_path, dst_img)

        lbl_src = SRC_LABELS / (img_path.stem + ".txt")
        dst_lbl = SPLIT_DIR / "labels" / split_name / lbl_src.name
        if lbl_src.exists():
            shutil.copy2(lbl_src, dst_lbl)
        else:
            # image has no objects — valid background image, leave label absent
            background += 1
    return background

print()
for imgs, name in [(train_imgs, "train"), (val_imgs, "val"), (test_imgs, "test")]:
    bg = copy_split(imgs, name)
    note = f"  ({bg} background images)" if bg else ""
    print(f"  {name:5s} -> {len(imgs)} images{note}")

print(f"\nSplit saved to {SPLIT_DIR}")
print(f"\nNext -> python analyse_dataset.py")
