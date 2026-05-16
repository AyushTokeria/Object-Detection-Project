"""
Step 2 — Split the Dataset

Right now all 128 images are sitting in one folder called train2017.
Thats fine for looking at the data but before we train we need to split
it into three separate buckets:

  train  — what the model actually learns from (we use 80%)
  val    — checked after every epoch to see if the model is improving (15%)
  test   — touched only ONCE at the very end to get a final honest score (5%)

why three splits and not just two?

  if you only had train + test you would end up peeking at your test set
  every time you tweak a hyperparameter. over time you'd accidentally overfit
  TO the test set just by making decisions based on it. val exists so you can
  tune freely without contaminating your final evaluation.

  think of it like this:
    train = your textbook
    val   = practice exams you take while studying
    test  = the real final exam, opened only once

with only 128 images this split is small, but the principle is the same
whether you have 128 or 128,000 images.
"""

import shutil
import random
from pathlib import Path

random.seed(42)  # fixing the seed means we get the same split every time we run this

# ── Source folders (where the data currently lives) ──────────────────────────
SRC_IMAGES = Path("../data/coco128/images/train2017")
SRC_LABELS = Path("../data/coco128/labels/train2017")

# ── Destination (our new organised split) ────────────────────────────────────
SPLIT_DIR = Path("../data/split")

# ── Split ratios ─────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.15
TEST_RATIO  = 0.05
# these should add up to 1.0 — just a sanity check
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

# ── Get all image paths and shuffle them ─────────────────────────────────────
# shuffling before splitting is important — the files are named with IDs that
# might have some ordering bias. we want each split to be a random sample.
all_images = sorted(SRC_IMAGES.glob("*.jpg"))
random.shuffle(all_images)

total = len(all_images)
n_train = int(total * TRAIN_RATIO)
n_val   = int(total * VAL_RATIO)
# test gets whatever is left so we dont lose any images to rounding
n_test  = total - n_train - n_val

train_imgs = all_images[:n_train]
val_imgs   = all_images[n_train:n_train + n_val]
test_imgs  = all_images[n_train + n_val:]

print(f"Total images : {total}")
print(f"Train        : {len(train_imgs)}  ({len(train_imgs)/total:.0%})")
print(f"Val          : {len(val_imgs)}   ({len(val_imgs)/total:.0%})")
print(f"Test         : {len(test_imgs)}    ({len(test_imgs)/total:.0%})")

# ── Copy files into the new folder structure ──────────────────────────────────
# we COPY rather than move so the original data/coco128 folder stays intact.
# good habit — never destroy your source data.
def copy_split(image_list, split_name):
    img_out = SPLIT_DIR / "images" / split_name
    lbl_out = SPLIT_DIR / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_labels = 0

    for img_path in image_list:
        # copy the image
        shutil.copy(img_path, img_out / img_path.name)

        # copy the matching label file (same stem, .txt extension)
        lbl_path = SRC_LABELS / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy(lbl_path, lbl_out / lbl_path.name)
            copied += 1
        else:
            # if no label file exists the image has no annotated objects.
            # yolo handles this fine — we just note it.
            missing_labels += 1

    print(f"  [{split_name}] copied {copied} image+label pairs, {missing_labels} images had no label")

print("\nCopying files...")
copy_split(train_imgs, "train")
copy_split(val_imgs,   "val")
copy_split(test_imgs,  "test")

print(f"\nDone. Split saved to: {SPLIT_DIR.resolve()}")

# ── Quick verification ────────────────────────────────────────────────────────
# just double-check the counts on disk match what we expected
for split in ["train", "val", "test"]:
    n_imgs = len(list((SPLIT_DIR / "images" / split).glob("*.jpg")))
    n_lbls = len(list((SPLIT_DIR / "labels" / split).glob("*.txt")))
    print(f"  {split:5s} -> {n_imgs} images, {n_lbls} labels on disk")
