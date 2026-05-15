"""
Step 4 — Create the Training Config (dataset.yaml)

YOLOv8 doesnt just take a folder of images and figure it out.
You have to give it a config file that tells it:

  - where the train images are
  - where the val images are
  - where the test images are
  - how many classes there are
  - what each class is called

this file is called a dataset yaml and its the thing you pass to model.train()
when we eventually train. without it yolo doesnt know what it's looking at.

think of it like the schema file for a database — the model needs to know the
structure before it can work with the data.
"""

import yaml
from pathlib import Path
import ultralytics as _ult

# ── Load the original class names from coco128 ───────────────────────────────
src_yaml = Path(_ult.__file__).parent / "cfg" / "datasets" / "coco128.yaml"
with open(src_yaml, encoding="utf-8") as f:
    src_cfg = yaml.safe_load(f)

class_names = src_cfg["names"]  # {0: 'person', 1: 'bicycle', ...}

# ── Build our config ──────────────────────────────────────────────────────────
# path is the root of the dataset — everything else is relative to it.
# yolo resolves these paths at training time.
split_dir = Path("../data/split").resolve()

config = {
    "path"  : str(split_dir),
    "train" : "images/train",
    "val"   : "images/val",
    "test"  : "images/test",
    "nc"    : len(class_names),       # number of classes
    "names" : class_names,            # the dict of {id: name}
}

# ── Save it ───────────────────────────────────────────────────────────────────
out_path = Path("../data/dataset.yaml")
with open(out_path, "w", encoding="utf-8") as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print(f"Saved config -> {out_path.resolve()}")
print(f"\nContents:")
print(f"  path   : {config['path']}")
print(f"  train  : {config['train']}")
print(f"  val    : {config['val']}")
print(f"  test   : {config['test']}")
print(f"  nc     : {config['nc']}")
print(f"  names  : (first 5) {[class_names[i] for i in range(5)]}")
