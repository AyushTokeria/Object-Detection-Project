# Object Detection Project

object detection project using yolo11. still in progress dont judge it

---

## whats this

using yolo11 to detect objects in images. phase 1 done on coco128, phase 2 underway on coco val2017 with a bigger model. eventually gonna train on custom objects too.

built this after finishing HouseIQ (house price prediction) so this is my first time doing anything with images and neural networks

---

## what's been done so far

**step 1 - setup**
- made the project folder
- installed ultralytics, opencv, matplotlib
- connected to github

**step 2 - explore the dataset**
- downloaded COCO128 (128 images, 80 classes)
- wrote a script to explore it and look at the labels
- visualised 6 random images with their ground truth bounding boxes drawn on
- ground truth = the human drawn boxes, not model predictions

**step 3 - split the dataset**
- split 128 images into train / val / test (80 / 15 / 5)
- train = what the model learns from
- val = checked after each epoch to see if its improving
- test = touched only once at the end for a final honest score
- 2 images had no label file — these are "background images" (no objects in them), yolo handles them fine

**step 4 - analyse the dataset**
- looked at class distribution, objects per image, bounding box sizes
- person is by far the most common class (254 instances)
- avg 7.4 objects per image, max 42
- 49% of bounding boxes are smaller than 1% of the image area (lots of small objects)
- only 71 of the 80 classes actually appear in this 128-image subset

**step 5 - create training config**
- made dataset.yaml — the file yolo reads to find images, labels, and class names
- points to our split folder with train/val/test paths

**step 6 - pre-training sanity check**
- verified all images have matching label files (or are valid background images)
- checked every label line has the right format and values between 0-1
- did a dry run with the pretrained model on val set — mAP50 = 0.615
  (this doesnt mean anything yet, we havent trained on our data, just confirms it all runs)

all checks passed. ready to train

---

## key things i learned about the data

person dominates everything. 254 instances vs 46 for car (next highest). so the model will naturally get better at detecting people than rare classes like toothbrush or hair drier.

nearly half the objects are really small (less than 1% of image area). small objects are harder to detect — something to keep in mind when evaluating results later.

---

## folder structure

```
Object Detection Project/
  data/
    coco128/                    <- phase 1 raw download, untouched
      images/train2017/
      labels/train2017/
    split/                      <- phase 1 train/val/test split
      images/  train/ val/ test/
      labels/  train/ val/ test/
    coco_val2017/               <- phase 2 data
      val2017/                  <- 5000 raw jpg images
      annotations/              <- coco json annotations
      labels/                   <- converted yolo txt labels (one per image)
      split/                    <- phase 2 train/val/test split
        images/  train/ val/ test/
        labels/  train/ val/ test/
    dataset.yaml                <- active training config (points to current phase)
  scripts/
    run_all.sh                  <- runs the full pipeline in order
    explore_dataset.py          <- downloads coco128, shows ground truth labels
    split_dataset.py            <- splits coco128 into train/val/test
    analyse_dataset.py          <- class distribution, box sizes, object counts
    create_config.py            <- generates dataset.yaml
    pre_training_check.py       <- validates data before training
    train.py                    <- trains the model
    download_coco_val.py        <- phase 2: downloads coco val2017 + converts annotations
    split_coco_val.py           <- phase 2: splits 5000 images into train/val/test
  runs/
    coco128_ground_truth.png    <- 6 sample images with labels (phase 1)
    dataset_analysis.png        <- charts: class distribution, box sizes etc
    pre_training_check.png      <- one image from each split with labels drawn
    train1/                     <- phase 1 training results (yolo11n, coco128)
    train2/                     <- phase 2 training results (yolo11s, coco val2017)
```

---

## the label format

each .txt file has one line per object in the image. format is:

```
class_id  center_x  center_y  width  height
```

all coordinates are normalised between 0 and 1 so they work on any image size. example from 000000000009.txt:

```
45 0.479 0.689 0.956 0.596   <- a bowl, almost centered, takes up most of the frame
50 0.637 0.733 0.494 0.511   <- broccoli, bottom half of image
```

---

## dataset stats

| thing        | number         |
|--|--|
| total images | 128            |
| train        | 102            |
| val          | 19             |
| test         | 7              |
| classes      | 80 (71 present)|
| total objects| 929            |
| avg per image| 7.4            |

---

## what is COCO128

COCO128 is a mini version of the COCO dataset (Common Objects in Context). the full COCO dataset has 118,000 images and took months to collect and annotate by hand. COCO128 is just the first 128 images from it — small enough to train on a laptop, big enough to actually learn something.

it covers 80 different object categories — everyday stuff like people, cars, dogs, chairs, laptops, bananas, toothbrushes etc. every image has bounding boxes drawn around every object in it, done by humans. thats what we're training on.

we're training on 102 of those 128 images (the train split). the other 26 are held back for val and test.

---

## what training actually means

training is not magic. heres whats literally happening:

the model starts with weights from `yolo11n.pt` — a file of ~3 million numbers that were learned by training on all 118k COCO images. those weights already encode things like "edges look like this", "wheels are round", "faces have two eyes". we dont throw that away. we start from there.

then we do fine-tuning — we show the model our training images one batch at a time. for each batch:

1. the model looks at the image and guesses where the objects are
2. we compare its guesses to the ground truth labels (the human drawn boxes)
3. we calculate how wrong it was — thats the loss
4. we nudge the weights slightly in the direction that makes the loss smaller — thats backpropagation
5. repeat for every batch, then do it all again for the next epoch

an epoch = one full pass through all training images. we do 50 epochs so the model sees every image 50 times total, getting a little better each time.

after each epoch it runs on the val set (images it hasnt trained on) to check its actually improving and not just memorising the training images. this is called overfitting and its a real problem — more on that after we see results.

---

## the training command (phase 1)

```python
model = YOLO("yolo11n.pt")
model.train(
    data="data/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    project="runs",
    name="train1"
)
```

- `epochs=50` — 50 full passes through the training data
- `imgsz=640` — all images resized to 640x640 before going into the model
- `batch=16` — 16 images processed at a time before weights are updated (gpu allows this)
- results saved to `runs/train1/`

---

## how long will it take

running on CPU (programmed with the assumption of no GPU on this machine). estimate: 30-60 minutes for 50 epochs on 102 images.

if it had a GPU it would be more like 2-3 minutes. thats why people use GPUs for training — same math, just way more parallelism. for now CPU is fine, its only 102 images.

---

## shifting to GPU

I realised that the utilisation of my GPU in my pc would significantly speed up the training time. I have a Nvidia GeForce RTX 3060 12GB, which can be utilised for this very case. As well as training time it will also be able to work better with larger datasets, for example if more than 102 images were used (for the future).

the original pytorch install that came with ultralytics was the CPU-only version (`torch 2.12.0+cpu`). to switch to GPU we had to uninstall it and reinstall with CUDA support:

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

the `cu126` part matches the CUDA version (12.6) reported by `nvidia-smi` on this machine. you have to match these — installing cu118 on a machine with CUDA 12.6 drivers would still work, but cu126 gets the best performance.

after reinstalling, pytorch confirmed the GPU:

```
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
```

training time estimate updated from 30-60 minutes (CPU) to 2-3 minutes (GPU). batch size can also be increased from 8 to 16 since the 3060 has 12GB VRAM.

---

## training results

trained yolo11n for 50 epochs on 102 images using the RTX 3060. took roughly 2-3 minutes total on GPU.

**overall performance on val set (best.pt):**

| metric | before training | after training |
|--|--|--|
| mAP50 | 0.615 | 0.683 |
| mAP50-95 | 0.493 | 0.541 |
| Precision | 0.706 | 0.707 |
| Recall | 0.507 | 0.586 |

mAP50 went from 0.615 to 0.683 — a genuine improvement on our specific data split.

**per class breakdown (best performing):**

| class | mAP50 |
|--|--|
| motorcycle | 0.995 |
| bus | 0.995 |
| elephant | 0.995 |
| umbrella | 0.995 |
| sandwich | 0.995 |
| spoon | 0.995 |
| person | 0.773 |

**worst performing:**

| class | mAP50 | why |
|--|--|--|
| mouse | 0.000 | only 1 image in val, barely any training examples |
| banana | 0.124 | same problem |
| fork | 0.090 | same problem |

the poor performers arent really model failures — its a data problem. with only 19 val images, classes that appear once or twice cant be evaluated properly. the model has barely seen them during training either.

**saved weights:**
- `runs/train1/weights/best.pt` — best checkpoint across all epochs (what you use for inference)
- `runs/train1/weights/last.pt` — final epoch checkpoint

---

## whats next (phase 1 complete)

phase 1 is done. trained yolo11n on coco128, got mAP50 up from 0.615 to 0.683. now moving onto phase 2.

---

## phase 2 — bigger data, bigger model

### what we're changing and why

**model: yolo11n → yolo11s**

yolo11n (nano) has ~3 million parameters. yolo11s (small) has ~9 million. more parameters means the model can learn more complex patterns — better on small objects, better on rare classes, better at separating things that look similar. the gap shows most on harder cases, which is exactly where yolo11n struggled (fork: 0.090, banana: 0.124 etc)

yolo11s is 3x the capacity for maybe 2x the training time. worth it.

we considered yolo11m (~25M params) but that pushes training to 2-3hrs on this machine. saving that for phase 3 (custom objects) where accuracy matters most.

**dataset: COCO128 → COCO val 2017**

| thing | coco128 (phase 1) | coco val 2017 (phase 2) |
|--|--|--|
| images | 128 | 5,000 |
| training images | 102 | ~3,900 |
| val images | 19 | ~750 |
| test images | 7 | ~350 |
| classes | 80 (71 present) | 80 (all present) |
| total annotations | 929 | ~36,000 |
| download size | ~27MB | ~1GB |

40x more images. all 80 classes actually represented properly. the rare classes that had 1-2 examples in phase 1 will now have dozens or hundreds. that directly fixes the data scarcity problem.

### expected results

| metric | phase 1 (yolo11n, coco128) | phase 2 target (yolo11s, coco val 2017) |
|--|--|--|
| mAP50 | 0.683 | 0.72 - 0.78 |
| mAP50-95 | 0.541 | 0.55 - 0.62 |
| training time | ~3 min | ~45-90 min |
| worst class mAP50 | 0.000 (mouse) | should be above 0.3 for most |

the big wins will be on classes that were starved of data in phase 1. person/bus/motorcycle were already good — those wont change much. the bottom of the table should lift significantly.

### training plan

```python
model = YOLO("yolo11s.pt")
model.train(
    data="data/dataset.yaml",   # same format, updated paths
    epochs=50,
    imgsz=640,
    batch=8,                    # see vram issue below
    device=0,
    project="runs",
    name="train2"               # separate from phase 1 results
)
```

estimated time: **60-120 minutes** on RTX 3060 12GB

### vram issue — batch size reduced from 16 to 8

first training attempt used batch=16 (same as phase 1). crashed at the end of epoch 1 with:

```
ptxas fatal: Memory allocation failure
RuntimeError: bad allocation
```

training itself ran fine at 4.35GB VRAM. the crash happened during the validation pass after epoch 1. yolo11s has 3x more parameters than yolo11n — each image needs more memory to process during inference, and validation runs without the memory optimisations that training uses (no gradient checkpointing etc).

fix: dropped batch from 16 to 8. this halves the number of images processed simultaneously, halving the peak VRAM requirement during validation. accuracy impact is negligible — batch=8 vs batch=16 on a 4000 image dataset makes no meaningful difference to final mAP.

---

## phase 2 training results

trained yolo11s for 50 epochs on 4,000 images using the RTX 3060. took roughly 90 minutes total (including the crash and resume — see vram issue above).

**overall performance on val set (best.pt):**

| metric | phase 1 (yolo11n, 102 imgs) | phase 2 (yolo11s, 4000 imgs) |
|--|--|--|
| mAP50 | 0.683 | 0.627 |
| mAP50-95 | 0.541 | 0.461 |
| Precision | 0.707 | 0.648 |
| Recall | 0.586 | 0.585 |

the numbers look lower than phase 1 but this is misleading. phase 1 was evaluated on 19 images — a tiny, easy val set. phase 2 is evaluated on 750 diverse images from across the full coco dataset. its a much harder benchmark. the model is genuinely better, the scoring just got harder.

**best performing classes:**

| class | mAP50 |
|--|--|
| airplane | 0.955 |
| toaster | 0.995 |
| bus | 0.888 |
| motorcycle | 0.856 |
| microwave | 0.917 |
| teddy bear | 0.839 |

**worst performing:**

| class | mAP50 | why |
|--|--|--|
| scissors | 0.027 | only 3 val images, visually hard to detect |
| book | 0.257 | 182 instances but heavily occluded/stacked |
| carrot | 0.263 | small object, often partially visible |

**the big win — classes that were broken in phase 1:**

mouse went from 0.000 → 0.784. fork, banana and other starved classes now have real scores. this was the whole point of the bigger dataset — classes that had 1-2 training examples in phase 1 now have dozens.

**saved weights:**
- `runs/train2/weights/best.pt` — best checkpoint (19.2MB, use this for inference)
- `runs/train2/weights/last.pt` — epoch 50 checkpoint

---

## phase 3 — custom object (planned)

after phase 2 we will train on a custom object class defined by us. this means:
- collecting our own images
- annotating them with bounding boxes (using roboflow)
- training yolo11m or yolo11s on our custom data

details tbd after phase 2 is complete.

---

## how to run (in order)

```bash
cd scripts
bash run_all.sh
```

or individually by phase:

**phase 1 (coco128, yolo11n):**
```bash
cd scripts
python explore_dataset.py
python split_dataset.py
python analyse_dataset.py
python create_config.py
python pre_training_check.py
python train.py
```

**phase 2 (coco val2017, yolo11s):**
```bash
cd scripts
python download_coco_val.py   # ~1.2gb download
python split_coco_val.py
python create_config.py
python pre_training_check.py
python train.py               # 45-90 min on rtx 3060
```

---

dependencies: ultralytics, opencv-python, matplotlib, pyyaml
