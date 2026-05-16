1"""
Phase 2 — Train YOLO11s on COCO val2017

fine-tunes yolo11s on 4,000 images for 50 epochs.
upgraded from yolo11n (phase 1) — 3x more parameters, noticeably better
on small objects and rare classes that were starved of data in phase 1.

results saved to runs/train2/

NOTE: the if __name__ == '__main__' guard is required on Windows.
pytorch spawns worker processes that re-import this file — without the guard
each worker would try to start training again, causing a crash.
"""

from ultralytics import YOLO
from pathlib import Path

if __name__ == "__main__":
    CONFIG  = Path(__file__).parent.parent / "data" / "dataset.yaml"
    OUT_DIR = Path(__file__).parent.parent / "runs"

    model = YOLO("yolo11s.pt")

    model.train(
        data=str(CONFIG),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,        # 0 = first GPU (RTX 3060)
        project=str(OUT_DIR),
        name="train2",
        exist_ok=True,
    )

    print("\nTraining complete. Results saved to runs/train2/")
