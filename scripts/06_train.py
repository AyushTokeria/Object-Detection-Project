1"""
Step 6 — Train YOLO11n on COCO128

this is the actual training run. everything before this was preparation.
the model will fine-tune on our 102 training images for 50 epochs,
checking against the 19 val images after each one.

results saved to runs/train1/

NOTE: the if __name__ == '__main__' guard is required on Windows.
pytorch spawns worker processes that re-import this file — without the guard
each worker would try to start training again, causing a crash.
"""

from ultralytics import YOLO
from pathlib import Path

if __name__ == "__main__":
    CONFIG  = Path(__file__).parent.parent / "data" / "dataset.yaml"
    OUT_DIR = Path(__file__).parent.parent / "runs"

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(CONFIG),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,        # 0 = first GPU (RTX 3060)
        project=str(OUT_DIR),
        name="train1",
        exist_ok=True,
    )

    print("\nTraining complete. Results saved to runs/train1/")
