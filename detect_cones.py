#!/usr/bin/env python3
"""dataset/ 전체 프레임에 cone_best.pt(yolo_ros가 실제 쓰는 것과 동일 threshold/imgsz)를
돌려서 프레임당 라바콘 검출 개수/최대 신뢰도를 CSV로 남긴다."""
import csv
import glob
import os

from ultralytics import YOLO

BASE = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE, "dataset")
MODEL_PATH = os.path.expanduser("~/code/xycar_ws/src/yolo_ros/cone_best.pt")
OUT_CSV = os.path.join(BASE, "cone_detect_result.csv")

CONF_THRESHOLD = 0.5   # yolo_node.py 기본 threshold 파라미터와 동일
IMGSZ = 640            # yolo_node.py 기본 imgsz_height/width와 동일


def main():
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.png")))
    assert files, f"{DATASET_DIR}에 png 없음"
    print(f"{len(files)}장 콘 검출 시작 — 모델: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    rows = []
    for i, fp in enumerate(files):
        result = model.predict(fp, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)[0]
        confs = result.boxes.conf.tolist() if result.boxes is not None else []
        rows.append({
            "file": os.path.basename(fp),
            "n_cones": len(confs),
            "max_conf": round(max(confs), 4) if confs else 0.0,
            "mean_conf": round(sum(confs) / len(confs), 4) if confs else 0.0,
        })
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_cone = sum(1 for r in rows if r["n_cones"] > 0)
    print(f"\n총 {len(rows)}장 중 라바콘 검출된 프레임: {n_cone}장 ({n_cone / len(rows):.1%})")
    print("CSV:", OUT_CSV)


if __name__ == "__main__":
    main()
