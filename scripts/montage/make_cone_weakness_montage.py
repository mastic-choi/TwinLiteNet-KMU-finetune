#!/usr/bin/env python3
"""cone_best.pt(YOLOv8, 라바콘 검출기)가 약한 두 가지 케이스를 골라 몽타주로 확인한다.
cone_detect_result.csv(전체 2123장, detect_cones.py 결과)만으로는 바운딩박스가 없어서
아래 프레임들만 다시 추론해서 박스/신뢰도를 그림.

1. 저신뢰도 검출(0.5~0.65, threshold=0.5 턱걸이) - 있긴 한데 애매하게 잡힌 프레임
2. "깜빡임" 미검출 - 바로 앞뒤 프레임엔 콘이 잡히는데 이 프레임만 0으로 빠짐(진짜 미검출 신호)
"""
import math
import os

import cv2
import numpy as np
from ultralytics import YOLO

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE, "dataset")
MODEL_PATH = os.path.expanduser("~/code/xycar_ws/src/yolo_ros/cone_best.pt")
OUT_PATH = os.path.join(BASE, "outputs", "montages", "cone_weakness_montage.png")
CONF_THRESHOLD = 0.5
IMGSZ = 640
THUMB_W, THUMB_H = 320, 240

WEAK_FRAMES = [
    'frame_000858.png', 'frame_001977.png', 'frame_001674.png', 'frame_000439.png',
    'frame_000940.png', 'frame_001013.png', 'frame_001951.png', 'frame_001312.png',
]
FLICKER_FRAMES = [
    'frame_000187.png', 'frame_000441.png', 'frame_000628.png', 'frame_000722.png',
    'frame_001001.png', 'frame_001011.png', 'frame_001026.png', 'frame_001040.png',
]

FRAMES = [(f, "weak(0.5~0.65)") for f in WEAK_FRAMES] + [(f, "flicker-miss") for f in FLICKER_FRAMES]


def main():
    model = YOLO(MODEL_PATH)
    thumbs = []
    for fname, kind in FRAMES:
        fp = os.path.join(DATASET_DIR, fname)
        result = model.predict(fp, conf=0.2, imgsz=IMGSZ, verbose=False)[0]  # threshold 낮춰서 "거의 잡힐 뻔한" 것도 보이게
        bgr = cv2.imread(fp)

        boxes = result.boxes
        n_above_thresh = 0
        if boxes is not None:
            for b in boxes:
                conf = float(b.conf[0])
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                color = (0, 255, 0) if conf >= CONF_THRESHOLD else (0, 165, 255)  # 초록=정식 검출, 주황=threshold 미달
                cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(bgr, f"{conf:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if conf >= CONF_THRESHOLD:
                    n_above_thresh += 1

        thumb = cv2.resize(bgr, (THUMB_W, THUMB_H))
        label = f"[{kind}] {fname} n>=0.5:{n_above_thresh}"
        cv2.rectangle(thumb, (0, 0), (THUMB_W, 18), (0, 0, 0), -1)
        cv2.putText(thumb, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
        print(f"[{kind}] {fname}: boxes(conf>=0.2)={len(boxes) if boxes is not None else 0}, >=0.5={n_above_thresh}")

    cols = 4
    rows_n = math.ceil(len(thumbs) / cols)
    grid = np.zeros((rows_n * THUMB_H, cols * THUMB_W, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * THUMB_H:(r + 1) * THUMB_H, c * THUMB_W:(c + 1) * THUMB_W] = t
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, grid)
    print(f"\n{len(thumbs)}장 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
