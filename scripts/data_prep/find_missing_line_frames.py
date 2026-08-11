#!/usr/bin/env python3
"""dataset_overlay/ 이미지에서 노란 ROI 박스(y=250~390) 안을 좌/중앙/우 3구간으로
나눠, ll 예측(순수 빨강 0,0,255 BGR) 픽셀이 각 구간에 충분히 있는지 검사한다.
3구간 중 하나라도 "안 잡힘"이면 missing_line_frames/에 복사한다 —
예: 좌우 차선은 잡히는데 중앙 노란 점선만 안 잡히는 케이스 탐지용.
"""
import csv
import glob
import os
import shutil

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OVERLAY_DIR = os.path.join(BASE, "dataset_overlay")
OUT_DIR = os.path.join(BASE, "missing_line_frames")
OUT_CSV = os.path.join(BASE, "missing_line_result.csv")

DL_ROI_Y0, DL_ROI_Y1 = 250, 390
RED_BGR = np.array([0, 0, 255], dtype=np.uint8)
MIN_PIXELS_PER_BAND = 15  # config.py DL_LL_SIDE_MIN_PIXELS와 동일 기준(밴드 내 ll 픽셀 최소치)


def main():
    files = sorted(glob.glob(os.path.join(OVERLAY_DIR, "*.png")))
    assert files, f"{OVERLAY_DIR}에 png 없음 — make_overlay_dataset.py 먼저 실행할 것"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{len(files)}장 검사 시작")

    rows = []
    n_missing = 0
    for i, fp in enumerate(files):
        img = cv2.imread(fp)
        h, w = img.shape[:2]
        roi = img[DL_ROI_Y0:DL_ROI_Y1]
        red_mask = np.all(roi == RED_BGR, axis=-1)

        band_w = w // 3
        left = red_mask[:, 0:band_w]
        center = red_mask[:, band_w:2 * band_w]
        right = red_mask[:, 2 * band_w:w]

        left_px = int(np.count_nonzero(left))
        center_px = int(np.count_nonzero(center))
        right_px = int(np.count_nonzero(right))

        left_ok = left_px >= MIN_PIXELS_PER_BAND
        center_ok = center_px >= MIN_PIXELS_PER_BAND
        right_ok = right_px >= MIN_PIXELS_PER_BAND
        n_detected = sum([left_ok, center_ok, right_ok])

        rows.append({
            "file": os.path.basename(fp),
            "left_px": left_px, "center_px": center_px, "right_px": right_px,
            "left_ok": left_ok, "center_ok": center_ok, "right_ok": right_ok,
            "n_detected": n_detected,
        })

        if n_detected < 3:
            shutil.copy(fp, os.path.join(OUT_DIR, os.path.basename(fp)))
            n_missing += 1

        if (i + 1) % 400 == 0:
            print(f"  {i + 1}/{len(files)}")

    rows.sort(key=lambda r: r["n_detected"])
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n총 {len(files)}장 중 3구간(좌/중앙/우) 다 안 잡힌 프레임: {n_missing}장 ({n_missing / len(files):.1%})")
    print("복사 위치:", OUT_DIR)
    print("CSV:", OUT_CSV)


if __name__ == "__main__":
    main()
