#!/usr/bin/env python3
"""skeleton화 전/후 ll 마스크 비교 몽타주.
파랑=원래 브러시 마스크(두께 들쭉날쭉), 빨강=skeleton+고정두께 재래스터화,
보라(겹침)=두 결과가 겹치는 영역."""
import math
import os

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "bootstrap")
REFINED = os.path.join(BASE, "bootstrap_refined")
OUT_PATH = os.path.join(BASE, "ll_refine_montage.png")
THUMB_W, THUMB_H = 320, 240


def main():
    files = sorted(f for f in os.listdir(os.path.join(SRC, "ll_masks")) if f.endswith(".png"))
    thumbs = []
    for fname in files:
        img = cv2.imread(os.path.join(SRC, "images", fname))
        old_ll = cv2.imread(os.path.join(SRC, "ll_masks", fname), cv2.IMREAD_GRAYSCALE) > 0
        new_ll = cv2.imread(os.path.join(REFINED, "ll_masks", fname), cv2.IMREAD_GRAYSCALE) > 0

        overlay = img.copy()
        overlay[old_ll] = (255, 80, 0)
        overlay[new_ll] = (0, 0, 255)
        both = old_ll & new_ll
        overlay[both] = (255, 0, 255)

        thumb = cv2.resize(overlay, (THUMB_W, THUMB_H))
        cv2.rectangle(thumb, (0, 0), (THUMB_W, 16), (0, 0, 0), -1)
        cv2.putText(thumb, fname, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)

    cols = 6
    rows_n = math.ceil(len(thumbs) / cols)
    grid = np.zeros((rows_n * THUMB_H, cols * THUMB_W, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * THUMB_H:(r + 1) * THUMB_H, c * THUMB_W:(c + 1) * THUMB_W] = t
    cv2.imwrite(OUT_PATH, grid)
    print(f"{len(thumbs)}장 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
