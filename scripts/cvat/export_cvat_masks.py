#!/usr/bin/env python3
"""팀원들이 CVAT에서 보정을 끝낸 뒤 'Segmentation mask 1.1' 포맷으로 export한 zip을
다시 raw/images, raw/da_masks, raw/ll_masks(노트북이 기대하는 최종 구조)로 되돌린다.

사용법:
  python export_cvat_masks.py /path/to/cvat_export.zip

prepare_cvat_import.py와 같은 색 규칙을 그대로 씀:
  drivable_area(초록,0,128,0) 또는 lane_line(빨강,128,0,0) 픽셀 -> da_mask=255
  lane_line(빨강,128,0,0) 픽셀만 -> ll_mask=255
(차선은 도로 위에 그려진 것이므로 lane_line 픽셀도 drivable_area에 포함시킴)
"""
import glob
import os
import shutil
import sys
import zipfile

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_IMAGES_TODO = os.path.join(BASE, "raw", "images_todo")
OUT_IMAGES = os.path.join(BASE, "raw", "images")
OUT_DA = os.path.join(BASE, "raw", "da_masks")
OUT_LL = os.path.join(BASE, "raw", "ll_masks")
EXTRACT_DIR = os.path.join(BASE, "cvat_export_extracted")

DA_COLOR_RGB = (0, 128, 0)
LL_COLOR_RGB = (128, 0, 0)


def main():
    if len(sys.argv) != 2:
        print("사용법: python export_cvat_masks.py /path/to/cvat_export.zip")
        sys.exit(1)
    zip_path = sys.argv[1]

    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    os.makedirs(EXTRACT_DIR)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(EXTRACT_DIR)

    seg_dir = os.path.join(EXTRACT_DIR, "SegmentationClass")
    assert os.path.isdir(seg_dir), f"{seg_dir} 없음 — 'Segmentation mask 1.1' 포맷으로 export했는지 확인"

    for d in (OUT_IMAGES, OUT_DA, OUT_LL):
        os.makedirs(d, exist_ok=True)

    files = sorted(glob.glob(os.path.join(seg_dir, "*.png")))
    print(f"{len(files)}장 변환")

    n_missing_orig = 0
    for fp in files:
        base = os.path.basename(fp)
        seg = cv2.imread(fp)  # BGR
        seg_rgb = seg[:, :, ::-1]

        da_pixels = np.all(seg_rgb == DA_COLOR_RGB, axis=-1)
        ll_pixels = np.all(seg_rgb == LL_COLOR_RGB, axis=-1)
        da_mask = (da_pixels | ll_pixels).astype(np.uint8) * 255
        ll_mask = ll_pixels.astype(np.uint8) * 255

        cv2.imwrite(os.path.join(OUT_DA, base), da_mask)
        cv2.imwrite(os.path.join(OUT_LL, base), ll_mask)

        orig_fp = os.path.join(RAW_IMAGES_TODO, base)
        if os.path.isfile(orig_fp):
            shutil.copy(orig_fp, os.path.join(OUT_IMAGES, base))
        else:
            n_missing_orig += 1

    if n_missing_orig:
        print(f"⚠ 원본 이미지를 raw/images_todo/에서 못 찾은 파일 {n_missing_orig}장 — raw/images/에 수동으로 채워넣을 것")

    print("완료:", OUT_IMAGES, "/", OUT_DA, "/", OUT_LL)


if __name__ == "__main__":
    main()
