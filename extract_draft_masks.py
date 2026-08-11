#!/usr/bin/env python3
"""best.onnx를 다시 돌리지 않고, 이미 만들어둔 dataset_overlay/의 색상
정보에서 da/ll 이진 마스크를 역산해서 노트북 cell-10과 같은 형식의
초안(draft) 마스크를 만든다.

make_overlay_dataset.py가 그린 방식(정확히 아는 고정 공식):
  overlay[da_mask] = floor(0.5*원본 + 0.5*(255,100,0))   (BGR, da만 참일 때)
  overlay[ll_mask] = (0,0,255) 순수 빨강                  (da 위에 덮어씀, ll이 참일 때 최종 색)

따라서:
  ll_mask = overlay가 정확히 (0,0,255)인 픽셀
  da_mask = overlay가 위 블렌드 공식과 정확히 일치하는 픽셀, 그리고 ll_mask도 합쳐준다
            (차선은 도로 위에 그려진 것이므로 물리적으로도 da에 포함되는 게 맞음)

dataset_diet/에 있는 470장만 대상으로 raw/images_todo, raw/da_masks_draft,
raw/ll_masks_draft를 만든다 — 이후 노트북 2번 섹션(사람이 보정)부터 이어가면 됨.
"""
import glob
import os

import cv2
import numpy as np

BASE = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE, "dataset")
OVERLAY_DIR = os.path.join(BASE, "dataset_overlay")
DIET_DIR = os.path.join(BASE, "dataset_diet")

OUT_IMAGES = os.path.join(BASE, "raw", "images_todo")
OUT_DA = os.path.join(BASE, "raw", "da_masks_draft")
OUT_LL = os.path.join(BASE, "raw", "ll_masks_draft")

DA_TINT = np.array([255, 100, 0], dtype=np.float64)  # BGR
LL_RED = np.array([0, 0, 255], dtype=np.uint8)


def main():
    for d in (OUT_IMAGES, OUT_DA, OUT_LL):
        os.makedirs(d, exist_ok=True)

    files = sorted(f for f in os.listdir(DIET_DIR) if f.endswith(".png"))
    assert files, f"{DIET_DIR}에 png 없음"
    print(f"{len(files)}장 마스크 역산 시작")

    for i, f in enumerate(files):
        orig = cv2.imread(os.path.join(RAW_DIR, f))
        overlay = cv2.imread(os.path.join(OVERLAY_DIR, f))
        assert orig is not None and overlay is not None, f"{f} 원본/오버레이 로드 실패"

        ll_mask = np.all(overlay == LL_RED, axis=-1)

        expected_blend = (0.5 * orig.astype(np.float64) + 0.5 * DA_TINT).astype(np.uint8)
        da_mask = np.all(overlay == expected_blend, axis=-1)
        da_mask = da_mask | ll_mask  # 차선 밑도 물리적으로는 도로(da)

        cv2.imwrite(os.path.join(OUT_IMAGES, f), orig)
        cv2.imwrite(os.path.join(OUT_DA, f), (da_mask.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(OUT_LL, f), (ll_mask.astype(np.uint8) * 255))

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)}")

    print("\n완료")
    print("원본 이미지:", OUT_IMAGES)
    print("da 초안 마스크:", OUT_DA)
    print("ll 초안 마스크:", OUT_LL)
    print("-> Drive의 raw/images_todo, raw/da_masks_draft, raw/ll_masks_draft에 그대로 업로드하면 됨")


if __name__ == "__main__":
    main()
