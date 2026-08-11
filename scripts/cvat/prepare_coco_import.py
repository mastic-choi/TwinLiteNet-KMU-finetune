#!/usr/bin/env python3
"""raw/da_masks_draft + raw/ll_masks_draft를 COCO instance segmentation JSON으로
변환한다. CVAT의 'Segmentation mask' import는 픽셀 계단을 그대로 폴리곤 점으로
따라가서 점이 지나치게 많아지는데, 여기서는 cv2.findContours + approxPolyDP로
미리 단순화한 폴리곤 좌표를 직접 만들어 넣어서 훨씬 적은 점으로 같은 모양을
표현한다. COCO는 배경을 명시할 필요가 없어서 'background' 라벨도 필요 없다.

CVAT에서: Task 열기 -> Actions -> Upload annotations -> 포맷 'COCO 1.0' ->
이 json 파일 업로드 (이미지는 이미 task에 있으니 json만 올리면 됨).
"""
import glob
import json
import os

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGES_DIR = os.path.join(BASE, "raw", "images_todo")
DA_DIR = os.path.join(BASE, "raw", "da_masks_draft")
LL_DIR = os.path.join(BASE, "raw", "ll_masks_draft")
OUT_JSON = os.path.join(BASE, "coco_import_polygons.json")

CAT_DA = 1  # drivable_area
CAT_LL = 2  # lane_line

MIN_CONTOUR_AREA = 15       # 이보다 작은 조각은 노이즈로 버림
EPS_RATIO = 0.005           # approxPolyDP epsilon = 이 비율 * 컨투어 둘레 (작을수록 원형 유지, 클수록 더 단순화)
MIN_EPS_PX = 1.2


def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA:
            continue
        peri = cv2.arcLength(c, True)
        eps = max(MIN_EPS_PX, EPS_RATIO * peri)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2).astype(np.float64)
        seg = pts.flatten().tolist()
        x, y, w, h = cv2.boundingRect(approx)
        polys.append({"segmentation": [seg], "area": float(area), "bbox": [float(x), float(y), float(w), float(h)]})
    return polys


def main():
    files = sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith(".png"))
    assert files, f"{IMAGES_DIR}에 png 없음"
    print(f"{len(files)}장 COCO 폴리곤 변환 시작 (epsilon 비율={EPS_RATIO}, 최소 {MIN_EPS_PX}px)")

    images = []
    annotations = []
    ann_id = 1
    total_pts_before = 0
    total_pts_after = 0

    for img_id, f in enumerate(files, start=1):
        img_path = os.path.join(IMAGES_DIR, f)
        h, w = cv2.imread(img_path).shape[:2]
        images.append({"id": img_id, "file_name": f, "width": w, "height": h})

        da = cv2.imread(os.path.join(DA_DIR, f), cv2.IMREAD_GRAYSCALE)
        ll = cv2.imread(os.path.join(LL_DIR, f), cv2.IMREAD_GRAYSCALE)

        # 단순화 전(원본 컨투어) 점 개수도 세서 얼마나 줄었는지 확인용
        raw_da_contours, _ = cv2.findContours((da > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_ll_contours, _ = cv2.findContours((ll > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_pts_before += sum(len(c) for c in raw_da_contours) + sum(len(c) for c in raw_ll_contours)

        for cat_id, mask in [(CAT_DA, da), (CAT_LL, ll)]:
            for poly in mask_to_polygons((mask > 0).astype(np.uint8)):
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": cat_id,
                    "iscrowd": 0, **poly,
                })
                total_pts_after += len(poly["segmentation"][0]) // 2
                ann_id += 1

        if img_id % 100 == 0:
            print(f"  {img_id}/{len(files)}")

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": CAT_DA, "name": "drivable_area"},
            {"id": CAT_LL, "name": "lane_line"},
        ],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(coco, f)

    print(f"\n완료: {OUT_JSON}")
    print(f"객체(폴리곤) {len(annotations)}개, 이미지 {len(images)}장")
    print(f"점 개수: 원래 컨투어 기준 {total_pts_before}개 -> 단순화 후 {total_pts_after}개 "
          f"({total_pts_after / max(total_pts_before,1):.1%})")


if __name__ == "__main__":
    main()
