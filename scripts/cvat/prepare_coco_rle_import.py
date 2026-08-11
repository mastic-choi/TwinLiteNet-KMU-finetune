#!/usr/bin/env python3
"""raw/da_masks_draft + raw/ll_masks_draft를 COCO RLE(iscrowd=1) 마스크로
변환한다. 폴리곤(iscrowd=0)은 CVAT이 점(vertex) 목록으로 가져와서 픽셀 계단을
그대로 따라가거나 단순화 과정에서 모양이 틀어질 수 있는데, RLE는 픽셀 자체를
그대로 인코딩하는 방식이라 CVAT이 폴리곤이 아니라 네이티브 'Mask'(브러시로
편집하는 비트맵) 객체로 가져온다 — 우리 PNG 마스크와 픽셀 단위로 완전히
동일하고, 점 관련 문제 자체가 없다.

이미지 하나당 da_mask 객체 1개 + ll_mask 객체 1개, 총 2개만 생김(폴리곤 버전은
차선 조각마다 여러 개였음) — Objects 패널도 훨씬 단순해짐.

CVAT에서: Task -> Actions -> Upload annotations -> 포맷 'COCO 1.0' -> 이 json 업로드.
"""
import os

import cv2
import numpy as np
import pycocotools.mask as mask_util
import json

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGES_DIR = os.path.join(BASE, "raw", "images_todo")
DA_DIR = os.path.join(BASE, "raw", "da_masks_draft")
LL_DIR = os.path.join(BASE, "raw", "ll_masks_draft")
OUT_JSON = os.path.join(BASE, "coco_import_rle.json")

CAT_DA = 1  # drivable_area
CAT_LL = 2  # lane_line


def encode_rle(binary_mask):
    fortran = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_util.encode(fortran)
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def main():
    files = sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith(".png"))
    assert files, f"{IMAGES_DIR}에 png 없음"
    print(f"{len(files)}장 COCO RLE 마스크 변환 시작")

    images = []
    annotations = []
    ann_id = 1

    for img_id, f in enumerate(files, start=1):
        img_path = os.path.join(IMAGES_DIR, f)
        h, w = cv2.imread(img_path).shape[:2]
        images.append({"id": img_id, "file_name": f, "width": w, "height": h})

        da = cv2.imread(os.path.join(DA_DIR, f), cv2.IMREAD_GRAYSCALE)
        ll = cv2.imread(os.path.join(LL_DIR, f), cv2.IMREAD_GRAYSCALE)

        for cat_id, mask in [(CAT_DA, da), (CAT_LL, ll)]:
            binary = (mask > 0).astype(np.uint8)
            if binary.sum() == 0:
                continue  # 빈 마스크는 객체 자체를 안 만듦(CVAT에서 빈 프레임으로 보임)
            rle = encode_rle(binary)
            area = float(mask_util.area(rle))
            x, y, bw, bh = mask_util.toBbox(rle).tolist()
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cat_id,
                "segmentation": {"counts": rle["counts"], "size": rle["size"]},
                "area": area, "bbox": [x, y, bw, bh], "iscrowd": 1,
            })
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
    print(f"객체(마스크) {len(annotations)}개, 이미지 {len(images)}장 (이미지당 최대 2개: da+ll)")


if __name__ == "__main__":
    main()
