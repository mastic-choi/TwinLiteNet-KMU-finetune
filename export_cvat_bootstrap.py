#!/usr/bin/env python3
"""CVAT에서 내려받은 'COCO 1.0' annotation zip + Jobs 목록 CSV를 가지고,
Job State가 'completed'인 프레임만 골라서 부트스트랩(미니 파인튜닝)용
raw/images, raw/da_masks, raw/ll_masks를 만든다.

State가 'completed'가 아닌 job의 프레임은 아직 사람이 검증 안 한 원래 초안
그대로일 수 있어서(=우리 모델이 원래 틀리던 부분이 '정답'으로 둔갑) 반드시
제외한다.

사용법:
  python export_cvat_bootstrap.py "<coco_export>.zip" "<jobs>.csv"
"""
import csv as csv_mod
import json
import os
import shutil
import sys
import zipfile

import cv2
import numpy as np
import pycocotools.mask as mask_util

BASE = os.path.dirname(__file__)
RAW_IMAGES_TODO = os.path.join(BASE, "raw", "images_todo")
OUT_DIR = os.path.join(BASE, "bootstrap")
OUT_IMAGES = os.path.join(OUT_DIR, "images")
OUT_DA = os.path.join(OUT_DIR, "da_masks")
OUT_LL = os.path.join(OUT_DIR, "ll_masks")


def load_completed_frame_indices(csv_path):
    completed = set()
    with open(csv_path, newline="") as f:
        for row in csv_mod.DictReader(f):
            if row["State"].strip().lower() == "completed":
                start, stop = int(row["Start Frame"]), int(row["Stop Frame"])
                completed.update(range(start, stop + 1))
    return completed


def decode_segmentation(seg, h, w):
    rle = mask_util.frPyObjects(seg, h, w)
    return mask_util.decode(rle).astype(bool)


def main():
    if len(sys.argv) != 3:
        print('사용법: python export_cvat_bootstrap.py "<coco_export>.zip" "<jobs>.csv"')
        sys.exit(1)
    zip_path, csv_path = sys.argv[1], sys.argv[2]

    completed_frames = load_completed_frame_indices(csv_path)
    print(f"completed 프레임(0-indexed) {len(completed_frames)}개")

    extract_dir = os.path.join(BASE, "_cvat_export_extracted")
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    json_path = None
    for root, _, files in os.walk(extract_dir):
        for fn in files:
            if fn.endswith(".json"):
                json_path = os.path.join(root, fn)
    assert json_path, "annotations json을 zip 안에서 못 찾음"

    coco = json.load(open(json_path))
    cat_name_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    images_by_id = {im["id"]: im for im in coco["images"]}
    # COCO image id는 1부터, CVAT의 0-indexed frame index는 id-1과 같음
    # (task 생성 시 Sorting=Lexicographical로 올렸고, 로컬 raw/images_todo도 같은 정렬이라 일치함)

    anns_by_image = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    for d in (OUT_IMAGES, OUT_DA, OUT_LL):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    n_done = 0
    for img_id, img_info in images_by_id.items():
        frame_idx = img_id - 1
        if frame_idx not in completed_frames:
            continue

        f = img_info["file_name"]
        h, w = img_info["height"], img_info["width"]
        da_mask = np.zeros((h, w), dtype=bool)
        ll_mask = np.zeros((h, w), dtype=bool)

        for a in anns_by_image.get(img_id, []):
            name = cat_name_by_id[a["category_id"]]
            if name not in ("drivable_area", "lane_line"):
                continue
            m = decode_segmentation(a["segmentation"], h, w)
            if name == "drivable_area":
                da_mask |= m
            else:
                ll_mask |= m
        da_mask |= ll_mask  # 차선은 도로 위에 그려진 것 -> da에도 포함

        src_img = os.path.join(RAW_IMAGES_TODO, f)
        assert os.path.isfile(src_img), f"{src_img} 없음"
        shutil.copy(src_img, os.path.join(OUT_IMAGES, f))
        cv2.imwrite(os.path.join(OUT_DA, f), (da_mask.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(OUT_LL, f), (ll_mask.astype(np.uint8) * 255))
        n_done += 1

    print(f"완료: {n_done}장 (completed job에 속한 프레임만)")
    print("저장 위치:", OUT_DIR)


if __name__ == "__main__":
    main()
