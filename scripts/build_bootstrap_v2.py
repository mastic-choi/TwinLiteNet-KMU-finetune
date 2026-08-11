#!/usr/bin/env python3
"""bootstrap_v2 생성: 기존 74장(사람 da+ll, bootstrap/) + 신규 60장(사람 da만,
Downloads/da_coco.json + labeling_handoff.zip) = 134장.

신규 60장은 labeling_handoff.zip(§2.10, 실패후보/콘/커브 100장 중 커브 위주)의
일부 - da는 사람이 직접 그린 폴리곤, ll은 사람이 안 그려서 YOLOPv2+skeleton 정제로
자동 생성(§2.9와 동일 방식). da_final = da | ll (기존 관례 유지).

목적: 이 134장은 pseudo label 오염이 전혀 없는 "깨끗한" da 소스라, 이걸로 먼저
small 부트스트랩 학습 -> 그 모델로 나머지(원래 470장 중 자동생성 396장)의 da
pseudo label을 재생성해서 순환 오염(§2.12) 문제를 끊는다.
"""
import json
import os
import shutil
import sys
import zipfile

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "YOLOPv2"))
from utils.utils import letterbox as yolopv2_letterbox  # noqa: E402
sys.path.insert(0, os.path.join(BASE, "scripts"))
from skeleton_polyline_utils import mask_to_polylines, polylines_to_mask  # noqa: E402

DA_COCO_JSON = os.path.expanduser("~/Downloads/da_coco.json")
HANDOFF_ZIP = os.path.join(BASE, "labeling_handoff.zip")
EXISTING_BOOTSTRAP = os.path.join(BASE, "bootstrap")
YOLOPV2_WEIGHTS = os.path.join(BASE, "YOLOPv2", "data", "weights", "yolopv2.pt")

OUT_DIR = os.path.join(BASE, "bootstrap_v2")
OUT_IMAGES = os.path.join(OUT_DIR, "images")
OUT_DA = os.path.join(OUT_DIR, "da_masks")
OUT_LL = os.path.join(OUT_DIR, "ll_masks")

LL_REFINE_WIDTH = 8
EXTRACT_DIR = os.path.join(BASE, "_handoff_extracted")


def decode_da(seg, h, w):
    rle = mask_util.frPyObjects(seg, h, w)
    m = mask_util.decode(rle)
    if m.ndim == 3:
        m = m.any(axis=2)
    return m.astype(bool)


def infer_ll_yolopv2(model, bgr):
    h0, w0 = bgr.shape[:2]
    img, _, _ = yolopv2_letterbox(bgr, 640, stride=32)
    inp = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    inp_t = torch.from_numpy(inp).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        [pred, anchor_grid], seg, ll = model(inp_t)
    ll_mask = torch.round(ll)[0, 0].numpy().astype(np.uint8)
    return cv2.resize(ll_mask, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)


def refine_ll(ll_mask):
    h, w = ll_mask.shape
    polylines = mask_to_polylines(ll_mask)
    if not polylines:
        return ll_mask, 0
    refined = polylines_to_mask(polylines, h, w, line_width=LL_REFINE_WIDTH).astype(bool)
    return refined, len(polylines)


def main():
    assert os.path.isfile(DA_COCO_JSON), f"{DA_COCO_JSON} 없음"
    assert os.path.isfile(HANDOFF_ZIP), f"{HANDOFF_ZIP} 없음"

    for d in (OUT_IMAGES, OUT_DA, OUT_LL):
        os.makedirs(d, exist_ok=True)

    # 1. 기존 74장 그대로 복사
    n_existing = 0
    for fn in sorted(os.listdir(os.path.join(EXISTING_BOOTSTRAP, "images"))):
        for sub, out in (("images", OUT_IMAGES), ("da_masks", OUT_DA), ("ll_masks", OUT_LL)):
            shutil.copy(os.path.join(EXISTING_BOOTSTRAP, sub, fn), os.path.join(out, fn))
        n_existing += 1
    print(f"기존 bootstrap {n_existing}장 복사 완료")

    # 2. handoff zip에서 이미지 추출
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    with zipfile.ZipFile(HANDOFF_ZIP) as zf:
        zf.extractall(EXTRACT_DIR)
    handoff_images_dir = os.path.join(EXTRACT_DIR, "labeling_handoff", "images")
    assert os.path.isdir(handoff_images_dir), f"{handoff_images_dir} 없음 (zip 구조 확인 필요)"

    # 3. da_coco.json 파싱 - annotation 있는 프레임만
    with open(DA_COCO_JSON) as f:
        coco = json.load(f)
    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    yolopv2_model = torch.jit.load(YOLOPV2_WEIGHTS, map_location="cpu")
    yolopv2_model.eval()

    n_new, n_ll_empty = 0, 0
    for img_id, anns in anns_by_image.items():
        info = images_by_id[img_id]
        fname, h, w = info["file_name"], info["height"], info["width"]

        src_img_path = os.path.join(handoff_images_dir, fname)
        assert os.path.isfile(src_img_path), f"{src_img_path} 없음"
        bgr = cv2.imread(src_img_path)
        assert bgr.shape[:2] == (h, w), f"{fname} 크기 불일치: json({h},{w}) vs 실제{bgr.shape[:2]}"

        da_mask = np.zeros((h, w), dtype=bool)
        for a in anns:
            da_mask |= decode_da(a["segmentation"], h, w)

        ll_raw = infer_ll_yolopv2(yolopv2_model, bgr)
        ll_mask, n_poly = refine_ll(ll_raw)
        if n_poly == 0:
            n_ll_empty += 1
        da_final = da_mask | ll_mask

        cv2.imwrite(os.path.join(OUT_IMAGES, fname), bgr)
        cv2.imwrite(os.path.join(OUT_DA, fname), (da_final.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(OUT_LL, fname), (ll_mask.astype(np.uint8) * 255))
        n_new += 1

    print(f"신규(사람 da + YOLOPv2 ll) {n_new}장 추가 완료 (ll 못 찾은 프레임 {n_ll_empty}개)")
    total = len(os.listdir(OUT_IMAGES))
    print(f"\n완료: bootstrap_v2 총 {total}장 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
