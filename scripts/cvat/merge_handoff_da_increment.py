#!/usr/bin/env python3
"""labeling_handoff(100장, da만 사람 라벨링 요청)의 후속 진행분 병합.

build_bootstrap_v2.py가 이미 이 핸드오프의 "신규 60장"(당시 완료분)을
bootstrap_v2/pseudo_dataset_v2에 반영했다. 이후 팀원이 22장을 추가로 완료해
82/100장이 됐는데(2026-08-12 기준, 나머지 18장은 아직 미완료), 이 22장만 골라서
같은 방식(da=사람 폴리곤, ll=YOLOPv2+skeleton, da_final=da|ll)으로
bootstrap_v2/(깨끗한 사람검증 소스)와 pseudo_dataset_v2/(실제 학습셋) 양쪽에
증분 병합한다. 이미 반영된 60장은 bootstrap_v2/images 목록과 대조해서 자동으로
건너뛴다.
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

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "YOLOPv2"))
from utils.utils import letterbox as yolopv2_letterbox  # noqa: E402
sys.path.insert(0, os.path.join(BASE, "scripts", "pseudo_label"))
from skeleton_polyline_utils import mask_to_polylines, polylines_to_mask  # noqa: E402

DA_COCO_JSON = os.path.expanduser("~/Downloads/da_coco(1).json")
HANDOFF_ZIP = os.path.join(BASE, "labeling_handoff.zip")
BOOTSTRAP_V2_DIR = os.path.join(BASE, "bootstrap_v2")
PSEUDO_DIR = os.path.join(BASE, "pseudo_dataset_v2")
YOLOPV2_WEIGHTS = os.path.join(BASE, "YOLOPv2", "data", "weights", "yolopv2.pt")

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

    already_done = set(os.listdir(os.path.join(BOOTSTRAP_V2_DIR, "images")))

    with open(DA_COCO_JSON) as f:
        coco = json.load(f)
    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    new_ids = {
        img_id for img_id in anns_by_image
        if images_by_id[img_id]["file_name"] not in already_done
    }
    print(f"이번 export: {len(anns_by_image)}장 라벨 완료 / 이미 반영됨: "
          f"{len(anns_by_image) - len(new_ids)}장 / 이번에 새로 병합: {len(new_ids)}장")
    if not new_ids:
        print("새로 병합할 프레임 없음 - 종료")
        return

    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    with zipfile.ZipFile(HANDOFF_ZIP) as zf:
        zf.extractall(EXTRACT_DIR)
    handoff_images_dir = os.path.join(EXTRACT_DIR, "labeling_handoff", "images")
    assert os.path.isdir(handoff_images_dir), f"{handoff_images_dir} 없음 (zip 구조 확인 필요)"

    yolopv2_model = torch.jit.load(YOLOPV2_WEIGHTS, map_location="cpu")
    yolopv2_model.eval()

    for d in (BOOTSTRAP_V2_DIR, PSEUDO_DIR):
        for sub in ("images", "da_masks", "ll_masks"):
            os.makedirs(os.path.join(d, sub), exist_ok=True)

    pseudo_before = set(os.listdir(os.path.join(PSEUDO_DIR, "images")))
    n_new, n_ll_empty, n_pseudo_upgraded, n_pseudo_added = 0, 0, 0, 0
    for img_id in sorted(new_ids, key=lambda i: images_by_id[i]["file_name"]):
        anns = anns_by_image[img_id]
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

        img_bytes = cv2.imencode(".png", bgr)[1]
        da_bytes = cv2.imencode(".png", da_final.astype(np.uint8) * 255)[1]
        ll_bytes = cv2.imencode(".png", ll_mask.astype(np.uint8) * 255)[1]

        for out_dir in (BOOTSTRAP_V2_DIR, PSEUDO_DIR):
            with open(os.path.join(out_dir, "images", fname), "wb") as f:
                f.write(img_bytes)
            with open(os.path.join(out_dir, "da_masks", fname), "wb") as f:
                f.write(da_bytes)
            with open(os.path.join(out_dir, "ll_masks", fname), "wb") as f:
                f.write(ll_bytes)

        if fname in pseudo_before:
            n_pseudo_upgraded += 1  # 기존 pseudo label -> 사람 라벨로 업그레이드
        else:
            n_pseudo_added += 1  # pseudo_dataset_v2에 신규 편입
        n_new += 1

    total_bv2 = len(os.listdir(os.path.join(BOOTSTRAP_V2_DIR, "images")))
    total_pseudo = len(os.listdir(os.path.join(PSEUDO_DIR, "images")))
    print(f"\n신규 병합: {n_new}장 (ll 못 찾은 프레임 {n_ll_empty}개)")
    print(f"  pseudo_dataset_v2 중 pseudo->사람 라벨 업그레이드: {n_pseudo_upgraded}장, "
          f"신규 편입: {n_pseudo_added}장")
    print(f"bootstrap_v2 총 {total_bv2}장 (사람검증 da+ll 소스)")
    print(f"pseudo_dataset_v2 총 {total_pseudo}장")


if __name__ == "__main__":
    main()
