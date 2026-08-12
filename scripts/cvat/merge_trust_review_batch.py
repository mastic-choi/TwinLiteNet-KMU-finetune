#!/usr/bin/env python3
"""da_trust_review.ipynb에서 신뢰/SAM보정 처리된 프레임(trust_review_output/trusted/)을
bootstrap_v2(사람검증 corpus)에 병합한다.

- images/da_masks: trust_review_output/trusted/에서 그대로 복사 (사람이 신뢰했거나
  SAM으로 직접 보정한 da - 이미 검증됨)
- ll_masks: 여기서는 검증 안 됨(da_trust_review.ipynb는 da만 다룸) -> §2.12
  build_bootstrap_v2.py의 "신규 60장" 처리와 동일한 관례로 YOLOPv2+skeleton
  자동 생성해서 채움
- da_final = da | ll (기존 관례 유지)
"""
import os
import sys

import cv2
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "YOLOPv2"))
from utils.utils import letterbox as yolopv2_letterbox  # noqa: E402
sys.path.insert(0, os.path.join(BASE, "scripts", "pseudo_label"))
from skeleton_polyline_utils import mask_to_polylines, polylines_to_mask  # noqa: E402

TRUSTED_DIR = os.path.join(BASE, "trust_review_output", "trusted")
BOOTSTRAP_DIR = os.path.join(BASE, "bootstrap_v2")
YOLOPV2_WEIGHTS = os.path.join(BASE, "YOLOPv2", "data", "weights", "yolopv2.pt")
LL_REFINE_WIDTH = 8


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
    existing = set(os.listdir(os.path.join(BOOTSTRAP_DIR, "images")))
    trusted_files = sorted(os.listdir(os.path.join(TRUSTED_DIR, "images")))
    new_files = [f for f in trusted_files if f not in existing]
    print(f"trusted {len(trusted_files)}장 중 신규 {len(new_files)}장 (겹침 {len(trusted_files)-len(new_files)}장은 스킵)")

    for d in ("images", "da_masks", "ll_masks"):
        os.makedirs(os.path.join(BOOTSTRAP_DIR, d), exist_ok=True)

    yolopv2_model = torch.jit.load(YOLOPV2_WEIGHTS, map_location="cpu")
    yolopv2_model.eval()

    n_ll_empty = 0
    for i, fname in enumerate(new_files):
        img = cv2.imread(os.path.join(TRUSTED_DIR, "images", fname))
        da = cv2.imread(os.path.join(TRUSTED_DIR, "da_masks", fname), cv2.IMREAD_GRAYSCALE) > 127

        ll_raw = infer_ll_yolopv2(yolopv2_model, img)
        ll_mask, n_poly = refine_ll(ll_raw)
        if n_poly == 0:
            n_ll_empty += 1
        da_final = da | ll_mask

        cv2.imwrite(os.path.join(BOOTSTRAP_DIR, "images", fname), img)
        cv2.imwrite(os.path.join(BOOTSTRAP_DIR, "da_masks", fname), (da_final.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(BOOTSTRAP_DIR, "ll_masks", fname), (ll_mask.astype(np.uint8) * 255))

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(new_files)}")

    total = len(os.listdir(os.path.join(BOOTSTRAP_DIR, "images")))
    print(f"\n병합 완료: 신규 {len(new_files)}장 추가 (ll 못 찾은 프레임 {n_ll_empty}개)")
    print(f"bootstrap_v2 총 {total}장")


if __name__ == "__main__":
    main()
