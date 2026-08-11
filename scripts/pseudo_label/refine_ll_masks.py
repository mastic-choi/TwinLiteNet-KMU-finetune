#!/usr/bin/env python3
"""브러시 마스크의 두께 노이즈(4~48px로 들쭉날쭉, ll IoU 정체의 원인으로 추정)를
없애기 위해 ll_masks를 skeleton화 -> polyline 단순화 -> 고정 두께로 재래스터화한다.
da_masks는 export_cvat_bootstrap.py의 기존 관례(da |= ll)대로 재합성해서 같이 저장.

사용법:
  python refine_ll_masks.py [--src bootstrap] [--out bootstrap_refined] [--width 8]
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np

from skeleton_polyline_utils import mask_to_polylines, polylines_to_mask

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="bootstrap")
    ap.add_argument("--out", default="bootstrap_refined")
    ap.add_argument("--width", type=int, default=8, help="재래스터화 고정 두께(px)")
    args = ap.parse_args()

    src = os.path.join(BASE, args.src)
    out = os.path.join(BASE, args.out)
    for d in ("images", "da_masks", "ll_masks"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    ll_files = sorted(glob.glob(os.path.join(src, "ll_masks", "*.png")))
    widths_before, widths_after = [], []

    for ll_path in ll_files:
        fname = os.path.basename(ll_path)
        ll_mask = cv2.imread(ll_path, cv2.IMREAD_GRAYSCALE)
        h, w = ll_mask.shape

        polylines = mask_to_polylines(ll_mask)
        ll_refined = polylines_to_mask(polylines, h, w, line_width=args.width)

        da_path = os.path.join(src, "da_masks", fname)
        da_mask = cv2.imread(da_path, cv2.IMREAD_GRAYSCALE)
        da_refined = np.maximum(da_mask, ll_refined * 255)

        img_path = os.path.join(src, "images", fname)
        shutil.copy(img_path, os.path.join(out, "images", fname))
        cv2.imwrite(os.path.join(out, "ll_masks", fname), ll_refined * 255)
        cv2.imwrite(os.path.join(out, "da_masks", fname), da_refined)

        dist_before = cv2.distanceTransform((ll_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        dist_after = cv2.distanceTransform(ll_refined, cv2.DIST_L2, 5)
        if (ll_mask > 0).any():
            widths_before.append(dist_before.max() * 2)
        if ll_refined.any():
            widths_after.append(dist_after.max() * 2)

    print(f"{len(ll_files)}장 처리 완료 -> {out}")
    if widths_before:
        wb = np.array(widths_before)
        print(f"[재래스터화 전] 두께(max, px) mean={wb.mean():.1f} std={wb.std():.1f} "
              f"min={wb.min():.1f} max={wb.max():.1f}")
    if widths_after:
        wa = np.array(widths_after)
        print(f"[재래스터화 후] 두께(max, px) mean={wa.mean():.1f} std={wa.std():.1f} "
              f"min={wa.min():.1f} max={wa.max():.1f}")


if __name__ == "__main__":
    main()
