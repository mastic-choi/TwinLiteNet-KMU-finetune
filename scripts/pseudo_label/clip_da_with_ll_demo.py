#!/usr/bin/env python3
"""하이브리드 아이디어 데모: 우리 da 라벨을, YOLOPv2 ll이 검출된 row에서만 그 두 선
사이 범위로 잘라낸다(ll이 없는 row는 원래 da 그대로 폴백 — 순수 geometric 재구성의
과소포함 문제를 피하기 위함). frame_000851 한 장에 적용해서 전/후 비교 이미지 생성."""
import os

import cv2
import numpy as np

BASE_DIR = os.path.expanduser("~/fine-tune")
PSEUDO_DIR = os.path.join(BASE_DIR, "pseudo_dataset_v2")
OUT_DIR = "/mnt/c/fine-tune/TwinLiteNet-KMU-finetune/outputs/montages"

FRAME = "frame_000851"
THUMB_W, THUMB_H = 400, 300
ROI_Y0, ROI_Y1 = 250, 390


def clip_da_with_ll(da_mask, ll_mask, edge_proximity=40):
    """row별로, **기존 da 가장자리 근처(edge_proximity px 이내)에 있는 선만** 경계로
    인정해서 그 바깥쪽만 잘라낸다. 화면 전체 기준 왼쪽/오른쪽으로 선을 분류하지 않음
    (커브에서 한 선이 row마다 좌/우가 뒤바뀌는 문제, 중앙 점선이 엉뚱하게 좌/우 중
    하나로 분류돼 안쪽을 잘라먹는 문제 둘 다 이 방식으로 회피 — da 안쪽 깊숙한 곳의
    선은 "경계선"이 아니라 "중앙선"으로 보고 무시함).
    da/ll 둘 다 없는 row, 또는 근처에 선이 없는 row는 원래 da 그대로 둠."""
    h, w = da_mask.shape
    clipped = da_mask.copy()
    n_clipped_rows = 0
    for y in range(h):
        da_xs = np.where(da_mask[y])[0]
        if len(da_xs) == 0:
            continue
        da_min, da_max = da_xs.min(), da_xs.max()

        ll_xs = np.where(ll_mask[y])[0]
        if len(ll_xs) == 0:
            continue  # ll 없는 row -> 원래 da 그대로 (폴백)

        left_candidates = ll_xs[ll_xs <= da_min + edge_proximity]
        right_candidates = ll_xs[ll_xs >= da_max - edge_proximity]

        new_min = left_candidates.min() if len(left_candidates) else da_min
        new_max = right_candidates.max() if len(right_candidates) else da_max

        before = clipped[y].sum()
        row = np.zeros(w, dtype=bool)
        row[new_min:new_max + 1] = da_mask[y, new_min:new_max + 1]
        clipped[y] = row
        if clipped[y].sum() < before:
            n_clipped_rows += 1
    return clipped, n_clipped_rows


def overlay(img0, da, ll, extra_ll_color=None):
    out = img0.copy()
    out[da] = (0.5 * out[da].astype(np.float64) + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
    out[ll] = (0, 0, 255)
    cv2.rectangle(out, (0, ROI_Y0), (out.shape[1], ROI_Y1), (0, 255, 255), 1)
    return out


def label(img, text):
    t = cv2.resize(img, (THUMB_W, THUMB_H))
    cv2.rectangle(t, (0, 0), (THUMB_W, 22), (0, 0, 0), -1)
    cv2.putText(t, text, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def main():
    img0 = cv2.imread(os.path.join(PSEUDO_DIR, "images", FRAME + ".png"))
    da = cv2.imread(os.path.join(PSEUDO_DIR, "da_masks", FRAME + ".png"), cv2.IMREAD_GRAYSCALE) > 0
    ll = cv2.imread(os.path.join(PSEUDO_DIR, "ll_masks", FRAME + ".png"), cv2.IMREAD_GRAYSCALE) > 0

    clipped, n_rows = clip_da_with_ll(da, ll)
    total_rows_with_ll = sum(1 for y in range(ll.shape[0]) if ll[y].any())
    da_px_before = int(da.sum())
    da_px_after = int(clipped.sum())
    print(f"ll 있는 row: {total_rows_with_ll} / 실제로 잘려나간 row: {n_rows}")
    print(f"da 픽셀 수: {da_px_before} -> {da_px_after} ({(da_px_before-da_px_after)/da_px_before*100:.1f}% 감소)")

    p1 = label(overlay(img0, da, np.zeros_like(ll)), "2. Our da label (before)")
    p2 = label(overlay(img0, da, ll), "+ YOLOPv2 ll overlay (boundary)")
    p3 = label(overlay(img0, clipped, ll), "da clipped to ll bounds (after)")

    grid = np.hstack([p1, p2, p3])
    out_path = os.path.join(OUT_DIR, "da_clip_with_ll_demo.png")
    cv2.imwrite(out_path, grid)
    print("저장:", out_path, grid.shape)


if __name__ == "__main__":
    main()
