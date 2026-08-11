#!/usr/bin/env python3
"""width-borrow 아이디어(확신 row의 폭을 빌려서 한쪽 선만 보이는 row의 반대쪽 경계를
추정)를 여러 프레임에 적용해서 좌(원본)/우(클리핑) 비교 몽타주 생성."""
import os

import cv2
import numpy as np

BASE_DIR = os.path.expanduser("~/fine-tune")
PSEUDO_DIR = os.path.join(BASE_DIR, "pseudo_dataset_v2")
OUT_PATH = "/mnt/c/fine-tune/TwinLiteNet-KMU-finetune/outputs/montages/width_borrow_montage_871_890.png"

THUMB_W, THUMB_H = 450, 340
ROI_Y0, ROI_Y1 = 250, 390
FRAMES = ["frame_000871", "frame_000876", "frame_000880", "frame_000884", "frame_000890"]


def find_confident_rows(ll_mask, min_gap=60):
    h = ll_mask.shape[0]
    confident = {}
    for y in range(h):
        xs = np.sort(np.where(ll_mask[y])[0])
        if len(xs) < 2:
            continue
        gaps = np.diff(xs)
        gi = int(np.argmax(gaps))
        if gaps[gi] < min_gap:
            continue
        left_xs, right_xs = xs[:gi + 1], xs[gi + 1:]
        confident[y] = (int(left_xs.min()), int(right_xs.max()))
    return confident


def idea1_width_borrow(da_mask, ll_mask):
    h, w = da_mask.shape
    confident = find_confident_rows(ll_mask)
    clipped = da_mask.copy()
    if not confident:
        return clipped
    conf_ys = np.array(sorted(confident.keys()))

    for y in range(h):
        if len(np.where(da_mask[y])[0]) == 0:
            continue
        if y in confident:
            left_b, right_b = confident[y]
        else:
            xs = np.where(ll_mask[y])[0]
            if len(xs) == 0:
                continue
            idx = np.searchsorted(conf_ys, y)
            cands = []
            if idx < len(conf_ys):
                cands.append(conf_ys[idx])
            if idx > 0:
                cands.append(conf_ys[idx - 1])
            nearest_y = min(cands, key=lambda cy: abs(cy - y))
            nl, nr = confident[nearest_y]
            width = nr - nl
            line_x = xs.mean()
            if abs(line_x - nl) < abs(line_x - nr):
                left_b = int(xs.min())
                right_b = left_b + width
            else:
                right_b = int(xs.max())
                left_b = right_b - width

        row = np.zeros(w, dtype=bool)
        lo, hi = max(0, int(left_b)), min(w, int(right_b) + 1)
        row[lo:hi] = da_mask[y, lo:hi]
        clipped[y] = row
    return clipped


def overlay(img0, da, ll):
    out = img0.copy()
    out[da] = (0.5 * out[da].astype(np.float64) + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
    out[ll] = (0, 0, 255)
    cv2.rectangle(out, (0, ROI_Y0), (out.shape[1], ROI_Y1), (0, 255, 255), 1)
    return out


def label(img, text):
    t = cv2.resize(img, (THUMB_W, THUMB_H))
    cv2.rectangle(t, (0, 0), (THUMB_W, 24), (0, 0, 0), -1)
    cv2.putText(t, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def main():
    rows = []
    for stem in FRAMES:
        fname = stem + ".png"
        img0 = cv2.imread(os.path.join(PSEUDO_DIR, "images", fname))
        da = cv2.imread(os.path.join(PSEUDO_DIR, "da_masks", fname), cv2.IMREAD_GRAYSCALE) > 0
        ll = cv2.imread(os.path.join(PSEUDO_DIR, "ll_masks", fname), cv2.IMREAD_GRAYSCALE) > 0

        clipped = idea1_width_borrow(da, ll)
        before_px, after_px = int(da.sum()), int(clipped.sum())
        pct = (before_px - after_px) / before_px * 100 if before_px else 0

        p_left = label(overlay(img0, da, np.zeros_like(ll)), f"{fname}  [현재 학습 데이터셋]")
        p_right = label(overlay(img0, clipped, ll), f"width-borrow  (-{pct:.1f}%)")

        rows.append(np.hstack([p_left, p_right]))
        print(f"{stem}: da {before_px} -> {after_px} px ({pct:.1f}% 감소)")

    grid = np.vstack(rows)
    header = np.full((30, grid.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(header, "1. 지금 학습에 쓰는 데이터셋 (원본 da)", (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header, "2. width-borrow로 크롭한 da (제안)", (THUMB_W + 10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    grid = np.vstack([header, grid])

    cv2.imwrite(OUT_PATH, grid)
    print("저장:", OUT_PATH, grid.shape)


if __name__ == "__main__":
    main()
