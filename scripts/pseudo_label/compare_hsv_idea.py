#!/usr/bin/env python3
"""HSV 색상 기반 da 필터링: 확실한 트랙 픽셀(화면 하단 중앙)에서 뽑은 HSV 범위 밖에
있는 da 픽셀은 제거. ll을 아예 안 쓰는 방식 - width-borrow와 나란히 비교."""
import os

import cv2
import numpy as np

BASE_DIR = os.path.expanduser("~/fine-tune")
PSEUDO_DIR = os.path.join(BASE_DIR, "pseudo_dataset_v2")
OUT_PATH = "/mnt/c/fine-tune/TwinLiteNet-KMU-finetune/outputs/montages/hsv_vs_widthborrow_compare.png"

THUMB_W, THUMB_H = 340, 255
ROI_Y0, ROI_Y1 = 250, 390
FRAMES = ["frame_000871", "frame_000876", "frame_000880", "frame_000884", "frame_000890"]

# 캘리브레이션 결과 기준 (5~95 percentile) - 약간 여유를 둠
H_LO, H_HI = 20, 130
S_LO, S_HI = 5, 200
V_LO, V_HI = 30, 255


def hsv_filter(img_bgr, da_mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    color_ok = (h >= H_LO) & (h <= H_HI) & (s >= S_LO) & (s <= S_HI) & (v >= V_LO) & (v <= V_HI)
    filtered = da_mask & color_ok
    # 작은 구멍/노이즈 정리
    filtered_u8 = (filtered.astype(np.uint8)) * 255
    filtered_u8 = cv2.morphologyEx(filtered_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return filtered_u8 > 0


def find_confident_rows(ll_mask, min_gap=60):
    hh = ll_mask.shape[0]
    confident = {}
    for y in range(hh):
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
    hh, ww = da_mask.shape
    confident = find_confident_rows(ll_mask)
    clipped = da_mask.copy()
    if not confident:
        return clipped
    conf_ys = np.array(sorted(confident.keys()))
    for y in range(hh):
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
        row = np.zeros(ww, dtype=bool)
        lo, hi = max(0, int(left_b)), min(ww, int(right_b) + 1)
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
    cv2.rectangle(t, (0, 0), (THUMB_W, 22), (0, 0, 0), -1)
    cv2.putText(t, text, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def main():
    rows = []
    for stem in FRAMES:
        fname = stem + ".png"
        img0 = cv2.imread(os.path.join(PSEUDO_DIR, "images", fname))
        da = cv2.imread(os.path.join(PSEUDO_DIR, "da_masks", fname), cv2.IMREAD_GRAYSCALE) > 0
        ll = cv2.imread(os.path.join(PSEUDO_DIR, "ll_masks", fname), cv2.IMREAD_GRAYSCALE) > 0

        c_hsv = hsv_filter(img0, da)
        c_wb = idea1_width_borrow(da, ll)

        pct_hsv = (da.sum() - c_hsv.sum()) / da.sum() * 100
        pct_wb = (da.sum() - c_wb.sum()) / da.sum() * 100

        p0 = label(overlay(img0, da, np.zeros_like(ll)), f"{fname} [원본]")
        p1 = label(overlay(img0, c_hsv, np.zeros_like(ll)), f"HSV 필터 (-{pct_hsv:.1f}%)")
        p2 = label(overlay(img0, c_wb, ll), f"width-borrow (-{pct_wb:.1f}%)")

        rows.append(np.hstack([p0, p1, p2]))
        print(f"{stem}: HSV -{pct_hsv:.1f}% / width-borrow -{pct_wb:.1f}%")

    grid = np.vstack(rows)
    cv2.imwrite(OUT_PATH, grid)
    print("저장:", OUT_PATH, grid.shape)


if __name__ == "__main__":
    main()
