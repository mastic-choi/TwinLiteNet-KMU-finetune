#!/usr/bin/env python3
"""frame_000876 한 장에 da-클리핑 아이디어 3개를 각각 적용해서 비교.
1) 폭 빌려쓰기(width-borrow): 확신 row(양쪽 선 다 보임)의 폭을, 한쪽 선만 보이는
   row에 빌려줘서 반대쪽 경계를 추정.
2) 좌/우 곡선 피팅: 확신 row들의 (y,left_x)/(y,right_x) 점만 모아 2차 곡선 피팅,
   전체 row에 적용.
3) da 자체 매끈함: ll 안 쓰고, 이웃 row 대비 da 경계가 갑자기 튀는 부분만 제거.
"""
import os

import cv2
import numpy as np

BASE_DIR = os.path.expanduser("~/fine-tune")
PSEUDO_DIR = os.path.join(BASE_DIR, "pseudo_dataset_v2")
OUT_PATH = "/mnt/c/fine-tune/TwinLiteNet-KMU-finetune/outputs/montages/clip_ideas_compare_876.png"

FRAME = "frame_000876"
ROI_Y0, ROI_Y1 = 250, 390


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


def idea2_curve_fit(da_mask, ll_mask, degree=2):
    h, w = da_mask.shape
    confident = find_confident_rows(ll_mask)
    clipped = da_mask.copy()
    if len(confident) < degree + 1:
        return clipped

    ys = np.array(sorted(confident.keys()))
    lxs = np.array([confident[y][0] for y in ys])
    rxs = np.array([confident[y][1] for y in ys])
    left_poly = np.polyfit(ys, lxs, degree)
    right_poly = np.polyfit(ys, rxs, degree)

    for y in range(h):
        if len(np.where(da_mask[y])[0]) == 0:
            continue
        left_b = np.polyval(left_poly, y)
        right_b = np.polyval(right_poly, y)
        if left_b >= right_b:
            continue
        row = np.zeros(w, dtype=bool)
        lo, hi = max(0, int(left_b)), min(w, int(right_b) + 1)
        row[lo:hi] = da_mask[y, lo:hi]
        clipped[y] = row
    return clipped


def idea3_da_smoothness(da_mask, window=15, tolerance=20):
    h, w = da_mask.shape
    lefts = np.full(h, -1, dtype=np.int32)
    rights = np.full(h, -1, dtype=np.int32)
    for y in range(h):
        xs = np.where(da_mask[y])[0]
        if len(xs):
            lefts[y] = xs.min()
            rights[y] = xs.max()

    clipped = da_mask.copy()
    for y in range(h):
        if lefts[y] < 0:
            continue
        y0, y1 = max(0, y - window), min(h, y + window + 1)
        nl = [lefts[yy] for yy in range(y0, y1) if yy != y and lefts[yy] >= 0]
        nr = [rights[yy] for yy in range(y0, y1) if yy != y and rights[yy] >= 0]
        if not nl or not nr:
            continue
        med_l, med_r = np.median(nl), np.median(nr)

        new_left = lefts[y]
        new_right = rights[y]
        if lefts[y] < med_l - tolerance:
            new_left = int(med_l)
        if rights[y] > med_r + tolerance:
            new_right = int(med_r)
        if new_left == lefts[y] and new_right == rights[y]:
            continue

        row = np.zeros(w, dtype=bool)
        lo, hi = max(0, new_left), min(w, new_right + 1)
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
    t = img.copy()
    cv2.rectangle(t, (0, 0), (t.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(t, text, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def main():
    img0 = cv2.imread(os.path.join(PSEUDO_DIR, "images", FRAME + ".png"))
    da = cv2.imread(os.path.join(PSEUDO_DIR, "da_masks", FRAME + ".png"), cv2.IMREAD_GRAYSCALE) > 0
    ll = cv2.imread(os.path.join(PSEUDO_DIR, "ll_masks", FRAME + ".png"), cv2.IMREAD_GRAYSCALE) > 0

    c1 = idea1_width_borrow(da, ll)
    c2 = idea2_curve_fit(da, ll)
    c3 = idea3_da_smoothness(da)

    for name, c in [("1.width-borrow", c1), ("2.curve-fit", c2), ("3.da-smoothness", c3)]:
        before, after = int(da.sum()), int(c.sum())
        print(f"{name}: {before} -> {after} px ({(before-after)/before*100:.1f}% 감소)")

    p0 = label(overlay(img0, da, ll), f"0. 원본 da (현재 학습 데이터셋)")
    p1 = label(overlay(img0, c1, ll), f"1. width-borrow ({(da.sum()-c1.sum())/da.sum()*100:.1f}% 감소)")
    p2 = label(overlay(img0, c2, ll), f"2. curve-fit ({(da.sum()-c2.sum())/da.sum()*100:.1f}% 감소)")
    p3 = label(overlay(img0, c3, ll), f"3. da-smoothness (ll 안씀, {(da.sum()-c3.sum())/da.sum()*100:.1f}% 감소)")

    grid = np.hstack([p0, p1, p2, p3])
    cv2.imwrite(OUT_PATH, grid)
    print("저장:", OUT_PATH, grid.shape)


if __name__ == "__main__":
    main()
