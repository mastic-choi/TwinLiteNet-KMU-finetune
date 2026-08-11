#!/usr/bin/env python3
"""da-클리핑 아이디어 3개(width-borrow / curve-fit / da-smoothness)를 여러 프레임에
적용해서 4열(원본/1/2/3) 비교 몽타주 생성."""
import os

import cv2
import numpy as np

BASE_DIR = os.path.expanduser("~/fine-tune")
PSEUDO_DIR = os.path.join(BASE_DIR, "pseudo_dataset_v2")
OUT_PATH = "/mnt/c/fine-tune/TwinLiteNet-KMU-finetune/outputs/montages/clip_ideas_compare_multi.png"

THUMB_W, THUMB_H = 340, 255
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

        c1 = idea1_width_borrow(da, ll)
        c2 = idea2_curve_fit(da, ll)
        c3 = idea3_da_smoothness(da)

        pct1 = (da.sum() - c1.sum()) / da.sum() * 100
        pct2 = (da.sum() - c2.sum()) / da.sum() * 100
        pct3 = (da.sum() - c3.sum()) / da.sum() * 100

        p0 = label(overlay(img0, da, np.zeros_like(ll)), f"{fname} [원본]")
        p1 = label(overlay(img0, c1, ll), f"1.width-borrow (-{pct1:.1f}%)")
        p2 = label(overlay(img0, c2, ll), f"2.curve-fit (-{pct2:.1f}%)")
        p3 = label(overlay(img0, c3, ll), f"3.da-smoothness (-{pct3:.1f}%)")

        rows.append(np.hstack([p0, p1, p2, p3]))
        print(f"{stem}: width-borrow -{pct1:.1f}% / curve-fit -{pct2:.1f}% / da-smoothness -{pct3:.1f}%")

    grid = np.vstack(rows)
    cv2.imwrite(OUT_PATH, grid)
    print("저장:", OUT_PATH, grid.shape)


if __name__ == "__main__":
    main()
