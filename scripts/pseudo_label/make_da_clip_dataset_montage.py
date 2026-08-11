#!/usr/bin/env python3
"""좌: 지금 학습에 쓰는 da 라벨(pseudo_dataset_v2/da_masks) 그대로.
우: 같은 da를 YOLOPv2 ll 라벨 기준으로 좌/우 선 바깥을 잘라낸 버전(하이브리드 클리핑).
여러 프레임에 적용해서 일관되게 작동하는지 비교."""
import os

import cv2
import numpy as np

BASE_DIR = os.path.expanduser("~/fine-tune")
PSEUDO_DIR = os.path.join(BASE_DIR, "pseudo_dataset_v2")
OUT_PATH = "/mnt/c/fine-tune/TwinLiteNet-KMU-finetune/outputs/montages/da_clip_dataset_montage_870_890.png"

THUMB_W, THUMB_H = 450, 340
ROI_Y0, ROI_Y1 = 250, 390
FRAMES = ["frame_000871", "frame_000876", "frame_000880", "frame_000884", "frame_000890"]  # ㅇ을 못 잡던 870~890 구간


def clip_da_with_ll(da_mask, ll_mask, min_gap_abs=60, min_gap_ratio=0.45):
    """row별로 ll 픽셀들을 x좌표로 정렬해서 가장 큰 간격(gap)을 찾되, 이걸 "진짜 좌/우
    분리"로 인정하려면 **절대 픽셀(min_gap_abs) 이상 + 그 row의 da 폭 대비 비율
    (min_gap_ratio) 이상**을 동시에 만족해야 함. 화면 먼 쪽(커브 진입부 등)은 원래
    좌/우 폭 자체가 좁아서, 한 선이 검출 중간에 끊긴 것(같은 선의 내부 gap)과 진짜
    좌우 분리를 절대 픽셀값만으론 구분 못 함 — da 폭 대비 비율을 같이 요구해서 이
    혼동을 줄임. 또한 갈라진 두 그룹 중 하나가 너무 얇으면(min_group_width 미만,
    노이즈 가능성) 역시 신뢰하지 않음. 조건 중 하나라도 애매하면 **그 row는 절대
    자르지 않고 원본 da 그대로 둠**(확신 없으면 안 건드리는 게 원칙)."""
    h, w = da_mask.shape
    clipped = da_mask.copy()
    for y in range(h):
        da_xs = np.where(da_mask[y])[0]
        if len(da_xs) == 0:
            continue
        da_width = da_xs.max() - da_xs.min() + 1

        xs = np.sort(np.where(ll_mask[y])[0])
        if len(xs) < 2:
            continue  # 선이 0~1개면 좌/우 구분 불가 -> 안 건드림

        gaps = np.diff(xs)
        gap_idx = int(np.argmax(gaps))
        gap = gaps[gap_idx]
        if gap < min_gap_abs or gap < da_width * min_gap_ratio:
            continue  # 절대/상대 기준 중 하나라도 미달 -> 확신 없음, 안 건드림

        left_xs = xs[:gap_idx + 1]
        right_xs = xs[gap_idx + 1:]

        row = da_mask[y].copy()
        row[:left_xs.min()] = False
        row[right_xs.max() + 1:] = False
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

        clipped = clip_da_with_ll(da, ll)
        before_px, after_px = int(da.sum()), int(clipped.sum())
        pct = (before_px - after_px) / before_px * 100 if before_px else 0

        p_left = label(overlay(img0, da, np.zeros_like(ll)),
                        f"{fname}  [현재 학습 데이터셋]")
        p_right = label(overlay(img0, clipped, ll),
                         f"ll 경계로 크롭한 da  (-{pct:.1f}%)")

        rows.append(np.hstack([p_left, p_right]))
        print(f"{stem}: da {before_px} -> {after_px} px ({pct:.1f}% 감소)")

    grid = np.vstack(rows)
    header = np.full((30, grid.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(header, "1. 지금 학습에 쓰는 데이터셋 (원본 da)", (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header, "2. YOLOPv2 ll 경계로 크롭한 da (제안)", (THUMB_W + 10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    grid = np.vstack([header, grid])

    cv2.imwrite(OUT_PATH, grid)
    print("저장:", OUT_PATH, grid.shape)


if __name__ == "__main__":
    main()
