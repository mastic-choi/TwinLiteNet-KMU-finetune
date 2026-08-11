#!/usr/bin/env python3
"""신모델(best(1).pth, small, 470장)이 실제 사람 GT(bootstrap 74장) 대비 da를 얼마나
과다포함(over-paint, false positive)/과소포함(under-paint, false negative)하는지 직접 측정.
목적: "da가 차선 밖까지 넓게 칠해진다"는 육안 관찰이 (a) pseudo label(396장, letterbox
버그로 학습된 모델이 생성) 특유의 문제인지, (b) 사람이 직접 그린 GT(74장)에서도 재현되는
더 근본적인 문제(threshold/undertraining 등)인지 구분하기 위함.
"""
import os
import sys
from argparse import Namespace

import cv2
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "_TwinLiteNetPlus_ref"))
from model.model import TwinLiteNetPlus  # noqa: E402

NEW_PTH = os.path.expanduser("~/Downloads/best(1).pth")
NEW_CONFIG = "small"
NEW_W, NEW_H = 640, 384

GT_IMAGES = os.path.join(BASE, "bootstrap", "images")
GT_DA = os.path.join(BASE, "bootstrap", "da_masks")

OUT_PATH = os.path.join(BASE, "outputs", "montages", "da_overpaint_check.png")
THUMB_W, THUMB_H = 320, 240
N_SHOW = 12


def infer_new(model, img0):
    h, w = img0.shape[:2]
    resized = cv2.resize(img0, (NEW_W, NEW_H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])).float()
    with torch.no_grad():
        out_da, _ = model(blob)
    da_mask = torch.argmax(out_da, dim=1)[0].numpy().astype(np.uint8)
    return cv2.resize(da_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def main():
    model = TwinLiteNetPlus(Namespace(config=NEW_CONFIG))
    state = torch.load(NEW_PTH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    names = sorted(os.path.splitext(f)[0] for f in os.listdir(GT_IMAGES))
    fp_ratios, fn_ratios, ious = [], [], []
    thumbs = []
    for i, n in enumerate(names):
        img0 = cv2.imread(os.path.join(GT_IMAGES, n + ".png"))
        if img0 is None:
            img0 = cv2.imread(os.path.join(GT_IMAGES, n + ".jpg"))
        gt = cv2.imread(os.path.join(GT_DA, n + ".png"), cv2.IMREAD_GRAYSCALE) > 127

        pred = infer_new(model, img0)

        tp = pred & gt
        fp = pred & ~gt  # 과다포함: GT엔 없는데 모델이 도로라고 칠한 부분
        fn = ~pred & gt  # 과소포함

        gt_area = max(gt.sum(), 1)
        fp_ratio = fp.sum() / gt_area  # GT 면적 대비 얼마나 "밖으로" 칠했는지
        fn_ratio = fn.sum() / gt_area
        iou = tp.sum() / max((pred | gt).sum(), 1)
        fp_ratios.append(fp_ratio)
        fn_ratios.append(fn_ratio)
        ious.append(iou)

        if i < N_SHOW:
            vis = img0.copy()
            vis[tp] = (0.5 * vis[tp].astype(np.float64) + 0.5 * np.array([0, 200, 0])).astype(np.uint8)   # TP=초록
            vis[fp] = (0.5 * vis[fp].astype(np.float64) + 0.5 * np.array([0, 0, 255])).astype(np.uint8)   # FP=빨강(과다포함)
            vis[fn] = (0.5 * vis[fn].astype(np.float64) + 0.5 * np.array([255, 0, 0])).astype(np.uint8)   # FN=파랑(과소포함)
            thumb = cv2.resize(vis, (THUMB_W, THUMB_H))
            cv2.rectangle(thumb, (0, 0), (THUMB_W, 18), (0, 0, 0), -1)
            cv2.putText(thumb, f"{n} IoU={iou:.2f} FP/GT={fp_ratio:.2f}", (3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            thumbs.append(thumb)

    print(f"74장 전체: mean IoU={np.mean(ious):.3f} | mean FP/GT(과다포함 비율)={np.mean(fp_ratios):.3f} "
          f"| mean FN/GT(과소포함 비율)={np.mean(fn_ratios):.3f}")
    print(f"과다포함(FP/GT) 상위 5장: ")
    order = np.argsort(fp_ratios)[::-1][:5]
    for idx in order:
        print(f"  {names[idx]}: FP/GT={fp_ratios[idx]:.3f} IoU={ious[idx]:.3f}")

    cols = 4
    rows_n = (len(thumbs) + cols - 1) // cols
    grid = np.full((rows_n * THUMB_H, cols * THUMB_W, 3), 30, dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * THUMB_H:(r + 1) * THUMB_H, c * THUMB_W:(c + 1) * THUMB_W] = t
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, grid)
    print("시각화 저장:", OUT_PATH, "(초록=맞음 TP, 빨강=과다포함 FP, 파랑=과소포함 FN)")


if __name__ == "__main__":
    main()
