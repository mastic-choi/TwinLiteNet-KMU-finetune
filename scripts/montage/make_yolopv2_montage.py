#!/usr/bin/env python3
"""YOLOPv2(사전학습, BDD100K 640x384 기준, https://github.com/CAIC-AD/YOLOPv2) 20장 몽타주.
우리 이미지(640x480, 4:3)는 BDD100K(1280x720, 16:9)와 종횡비가 달라서 레포의 하드코딩된
crop[12:372] 후처리를 그대로 쓰면 안 맞음 -> letterbox 패딩이 우리 케이스엔 거의 0이라
(480이 stride=32 배수) crop 없이 letterbox 해상도 그대로 argmax/round해서 원본 크기로
resize하는 방식으로 대체함.
"""
import math
import os
import sys

import cv2
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YOLOPV2_DIR = os.path.join(BASE, "YOLOPv2")
sys.path.insert(0, YOLOPV2_DIR)
from utils.utils import letterbox  # noqa: E402

IMAGES_DIR = os.path.join(BASE, "raw", "images_todo")
OUT_PATH = os.path.join(BASE, "outputs", "montages", "yolopv2_montage.png")
WEIGHTS = os.path.join(YOLOPV2_DIR, "data", "weights", "yolopv2.pt")
THUMB_W, THUMB_H = 320, 240
ROI_Y0, ROI_Y1 = 250, 390

FRAMES = [
    'frame_000024.png', 'frame_000173.png', 'frame_000268.png', 'frame_000380.png',
    'frame_000457.png', 'frame_000543.png', 'frame_000674.png', 'frame_000750.png',
    'frame_000861.png', 'frame_000968.png', 'frame_001058.png', 'frame_001156.png',
    'frame_001225.png', 'frame_001389.png', 'frame_001476.png', 'frame_001595.png',
    'frame_001680.png', 'frame_001758.png', 'frame_001864.png', 'frame_001966.png',
]


def roi_cov(mask):
    roi = mask[ROI_Y0:ROI_Y1]
    return float(np.count_nonzero(roi)) / roi.size


def main():
    model = torch.jit.load(WEIGHTS, map_location="cpu")
    model.eval()

    thumbs = []
    for fname in FRAMES:
        img0 = cv2.imread(os.path.join(IMAGES_DIR, fname))
        h0, w0 = img0.shape[:2]
        img, _, _ = letterbox(img0, 640, stride=32)
        inp = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        inp_t = torch.from_numpy(inp).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            [pred, anchor_grid], seg, ll = model(inp_t)

        da = torch.argmax(seg, dim=1)[0].numpy().astype(np.uint8)
        ll_mask = torch.round(ll)[0, 0].numpy().astype(np.uint8)

        da_full = cv2.resize(da, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
        ll_full = cv2.resize(ll_mask, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
        da_cov, ll_cov = roi_cov(da_full), roi_cov(ll_full)

        overlay = img0.copy()
        overlay[da_full] = (0.5 * overlay[da_full].astype(np.float64) + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
        overlay[ll_full] = (0, 0, 255)
        cv2.rectangle(overlay, (0, ROI_Y0), (w0, ROI_Y1), (0, 255, 255), 1)

        thumb = cv2.resize(overlay, (THUMB_W, THUMB_H))
        label = f"{fname} da={da_cov:.3f} ll={ll_cov:.3f}"
        cv2.rectangle(thumb, (0, 0), (THUMB_W, 18), (0, 0, 0), -1)
        cv2.putText(thumb, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
        print(f"{fname}: da={da_cov:.3f} ll={ll_cov:.3f}")

    cols = 5
    rows_n = math.ceil(len(thumbs) / cols)
    grid = np.zeros((rows_n * THUMB_H, cols * THUMB_W, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * THUMB_H:(r + 1) * THUMB_H, c * THUMB_W:(c + 1) * THUMB_W] = t
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, grid)
    print(f"\n{len(thumbs)}장 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
