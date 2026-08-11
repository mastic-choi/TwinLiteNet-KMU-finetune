#!/usr/bin/env python3
"""Mask2Former(Cityscapes 사전학습, facebook/mask2former-swin-large-cityscapes-semantic)
20장 몽타주. SegFormer와 마찬가지로 road(da)만 비교 가능(ll 클래스 없음)."""
import math
import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE, "raw", "images_todo")
OUT_PATH = os.path.join(BASE, "outputs", "montages", "mask2former_montage.png")
THUMB_W, THUMB_H = 320, 240
ROI_Y0, ROI_Y1 = 250, 390

FRAMES = [
    'frame_000024.png', 'frame_000173.png', 'frame_000268.png', 'frame_000380.png',
    'frame_000457.png', 'frame_000543.png', 'frame_000674.png', 'frame_000750.png',
    'frame_000861.png', 'frame_000968.png', 'frame_001058.png', 'frame_001156.png',
    'frame_001225.png', 'frame_001389.png', 'frame_001476.png', 'frame_001595.png',
    'frame_001680.png', 'frame_001758.png', 'frame_001864.png', 'frame_001966.png',
]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def roi_cov(mask):
    roi = mask[ROI_Y0:ROI_Y1]
    return float(np.count_nonzero(roi)) / roi.size


def main():
    model_id = "facebook/mask2former-swin-large-cityscapes-semantic"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).to(DEVICE)
    model.eval()
    road_id = [k for k, v in model.config.id2label.items() if v == "road"][0]
    print("road class id:", road_id)

    thumbs = []
    for fname in FRAMES:
        img_path = os.path.join(IMAGES_DIR, fname)
        image = Image.open(img_path).convert("RGB")
        bgr = cv2.imread(img_path)
        h, w = bgr.shape[:2]

        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])[0]
        pred = result.cpu().numpy()
        da_mask = pred == road_id
        da_cov = roi_cov(da_mask)

        overlay = bgr.copy()
        overlay[da_mask] = (0.5 * overlay[da_mask].astype(np.float64) + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
        cv2.rectangle(overlay, (0, ROI_Y0), (w, ROI_Y1), (0, 255, 255), 1)

        thumb = cv2.resize(overlay, (THUMB_W, THUMB_H))
        label = f"{fname} da={da_cov:.3f} (ll 클래스 없음)"
        cv2.rectangle(thumb, (0, 0), (THUMB_W, 18), (0, 0, 0), -1)
        cv2.putText(thumb, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
        print(f"{fname}: da={da_cov:.3f}")

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
