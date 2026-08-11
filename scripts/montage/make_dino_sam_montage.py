#!/usr/bin/env python3
"""Grounding DINO(zero-shot 텍스트 프롬프트 "lane line."/"road.")로 박스를 찾고,
그 박스를 SAM2 프롬프트로 줘서 마스크를 만드는 방식 20장 몽타주.
"""
import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import SAM

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGES_DIR = os.path.join(BASE, "raw", "images_todo")
OUT_PATH = os.path.join(BASE, "outputs", "montages", "dino_sam_montage.png")
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


def detect(processor, dino, image, text):
    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = dino(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.2, text_threshold=0.2,
        target_sizes=[image.size[::-1]],
    )
    return results[0]


def sam_mask_from_box(sam, img_path, box, shape):
    res = sam(img_path, bboxes=[box], verbose=False)
    if res[0].masks is None:
        return np.zeros(shape, dtype=bool)
    return res[0].masks.data[0].cpu().numpy().astype(bool)


def main():
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)
    sam = SAM("sam2.1_b.pt")

    thumbs = []
    for fname in FRAMES:
        img_path = os.path.join(IMAGES_DIR, fname)
        bgr = cv2.imread(img_path)
        h, w = bgr.shape[:2]
        image = Image.open(img_path).convert("RGB")

        da_mask = np.zeros((h, w), dtype=bool)
        ll_mask = np.zeros((h, w), dtype=bool)

        road = detect(processor, dino, image, "road.")
        if len(road["boxes"]) > 0:
            box = road["boxes"][road["scores"].argmax()].tolist()
            da_mask = sam_mask_from_box(sam, img_path, box, (h, w))

        lane = detect(processor, dino, image, "lane line.")
        if len(lane["boxes"]) > 0:
            box = lane["boxes"][lane["scores"].argmax()].tolist()
            ll_mask = sam_mask_from_box(sam, img_path, box, (h, w))

        da_cov, ll_cov = roi_cov(da_mask), roi_cov(ll_mask)

        overlay = bgr.copy()
        overlay[da_mask] = (0.5 * overlay[da_mask].astype(np.float64) + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
        overlay[ll_mask] = (0, 0, 255)
        cv2.rectangle(overlay, (0, ROI_Y0), (w, ROI_Y1), (0, 255, 255), 1)

        thumb = cv2.resize(overlay, (THUMB_W, THUMB_H))
        label = f"{fname} da={da_cov:.3f} ll={ll_cov:.3f}"
        cv2.rectangle(thumb, (0, 0), (THUMB_W, 18), (0, 0, 0), -1)
        cv2.putText(thumb, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
        print(f"{fname}: da={da_cov:.3f} ll={ll_cov:.3f}")

    cols = 5
    import math
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
