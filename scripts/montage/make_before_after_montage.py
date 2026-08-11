#!/usr/bin/env python3
"""실패 후보 + 정상 케이스를 섞어서, 예전(best.onnx, dataset_overlay/ 재사용) vs
지금(부트스트랩 파인튜닝 twinlitenetplus_small_finetuned.onnx) 결과를 나란히
붙여서 비교 몽타주를 만든다."""
import math
import os

import cv2
import numpy as np
import onnxruntime as ort

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGES_DIR = os.path.join(BASE, "raw", "images_todo")
OVERLAY_DIR = os.path.join(BASE, "dataset_overlay")  # 예전 best.onnx 결과 재사용
ONNX_PATH = os.path.expanduser("~/Downloads/twinlitenetplus_small_finetuned.onnx")
OUT_PATH = os.path.join(BASE, "before_after_montage.png")

MODEL_W, MODEL_H = 640, 384
DA_THRESH = 0.5
LL_THRESH = 0.7
DL_ROI_Y0, DL_ROI_Y1 = 250, 390
PANE_W, PANE_H = 280, 210  # old/new 한 쪽 크기

FAIL_FRAMES = [
    'frame_001082.png', 'frame_001372.png', 'frame_001884.png', 'frame_002104.png',
    'frame_001604.png', 'frame_002106.png', 'frame_001973.png', 'frame_001974.png',
    'frame_000928.png', 'frame_001951.png', 'frame_000887.png', 'frame_001693.png',
    'frame_001915.png', 'frame_000924.png', 'frame_001902.png', 'frame_000853.png',
]
GOOD_FRAMES = [
    'frame_001495.png', 'frame_000855.png', 'frame_001200.png', 'frame_001108.png',
    'frame_000627.png', 'frame_001488.png', 'frame_001648.png', 'frame_001935.png',
]
ALL_FRAMES = FAIL_FRAMES + GOOD_FRAMES


def softmax_fg(logits_2ch):
    m = logits_2ch.max(axis=0, keepdims=True)
    e = np.exp(logits_2ch - m)
    return (e / e.sum(axis=0, keepdims=True))[1]


def main():
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    tiles = []
    for f in ALL_FRAMES:
        old_img = cv2.imread(os.path.join(OVERLAY_DIR, f))
        old_pane = cv2.resize(old_img, (PANE_W, PANE_H))
        cv2.rectangle(old_pane, (0, 0), (PANE_W, 16), (0, 0, 0), -1)
        cv2.putText(old_pane, "OLD (best.onnx)", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        bgr = cv2.imread(os.path.join(IMAGES_DIR, f))
        h, w = bgr.shape[:2]
        resized = cv2.resize(bgr, (MODEL_W, MODEL_H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])
        da_out, ll_out = sess.run(out_names, {in_name: blob})
        da_prob = cv2.resize(softmax_fg(da_out[0]), (w, h))
        ll_prob = cv2.resize(softmax_fg(ll_out[0]), (w, h))
        da_mask = da_prob >= DA_THRESH
        ll_mask = ll_prob >= LL_THRESH
        ll_cov = float(np.count_nonzero(ll_mask[DL_ROI_Y0:DL_ROI_Y1])) / ll_mask[DL_ROI_Y0:DL_ROI_Y1].size

        new_overlay = bgr.copy()
        new_overlay[da_mask] = (0.5 * new_overlay[da_mask].astype(np.float64) + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
        new_overlay[ll_mask] = (0, 0, 255)
        new_pane = cv2.resize(new_overlay, (PANE_W, PANE_H))
        cv2.rectangle(new_pane, (0, 0), (PANE_W, 16), (0, 0, 0), -1)
        cv2.putText(new_pane, f"NEW (bootstrap) ll={ll_cov:.3f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        pair = np.hstack([old_pane, new_pane])
        cv2.rectangle(pair, (0, PANE_H - 14), (pair.shape[1], PANE_H), (0, 0, 0), -1)
        tag = "FAIL후보" if f in FAIL_FRAMES else "정상"
        cv2.putText(pair, f"{f} [{tag}]", (3, PANE_H - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(pair)

    cols = 3
    rows_n = math.ceil(len(tiles) / cols)
    tile_h, tile_w = tiles[0].shape[:2]
    grid = np.zeros((rows_n * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = t
    cv2.imwrite(OUT_PATH, grid)
    print("완료:", OUT_PATH, f"({len(tiles)}쌍, 실패후보 {len(FAIL_FRAMES)} + 정상 {len(GOOD_FRAMES)})")


if __name__ == "__main__":
    main()
