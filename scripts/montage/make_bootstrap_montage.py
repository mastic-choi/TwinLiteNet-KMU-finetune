#!/usr/bin/env python3
"""부트스트랩 파인튜닝된 twinlitenetplus_small_finetuned.onnx로 원래 실패했던
23장(triage_result.csv의 failure_candidate)을 다시 돌려서 개선됐는지 확인하는
몽타주를 만든다. 원조 best.onnx(640x360)와 달리 TwinLiteNet+는 640x384 입력."""
import glob
import math
import os

import cv2
import numpy as np
import onnxruntime as ort

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGES_DIR = os.path.join(BASE, "raw", "images_todo")
ONNX_PATH = os.path.expanduser("~/Downloads/twinlitenetplus_small_finetuned.onnx")
OUT_PATH = os.path.join(BASE, "bootstrap_montage.png")

MODEL_W, MODEL_H = 640, 384   # TwinLiteNet+ 학습 해상도 (원조 TwinLiteNet 640x360과 다름)
DA_THRESH = 0.5
LL_THRESH = 0.7
DL_ROI_Y0, DL_ROI_Y1 = 250, 390
THUMB_W, THUMB_H = 320, 240

FAILURE_FRAMES = [
    'frame_001082.png', 'frame_001372.png', 'frame_001884.png', 'frame_002104.png',
    'frame_001604.png', 'frame_002106.png', 'frame_001973.png', 'frame_001974.png',
    'frame_000928.png', 'frame_001951.png', 'frame_000887.png', 'frame_001693.png',
    'frame_001915.png', 'frame_000924.png', 'frame_001902.png', 'frame_000853.png',
    'frame_000926.png', 'frame_001444.png', 'frame_000397.png', 'frame_001914.png',
    'frame_001637.png', 'frame_000850.png', 'frame_001952.png',
]


def softmax_fg(logits_2ch):
    m = logits_2ch.max(axis=0, keepdims=True)
    e = np.exp(logits_2ch - m)
    return (e / e.sum(axis=0, keepdims=True))[1]


def main():
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    thumbs = []
    for f in FAILURE_FRAMES:
        fp = os.path.join(IMAGES_DIR, f)
        bgr = cv2.imread(fp)
        h, w = bgr.shape[:2]
        resized = cv2.resize(bgr, (MODEL_W, MODEL_H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])
        da_out, ll_out = sess.run(out_names, {in_name: blob})
        da_prob = cv2.resize(softmax_fg(da_out[0]), (w, h))
        ll_prob = cv2.resize(softmax_fg(ll_out[0]), (w, h))

        da_mask = da_prob >= DA_THRESH
        ll_mask = ll_prob >= LL_THRESH
        da_cov = float(np.count_nonzero(da_mask[DL_ROI_Y0:DL_ROI_Y1])) / da_mask[DL_ROI_Y0:DL_ROI_Y1].size
        ll_cov = float(np.count_nonzero(ll_mask[DL_ROI_Y0:DL_ROI_Y1])) / ll_mask[DL_ROI_Y0:DL_ROI_Y1].size

        overlay = bgr.copy()
        overlay[da_mask] = (0.5 * overlay[da_mask].astype(np.float64) + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
        overlay[ll_mask] = (0, 0, 255)
        cv2.rectangle(overlay, (0, DL_ROI_Y0), (w, DL_ROI_Y1), (0, 255, 255), 1)

        thumb = cv2.resize(overlay, (THUMB_W, THUMB_H))
        label = f"{f} da={da_cov:.3f} ll={ll_cov:.3f}"
        cv2.rectangle(thumb, (0, 0), (THUMB_W, 18), (0, 0, 0), -1)
        cv2.putText(thumb, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append((f, da_cov, ll_cov, thumb))

    cols = 5
    rows_n = math.ceil(len(thumbs) / cols)
    grid = np.zeros((rows_n * THUMB_H, cols * THUMB_W, 3), dtype=np.uint8)
    for i, (_, _, _, t) in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * THUMB_H:(r + 1) * THUMB_H, c * THUMB_W:(c + 1) * THUMB_W] = t
    cv2.imwrite(OUT_PATH, grid)

    print(f"{len(thumbs)}장 처리 완료 -> {OUT_PATH}")
    n_still_fail = sum(1 for _, da, ll, _ in thumbs if ll < 0.005)
    avg_ll = sum(ll for _, _, ll, _ in thumbs) / len(thumbs)
    print(f"ll_cov < 0.005(여전히 미검출): {n_still_fail}/{len(thumbs)}장")
    print(f"평균 ll_cov: {avg_ll:.4f}")


if __name__ == "__main__":
    main()
