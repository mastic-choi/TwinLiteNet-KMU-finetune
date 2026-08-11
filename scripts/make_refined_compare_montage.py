#!/usr/bin/env python3
"""old(best.onnx) vs 기존 finetuned(brush GT) vs 신규 finetuned_refined(skeleton/polyline GT)
3-way 비교 몽타주. missing_line_result.csv 기준 "원래 잘 잡히던" 20장(전체 범위에서 고르게
샘플링) + "원래 실패하던" 10장(make_bootstrap_montage.py의 FAILURE_FRAMES 재사용)을 대상으로.

각 열: old / finetuned / finetuned_refined. 숫자는 ROI(y=250:390) 안 커버리지(%).
"""
import math
import os

import cv2
import numpy as np
import onnxruntime as ort

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE, "raw", "images_todo")
OUT_PATH = os.path.join(BASE, "outputs", "montages", "refined_compare_montage.png")

OLD_ONNX = os.path.expanduser("~/code/UMK/track_drive/track_drive/models/best.onnx")
PREV_ONNX = os.path.expanduser("~/Downloads/twinlitenetplus_small_finetuned.onnx")
NEW_ONNX = os.path.expanduser("~/Downloads/twinlitenetplus_small_finetuned_refined.onnx")

MODELS = [
    ("old", OLD_ONNX, 640, 360),
    ("finetuned(brush)", PREV_ONNX, 640, 384),
    ("finetuned(refined)", NEW_ONNX, 640, 384),
]

DA_THRESH, LL_THRESH = 0.5, 0.7
ROI_Y0, ROI_Y1 = 250, 390
THUMB_W, THUMB_H = 300, 225

# 원래 잘 잡히던 20장(missing_line_result.csv n_detected==3, 전체 구간에서 고르게 샘플링)
SUCCESS_FRAMES = [
    'frame_000024.png', 'frame_000173.png', 'frame_000268.png', 'frame_000380.png',
    'frame_000457.png', 'frame_000543.png', 'frame_000674.png', 'frame_000750.png',
    'frame_000861.png', 'frame_000968.png', 'frame_001058.png', 'frame_001156.png',
    'frame_001225.png', 'frame_001389.png', 'frame_001476.png', 'frame_001595.png',
    'frame_001680.png', 'frame_001758.png', 'frame_001864.png', 'frame_001966.png',
]

# 원래 실패하던 프레임 중 10장 (make_bootstrap_montage.py의 FAILURE_FRAMES와 동일 소스)
FAILURE_FRAMES = [
    'frame_001082.png', 'frame_001372.png', 'frame_001884.png', 'frame_002104.png',
    'frame_001604.png', 'frame_002106.png', 'frame_001973.png', 'frame_001974.png',
    'frame_000928.png', 'frame_001951.png',
]

FRAMES = [(f, "success") for f in SUCCESS_FRAMES] + [(f, "failure") for f in FAILURE_FRAMES]


def softmax_fg(logits_2ch):
    m = logits_2ch.max(axis=0, keepdims=True)
    e = np.exp(logits_2ch - m)
    return (e / e.sum(axis=0, keepdims=True))[1]


def infer_overlay(sess, bgr, model_w, model_h):
    h, w = bgr.shape[:2]
    resized = cv2.resize(bgr, (model_w, model_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])
    da_out, ll_out = sess.run([o.name for o in sess.get_outputs()], {sess.get_inputs()[0].name: blob})
    da_prob = cv2.resize(softmax_fg(da_out[0]), (w, h))
    ll_prob = cv2.resize(softmax_fg(ll_out[0]), (w, h))

    da_mask = da_prob >= DA_THRESH
    ll_mask = ll_prob >= LL_THRESH
    da_cov = float(np.count_nonzero(da_mask[ROI_Y0:ROI_Y1])) / da_mask[ROI_Y0:ROI_Y1].size
    ll_cov = float(np.count_nonzero(ll_mask[ROI_Y0:ROI_Y1])) / ll_mask[ROI_Y0:ROI_Y1].size

    overlay = bgr.copy()
    overlay[da_mask] = (0.5 * overlay[da_mask].astype(np.float64) + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
    overlay[ll_mask] = (0, 0, 255)
    cv2.rectangle(overlay, (0, ROI_Y0), (w, ROI_Y1), (0, 255, 255), 1)
    return overlay, da_cov, ll_cov


def main():
    missing = [p for _, p, _, _ in MODELS if not os.path.isfile(p)]
    if missing:
        print("다음 onnx 파일이 없습니다 — 준비되면 다시 실행하세요:")
        for p in missing:
            print(" ", p)
        return

    sessions = [(name, ort.InferenceSession(p, providers=["CPUExecutionProvider"]), w, h)
                for name, p, w, h in MODELS]

    rows = []
    for fname, kind in FRAMES:
        fp = os.path.join(IMAGES_DIR, fname)
        bgr = cv2.imread(fp)
        if bgr is None:
            print(f"이미지 없음, 스킵: {fname}")
            continue
        cells = []
        for name, sess, mw, mh in sessions:
            overlay, da_cov, ll_cov = infer_overlay(sess, bgr, mw, mh)
            thumb = cv2.resize(overlay, (THUMB_W, THUMB_H))
            label = f"{name} da={da_cov:.3f} ll={ll_cov:.3f}"
            cv2.rectangle(thumb, (0, 0), (THUMB_W, 16), (0, 0, 0), -1)
            cv2.putText(thumb, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(thumb)
        row_img = np.hstack(cells)
        tag = f"[{kind}] {fname}"
        cv2.rectangle(row_img, (0, THUMB_H - 16), (120, THUMB_H), (0, 0, 0), -1)
        cv2.putText(row_img, tag, (3, THUMB_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(row_img)

    grid = np.vstack(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, grid)
    print(f"{len(rows)}장 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
