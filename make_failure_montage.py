#!/usr/bin/env python3
"""triage_result.csv의 failure_candidate=True 프레임만 골라 da(파랑)/ll(빨강)
예측 오버레이를 씌운 썸네일 몽타주를 만든다 — 사람이 한 장씩 안 열어봐도
빠르게 실패 유형(글레어/노란선/그냥 빈 도로 등)을 훑어볼 수 있게."""
import csv
import math
import os

import cv2
import numpy as np
import onnxruntime as ort

BASE = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE, "dataset")
CSV_PATH = os.path.join(BASE, "triage_result.csv")
ONNX_PATH = os.path.expanduser("~/code/UMK/track_drive/track_drive/models/best.onnx")
OUT_PATH = os.path.join(BASE, "failure_montage.png")

DL_INPUT_W, DL_INPUT_H = 640, 360
DL_ROI_Y0, DL_ROI_Y1 = 250, 390
DL_FG_THRESHOLD = 0.5
DL_LL_FG_THRESHOLD = 0.7
THUMB_W, THUMB_H = 320, 240


def softmax_fg(logits_2ch):
    m = logits_2ch.max(axis=0, keepdims=True)
    e = np.exp(logits_2ch - m)
    return (e / e.sum(axis=0, keepdims=True))[1]


def main():
    rows = list(csv.DictReader(open(CSV_PATH)))
    candidates = [r for r in rows if r["failure_candidate"] == "True"]
    candidates.sort(key=lambda r: float(r["ll_cov"]))
    print(f"{len(candidates)}장 몽타주 생성")

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    thumbs = []
    for r in candidates:
        fp = os.path.join(DATASET_DIR, r["file"])
        bgr = cv2.imread(fp)
        h, w = bgr.shape[:2]
        resized = cv2.resize(bgr, (DL_INPUT_W, DL_INPUT_H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])
        da_out, ll_out = sess.run(out_names, {in_name: blob})
        da_prob = cv2.resize(softmax_fg(da_out[0]), (w, h))
        ll_prob = cv2.resize(softmax_fg(ll_out[0]), (w, h))

        overlay = bgr.copy()
        overlay[da_prob >= DL_FG_THRESHOLD] = (
            0.5 * overlay[da_prob >= DL_FG_THRESHOLD] + 0.5 * np.array([255, 100, 0])
        ).astype(np.uint8)
        overlay[ll_prob >= DL_LL_FG_THRESHOLD] = (0, 0, 255)
        cv2.rectangle(overlay, (0, DL_ROI_Y0), (w, DL_ROI_Y1), (0, 255, 255), 1)

        thumb = cv2.resize(overlay, (THUMB_W, THUMB_H))
        label = f"{r['file']} ll={r['ll_cov']} da={r['da_cov']}"
        cv2.rectangle(thumb, (0, 0), (THUMB_W, 18), (0, 0, 0), -1)
        cv2.putText(thumb, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)

    cols = 5
    rows_n = math.ceil(len(thumbs) / cols)
    grid = np.zeros((rows_n * THUMB_H, cols * THUMB_W, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * THUMB_H:(r + 1) * THUMB_H, c * THUMB_W:(c + 1) * THUMB_W] = t

    cv2.imwrite(OUT_PATH, grid)
    print("저장:", OUT_PATH)


if __name__ == "__main__":
    main()
