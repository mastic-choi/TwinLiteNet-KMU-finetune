#!/usr/bin/env python3
"""dataset/ 전체 프레임에 best.onnx(원조 TwinLiteNet)로 ll 추론 -> 각 차선(연결 성분)의
곡률을 대략적으로 측정해서 "커브 구간" 후보를 찾는다. triage_dataset.py와 동일한 모델/
전처리/ROI를 씀(직접 비교 가능하게).

곡률 측정: ll 마스크에서 노이즈 아닌 연결 성분(차선 하나)마다 위쪽 1/3과 아래쪽 1/3 구간의
평균 x위치로 기울기(dx/dy)를 각각 구하고, 그 차이(각도 차이)를 곡률 점수로 씀 — 직선이면
위/아래 기울기가 비슷하고, 커브면 크게 달라짐.
"""
import csv
import glob
import math
import os

import cv2
import numpy as np
import onnxruntime as ort

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
ONNX_PATH = os.path.expanduser("~/code/UMK/track_drive/track_drive/models/best.onnx")
OUT_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "curve_result.csv")

DL_INPUT_W, DL_INPUT_H = 640, 360
DL_ROI_Y0, DL_ROI_Y1 = 250, 390
DL_LL_FG_THRESHOLD = 0.7
MIN_COMPONENT_AREA = 80


def softmax_fg(logits_2ch):
    m = logits_2ch.max(axis=0, keepdims=True)
    e = np.exp(logits_2ch - m)
    return (e / e.sum(axis=0, keepdims=True))[1]


def component_curve_score(comp_mask):
    ys, xs = np.where(comp_mask)
    if len(ys) < 10:
        return 0.0
    y0, y1 = ys.min(), ys.max()
    if y1 - y0 < 15:
        return 0.0
    third = (y1 - y0) / 3.0
    top_band = ys <= y0 + third
    bot_band = ys >= y1 - third
    if top_band.sum() < 3 or bot_band.sum() < 3:
        return 0.0

    def slope(band_ys, band_xs):
        # x = a*y + b 선형 회귀 기울기
        A = np.vstack([band_ys, np.ones_like(band_ys)]).T
        a, _ = np.linalg.lstsq(A.astype(np.float64), band_xs.astype(np.float64), rcond=None)[0]
        return a

    a_top = slope(ys[top_band], xs[top_band])
    a_bot = slope(ys[bot_band], xs[bot_band])
    # 기울기(dx/dy) 차이를 각도 차이(도)로 변환
    ang_top = math.degrees(math.atan(a_top))
    ang_bot = math.degrees(math.atan(a_bot))
    return abs(ang_top - ang_bot)


def main():
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.png")))
    assert files, f"{DATASET_DIR}에 png 없음"
    print(f"{len(files)}장 처리 시작 — 모델: {ONNX_PATH}")

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    rows = []
    for i, fp in enumerate(files):
        bgr = cv2.imread(fp)
        h, w = bgr.shape[:2]
        resized = cv2.resize(bgr, (DL_INPUT_W, DL_INPUT_H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])
        _, ll_out = sess.run(out_names, {in_name: blob})
        ll_prob = cv2.resize(softmax_fg(ll_out[0]), (w, h))

        y0, y1 = max(0, min(DL_ROI_Y0, h)), max(0, min(DL_ROI_Y1, h))
        ll_mask = (ll_prob[y0:y1] >= DL_LL_FG_THRESHOLD).astype(np.uint8)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(ll_mask, connectivity=8)
        best_score = 0.0
        for c in range(1, n):
            if stats[c, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
                continue
            score = component_curve_score(labels == c)
            best_score = max(best_score, score)

        rows.append({"file": os.path.basename(fp), "curve_score": round(best_score, 2)})
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)}")

    rows.sort(key=lambda r: -r["curve_score"])
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n총 {len(rows)}장, curve_score>15(커브 후보): {sum(1 for r in rows if r['curve_score']>15)}장")
    print(f"CSV 저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
