#!/usr/bin/env python3
"""
dataset/ 안 원본 프레임을 실차 배포 중인 best.onnx(원조 TwinLiteNet)로 돌려서
da_cov/ll_cov/밝기 지표를 계산하고, 실패 후보(라벨링 우선순위) 상위 프레임을
CSV로 뽑아낸다.

기준값은 track_drive/track_drive/config.py, perception/dl_lane.py에서 실제
실차 로직이 쓰는 값을 그대로 가져옴:
  - DL_INPUT_W/H = 640/360      (perception/dl_lane.py)
  - DL_ROI_Y0/Y1 = 250/390      (원본 640x480 프레임 절대 좌표, config.py)
  - DL_FG_THRESHOLD = 0.5       (da 이진화 임계값, config.py)
  - DL_LL_FG_THRESHOLD = 0.7    (ll 이진화 임계값, config.py)
  - DL_LL_SANITY_MIN_RATIO = 0.005  (이 미만이면 실차에서도 "차선 안 보임"으로 무효 처리, config.py)

BEV 워프는 재현하지 않음(원근 상태 ROI 기준 coverage) — 절대값보다 "프레임 간
상대 비교"용 지표이므로 실차 디버그 창의 ll_cov와 완전히 같은 숫자는 아님.
"""
import csv
import glob
import os

import cv2
import numpy as np
import onnxruntime as ort

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
ONNX_PATH = os.path.expanduser(
    "~/code/UMK/track_drive/track_drive/models/best.onnx"
)
OUT_CSV = os.path.join(os.path.dirname(__file__), "triage_result.csv")

DL_INPUT_W, DL_INPUT_H = 640, 360
DL_ROI_Y0, DL_ROI_Y1 = 250, 390
DL_FG_THRESHOLD = 0.5
DL_LL_FG_THRESHOLD = 0.7
DL_LL_SANITY_MIN_RATIO = 0.005

OVEREXPOSED_VALUE = 240


def softmax_fg(logits_2ch):
    m = logits_2ch.max(axis=0, keepdims=True)
    e = np.exp(logits_2ch - m)
    return (e / e.sum(axis=0, keepdims=True))[1]


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
        da_out, ll_out = sess.run(out_names, {in_name: blob})
        da_prob = cv2.resize(softmax_fg(da_out[0]), (w, h))
        ll_prob = cv2.resize(softmax_fg(ll_out[0]), (w, h))

        y0, y1 = max(0, min(DL_ROI_Y0, h)), max(0, min(DL_ROI_Y1, h))
        da_roi = da_prob[y0:y1]
        ll_roi = ll_prob[y0:y1]

        da_mask = da_roi >= DL_FG_THRESHOLD
        ll_mask = ll_roi >= DL_LL_FG_THRESHOLD
        da_cov = float(np.count_nonzero(da_mask)) / da_mask.size if da_mask.size else 0.0
        ll_cov = float(np.count_nonzero(ll_mask)) / ll_mask.size if ll_mask.size else 0.0

        gray_roi = cv2.cvtColor(bgr[y0:y1], cv2.COLOR_BGR2GRAY)
        brightness_mean = float(gray_roi.mean())
        overexposed_ratio = float(np.count_nonzero(gray_roi >= OVEREXPOSED_VALUE)) / gray_roi.size

        ll_fail = ll_cov < DL_LL_SANITY_MIN_RATIO
        da_fail = da_cov < 0.05  # 실차 DL_DA_MIN_COMPONENT_AREA와 정확히 등가는 아닌 러프 임계값
        glare_suspect = overexposed_ratio > 0.15 or brightness_mean > 200

        rows.append({
            "file": os.path.basename(fp),
            "da_cov": round(da_cov, 4),
            "ll_cov": round(ll_cov, 4),
            "brightness_mean": round(brightness_mean, 1),
            "overexposed_ratio": round(overexposed_ratio, 4),
            "ll_fail": ll_fail,
            "da_fail": da_fail,
            "glare_suspect": glare_suspect,
            "failure_candidate": ll_fail or da_fail or glare_suspect,
        })

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)}")

    rows.sort(key=lambda r: r["ll_cov"])
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_fail = sum(r["failure_candidate"] for r in rows)
    n_ll_fail = sum(r["ll_fail"] for r in rows)
    n_da_fail = sum(r["da_fail"] for r in rows)
    n_glare = sum(r["glare_suspect"] for r in rows)
    print(f"\n총 {len(rows)}장")
    print(f"실패 후보(합집합): {n_fail}장 ({n_fail / len(rows):.1%})")
    print(f"  - ll_cov < {DL_LL_SANITY_MIN_RATIO}: {n_ll_fail}장")
    print(f"  - da_cov < 0.05: {n_da_fail}장")
    print(f"  - 글레어/과다노출 의심: {n_glare}장")
    print(f"CSV 저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
