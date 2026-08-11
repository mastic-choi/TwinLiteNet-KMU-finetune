#!/usr/bin/env python3
"""dataset/ 전체 프레임에 best.onnx의 da(파랑)/ll(빨강) 예측 마스크를 오버레이해서
dataset_overlay/에 같은 파일명으로 저장한다 — 전체를 사람이 훑어보며 실패/성공을
직접 판단할 때 쓰는 용도."""
import glob
import os

import cv2
import numpy as np
import onnxruntime as ort

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE, "dataset")
OUT_DIR = os.path.join(BASE, "dataset_overlay")
ONNX_PATH = os.path.expanduser("~/code/UMK/track_drive/track_drive/models/best.onnx")

DL_INPUT_W, DL_INPUT_H = 640, 360
DL_ROI_Y0, DL_ROI_Y1 = 250, 390
DL_FG_THRESHOLD = 0.5
DL_LL_FG_THRESHOLD = 0.7


def softmax_fg(logits_2ch):
    m = logits_2ch.max(axis=0, keepdims=True)
    e = np.exp(logits_2ch - m)
    return (e / e.sum(axis=0, keepdims=True))[1]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.png")))
    assert files, f"{DATASET_DIR}에 png 없음"
    print(f"{len(files)}장 오버레이 생성 시작 -> {OUT_DIR}")

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    for i, fp in enumerate(files):
        bgr = cv2.imread(fp)
        h, w = bgr.shape[:2]

        resized = cv2.resize(bgr, (DL_INPUT_W, DL_INPUT_H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])
        da_out, ll_out = sess.run(out_names, {in_name: blob})
        da_prob = cv2.resize(softmax_fg(da_out[0]), (w, h))
        ll_prob = cv2.resize(softmax_fg(ll_out[0]), (w, h))

        da_mask = da_prob >= DL_FG_THRESHOLD
        ll_mask = ll_prob >= DL_LL_FG_THRESHOLD
        da_cov = float(np.count_nonzero(da_mask[DL_ROI_Y0:DL_ROI_Y1])) / da_mask[DL_ROI_Y0:DL_ROI_Y1].size
        ll_cov = float(np.count_nonzero(ll_mask[DL_ROI_Y0:DL_ROI_Y1])) / ll_mask[DL_ROI_Y0:DL_ROI_Y1].size

        overlay = bgr.copy()
        overlay[da_mask] = (0.5 * overlay[da_mask] + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
        overlay[ll_mask] = (0, 0, 255)
        cv2.rectangle(overlay, (0, DL_ROI_Y0), (w, DL_ROI_Y1), (0, 255, 255), 1)

        label = f"da={da_cov:.3f} ll={ll_cov:.3f}"
        cv2.rectangle(overlay, (0, 0), (200, 18), (0, 0, 0), -1)
        cv2.putText(overlay, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        out_fp = os.path.join(OUT_DIR, os.path.basename(fp))
        cv2.imwrite(out_fp, overlay)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)}")

    print("완료:", OUT_DIR)


if __name__ == "__main__":
    main()
