#!/usr/bin/env python3
"""build_pseudo_label_dataset.py의 v2 - §2.12/§3에서 결정한 대로:
- 대상 프레임: dataset_diet_v2/(1027장, stride=1 dedup 상한까지 확장, §3 사용자 확인)
- da: bootstrap_v2 모델(134장 사람 라벨로만 학습, letterbox 버그/pseudo label 오염 없음)
- ll: YOLOPv2 + skeleton 정제 (기존과 동일, §2.9)
- da_final = da | ll (기존 관례 유지)
"""
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "YOLOPv2"))
from utils.utils import letterbox as yolopv2_letterbox  # noqa: E402
sys.path.insert(0, os.path.join(BASE, "scripts"))
from skeleton_polyline_utils import mask_to_polylines, polylines_to_mask  # noqa: E402

IMAGES_DIR = os.path.join(BASE, "dataset_diet_v2")
OUT_DIR = os.path.join(BASE, "pseudo_dataset_v2")
OUT_IMAGES = os.path.join(OUT_DIR, "images")
OUT_DA = os.path.join(OUT_DIR, "da_masks")
OUT_LL = os.path.join(OUT_DIR, "ll_masks")

DA_ONNX = os.path.join(BASE, "outputs", "onnx", "twinlitenetplus_small_bootstrap_v2.onnx")
YOLOPV2_WEIGHTS = os.path.join(BASE, "YOLOPv2", "data", "weights", "yolopv2.pt")

DA_W, DA_H = 640, 384
LL_REFINE_WIDTH = 8


def softmax_fg(logits_2ch):
    m = logits_2ch.max(axis=0, keepdims=True)
    e = np.exp(logits_2ch - m)
    return (e / e.sum(axis=0, keepdims=True))[1]


def infer_da(sess, bgr):
    h, w = bgr.shape[:2]
    resized = cv2.resize(bgr, (DA_W, DA_H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])
    da_out, _ = sess.run(["da", "ll"], {"images": blob})
    da_mask = np.argmax(da_out[0], axis=0).astype(np.uint8)  # bootstrap_v2 onnx는 (2,H,W) 로짓 - argmax로 이진화
    return cv2.resize(da_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def infer_ll_yolopv2(model, bgr):
    h0, w0 = bgr.shape[:2]
    img, _, _ = yolopv2_letterbox(bgr, 640, stride=32)
    inp = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    inp_t = torch.from_numpy(inp).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        [pred, anchor_grid], seg, ll = model(inp_t)
    ll_mask = torch.round(ll)[0, 0].numpy().astype(np.uint8)
    return cv2.resize(ll_mask, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)


def refine_ll(ll_mask):
    h, w = ll_mask.shape
    polylines = mask_to_polylines(ll_mask)
    if not polylines:
        return ll_mask, 0
    refined = polylines_to_mask(polylines, h, w, line_width=LL_REFINE_WIDTH).astype(bool)
    return refined, len(polylines)


def main():
    assert os.path.isfile(DA_ONNX), f"{DA_ONNX} 없음"
    for d in (OUT_IMAGES, OUT_DA, OUT_LL):
        os.makedirs(d, exist_ok=True)

    da_sess = ort.InferenceSession(DA_ONNX, providers=["CPUExecutionProvider"])
    yolopv2_model = torch.jit.load(YOLOPV2_WEIGHTS, map_location="cpu")
    yolopv2_model.eval()

    frames = sorted(os.listdir(IMAGES_DIR))
    n_ll_empty = 0
    for i, fname in enumerate(frames):
        bgr = cv2.imread(os.path.join(IMAGES_DIR, fname))

        da_mask = infer_da(da_sess, bgr)
        ll_raw = infer_ll_yolopv2(yolopv2_model, bgr)
        ll_mask, n_poly = refine_ll(ll_raw)
        if n_poly == 0:
            n_ll_empty += 1
        da_final = da_mask | ll_mask

        cv2.imwrite(os.path.join(OUT_IMAGES, fname), bgr)
        cv2.imwrite(os.path.join(OUT_DA, fname), (da_final.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(OUT_LL, fname), (ll_mask.astype(np.uint8) * 255))

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(frames)}")

    print(f"\n완료: {len(frames)}장 -> {OUT_DIR}")
    print(f"YOLOPv2가 차선을 하나도 못 찾은 프레임: {n_ll_empty}/{len(frames)}")


if __name__ == "__main__":
    main()
