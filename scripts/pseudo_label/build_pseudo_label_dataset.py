#!/usr/bin/env python3
"""파인튜닝용 의사 라벨(pseudo-label) 데이터셋 생성.
- da: 우리 파인튜닝 모델(twinlitenetplus_small_finetuned_refined.onnx) 예측 그대로 사용
  (bootstrap GT로 직접 검증한 결과 old/DINO/SAM/SegFormer/Mask2Former 등 그 무엇보다도
  정확했음 — IoU 0.852, PROGRESS.md 참고)
- ll: YOLOPv2(BDD100K pretrained) 예측을 skeleton화 -> polyline 단순화 -> 고정 두께
  재래스터화해서 사용 (bootstrap GT 대비 IoU 0.412로 우리 모델(0.055)보다 압도적으로
  정확했음, SAM 없이 원본 그대로가 제일 좋았음)
- da는 ll을 합집합으로 포함(차선은 도로 위에 있으므로) — 기존 export_cvat_bootstrap.py 관례.

대상: raw/images_todo/ 전체(470장) — CVAT 라벨링 진행 상태와 무관하게 완전 자동 생성.
출력: pseudo_dataset/images, da_masks, ll_masks (bootstrap/ 폴더와 동일 구조 -> Colab
노트북에 그대로 재사용 가능)
"""
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "YOLOPv2"))
from utils.utils import letterbox as yolopv2_letterbox  # noqa: E402

from skeleton_polyline_utils import mask_to_polylines, polylines_to_mask

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGES_DIR = os.path.join(BASE, "raw", "images_todo")
OUT_DIR = os.path.join(BASE, "pseudo_dataset")
OUT_IMAGES = os.path.join(OUT_DIR, "images")
OUT_DA = os.path.join(OUT_DIR, "da_masks")
OUT_LL = os.path.join(OUT_DIR, "ll_masks")

DA_ONNX = os.path.expanduser("~/Downloads/twinlitenetplus_small_finetuned_refined.onnx")
YOLOPV2_WEIGHTS = os.path.join(BASE, "YOLOPv2", "data", "weights", "yolopv2.pt")

DA_W, DA_H = 640, 384
DA_THRESH = 0.5
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
    da_out, _ = sess.run([o.name for o in sess.get_outputs()], {sess.get_inputs()[0].name: blob})
    da_prob = cv2.resize(softmax_fg(da_out[0]), (w, h))
    return da_prob >= DA_THRESH


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
    for d in (OUT_IMAGES, OUT_DA, OUT_LL):
        os.makedirs(d, exist_ok=True)

    da_sess = ort.InferenceSession(DA_ONNX, providers=["CPUExecutionProvider"])
    yolopv2_model = torch.jit.load(YOLOPV2_WEIGHTS, map_location="cpu")
    yolopv2_model.eval()

    frames = sorted(os.listdir(IMAGES_DIR))
    n_ll_empty = 0
    for i, fname in enumerate(frames):
        img_path = os.path.join(IMAGES_DIR, fname)
        bgr = cv2.imread(img_path)

        da_mask = infer_da(da_sess, bgr)
        ll_raw = infer_ll_yolopv2(yolopv2_model, bgr)
        ll_mask, n_poly = refine_ll(ll_raw)
        if n_poly == 0:
            n_ll_empty += 1
        da_final = da_mask | ll_mask

        cv2.imwrite(os.path.join(OUT_IMAGES, fname), bgr)
        cv2.imwrite(os.path.join(OUT_DA, fname), (da_final.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(OUT_LL, fname), (ll_mask.astype(np.uint8) * 255))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(frames)}")

    print(f"\n완료: {len(frames)}장 -> {OUT_DIR}")
    print(f"YOLOPv2가 차선을 하나도 못 찾은 프레임: {n_ll_empty}/{len(frames)} (사람이 확인 권장)")


if __name__ == "__main__":
    main()
