#!/usr/bin/env python3
"""lap_005(신규 raw 프레임, new_raw_frames/의 lap005_ 접두어분, 2734장)에 대해
images/da_masks/ll_masks 구조의 pseudo-label 데이터셋을 만든다.

- da: 6-앙상블(신규 부트스트랩 5개, outputs/ensemble_bootstrap_v1 + 기존 배포모델
  outputs/models/best.onnx) 확률 평균 소프트보팅, §2.20에서 확정한 방식
- ll: YOLOPv2 + skeleton 정제 (기존과 동일, §2.9)
- da_final = da | ll (기존 관례 유지)

da_trust_review.ipynb의 사람 리뷰(신뢰/비신뢰/SAM보정)와는 별개 - 이건 빠르게
전체 자동 라벨을 만들어두는 용도(리뷰 결과와 나중에 대조/병합 가능).
"""
import os
import sys
from argparse import Namespace

import cv2
import numpy as np
import onnxruntime as ort
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "_TwinLiteNetPlus_ref"))
from model.model import TwinLiteNetPlus  # noqa: E402
sys.path.insert(0, os.path.join(BASE, "YOLOPv2"))
from utils.utils import letterbox as yolopv2_letterbox  # noqa: E402
sys.path.insert(0, os.path.join(BASE, "scripts", "pseudo_label"))
from skeleton_polyline_utils import mask_to_polylines, polylines_to_mask  # noqa: E402

IMAGES_DIR = os.path.join(BASE, "new_raw_frames")
PREFIX = "lap005_"
OUT_DIR = os.path.join(BASE, "pseudo_dataset_lap005")
OUT_IMAGES = os.path.join(OUT_DIR, "images")
OUT_DA = os.path.join(OUT_DIR, "da_masks")
OUT_LL = os.path.join(OUT_DIR, "ll_masks")

ENSEMBLE_DIR = os.path.join(BASE, "outputs", "ensemble_bootstrap_v1")
DEPLOYED_ONNX = os.path.join(BASE, "outputs", "models", "best.onnx")
YOLOPV2_WEIGHTS = os.path.join(BASE, "YOLOPv2", "data", "weights", "yolopv2.pt")
N_SEEDS = 5
CONFIG = "medium"
DA_W, DA_H = 640, 384
LL_REFINE_WIDTH = 8


def softmax_fg_np(l):
    m = l.max(axis=0, keepdims=True)
    e = np.exp(l - m)
    return (e / e.sum(axis=0, keepdims=True))[1]


def to_blob(img0):
    r = cv2.resize(img0, (DA_W, DA_H))
    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])


def infer_da_ensemble(torch_models, onnx_sess, img0):
    h, w = img0.shape[:2]
    blob_np = to_blob(img0)
    blob_t = torch.from_numpy(blob_np).float()
    probs = []
    for m in torch_models:
        with torch.no_grad():
            out_da, _ = m(blob_t)
        probs.append(torch.softmax(out_da, dim=1)[0, 1].numpy())
    da_out, _ = onnx_sess.run(["da", "ll"], {"images": blob_np})
    probs.append(softmax_fg_np(da_out[0]))
    avg = np.mean(probs, axis=0)
    return cv2.resize(avg, (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5


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
    torch_models = []
    for seed in range(N_SEEDS):
        m = TwinLiteNetPlus(Namespace(config=CONFIG))
        state = torch.load(os.path.join(ENSEMBLE_DIR, f"seed{seed}", "best.pth"), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        m.load_state_dict(state)
        m.eval()
        torch_models.append(m)
    onnx_sess = ort.InferenceSession(DEPLOYED_ONNX, providers=["CPUExecutionProvider"])
    print(f"da 6-앙상블 로드 완료 ({N_SEEDS}+1개)")

    yolopv2_model = torch.jit.load(YOLOPV2_WEIGHTS, map_location="cpu")
    yolopv2_model.eval()
    print("YOLOPv2 로드 완료")

    for d in (OUT_IMAGES, OUT_DA, OUT_LL):
        os.makedirs(d, exist_ok=True)

    frames = sorted(f for f in os.listdir(IMAGES_DIR) if f.startswith(PREFIX))
    print(f"대상 {len(frames)}장 (new_raw_frames/{PREFIX}*)")

    n_ll_empty = 0
    for i, fname in enumerate(frames):
        bgr = cv2.imread(os.path.join(IMAGES_DIR, fname))

        da_mask = infer_da_ensemble(torch_models, onnx_sess, bgr)
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
