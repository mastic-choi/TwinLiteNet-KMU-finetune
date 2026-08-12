#!/usr/bin/env python3
"""bootstrap_v2 156장 전체를 6-앙상블(신규 부트스트랩 5개 + 기존 배포모델 medium_v2)
da 소프트보팅 결과로 한 장의 contact sheet에 표시. TP=초록/FP=빨강/FN=파랑,
ll은 생략(썸네일이 너무 빽빽해져서) - da만.
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

ENSEMBLE_DIR = os.path.join(BASE, "outputs", "ensemble_bootstrap_v1")
N_SEEDS = 5
CONFIG = "medium"
IN_W, IN_H = 640, 384
DEPLOYED_ONNX = os.path.join(BASE, "outputs", "models", "best.onnx")

GT_DIR = os.path.join(BASE, "bootstrap_v2")
GT_IMAGES = os.path.join(GT_DIR, "images")
GT_DA = os.path.join(GT_DIR, "da_masks")

OUT_PATH = os.path.join(BASE, "outputs", "montages", "ensemble6_da_full156.png")
THUMB_W, THUMB_H = 180, 135
COLS = 12


def softmax_fg_np(l):
    m = l.max(axis=0, keepdims=True)
    e = np.exp(l - m)
    return (e / e.sum(axis=0, keepdims=True))[1]


def to_blob(img0):
    r = cv2.resize(img0, (IN_W, IN_H))
    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])


def overlay(img0, pred, gt):
    vis = img0.copy()
    tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt
    vis[tp] = (0.5 * vis[tp].astype(np.float64) + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
    vis[fp] = (0.5 * vis[fp].astype(np.float64) + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    vis[fn] = (0.5 * vis[fn].astype(np.float64) + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    return vis


def main():
    torch_models = []
    for seed in range(N_SEEDS):
        m = TwinLiteNetPlus(Namespace(config=CONFIG))
        s = torch.load(f"{ENSEMBLE_DIR}/seed{seed}/best.pth", map_location="cpu")
        if isinstance(s, dict) and "state_dict" in s:
            s = s["state_dict"]
        m.load_state_dict(s)
        m.eval()
        torch_models.append(m)
    sess = ort.InferenceSession(DEPLOYED_ONNX, providers=["CPUExecutionProvider"])
    print(f"모델 {N_SEEDS}+1개 로드 완료")

    names = sorted(os.path.splitext(f)[0] for f in os.listdir(GT_IMAGES))
    print(f"대상: {len(names)}장 전체")

    thumbs = []
    ious = []
    for i, n in enumerate(names):
        img0 = cv2.imread(os.path.join(GT_IMAGES, n + ".png"))
        h, w = img0.shape[:2]
        gt = cv2.imread(os.path.join(GT_DA, n + ".png"), cv2.IMREAD_GRAYSCALE) > 127

        blob_np = to_blob(img0)
        blob_t = torch.from_numpy(blob_np).float()
        probs = []
        for m in torch_models:
            with torch.no_grad():
                out_da, _ = m(blob_t)
            probs.append(torch.softmax(out_da, dim=1)[0, 1].numpy())
        da_out, _ = sess.run(["da", "ll"], {"images": blob_np})
        probs.append(softmax_fg_np(da_out[0]))

        avg = np.mean(probs, axis=0)
        pred = cv2.resize(avg, (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5

        iou = (pred & gt).sum() / max((pred | gt).sum(), 1)
        ious.append(iou)

        vis = overlay(img0, pred, gt)
        thumb = cv2.resize(vis, (THUMB_W, THUMB_H))
        cv2.rectangle(thumb, (0, 0), (THUMB_W, 12), (0, 0, 0), -1)
        cv2.putText(thumb, f"{n[-4:]} {iou:.2f}", (1, 9), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        thumbs.append(thumb)

        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(names)}")

    rows = (len(thumbs) + COLS - 1) // COLS
    grid = np.full((rows * THUMB_H, COLS * THUMB_W, 3), 40, dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, COLS)
        grid[r * THUMB_H:(r + 1) * THUMB_H, c * THUMB_W:(c + 1) * THUMB_W] = t

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, grid)
    print(f"\n평균 da IoU (156장 전체, 6-앙상블) = {np.mean(ious):.3f}")
    print(f"저장: {OUT_PATH} ({grid.shape[1]}x{grid.shape[0]}px, 초록=TP 빨강=FP 파랑=FN)")


if __name__ == "__main__":
    main()
