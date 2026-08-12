#!/usr/bin/env python3
"""새로 수집한 raw 프레임(GT 없음)에 6-앙상블(신규 부트스트랩 5개 + 기존 배포모델)
da 소프트보팅 결과를 오버레이로 미리 보여준다. TP/FP/FN 색 구분은 GT가 있어야
가능하니 여기선 단순히 예측된 da 영역만 초록으로 덧칠(da_trust_review.ipynb의
overlay_da()와 동일한 스타일).

사용: python scripts/eval/preview_ensemble_on_new_data.py <입력폴더> [--n 24] [--out 경로]
"""
import argparse
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
DEPLOYED_ONNX = os.path.join(BASE, "outputs", "models", "best.onnx")
N_SEEDS = 5
CONFIG = "medium"
IN_W, IN_H = 640, 384
THUMB_W, THUMB_H = 220, 165


def softmax_fg_np(l):
    m = l.max(axis=0, keepdims=True)
    e = np.exp(l - m)
    return (e / e.sum(axis=0, keepdims=True))[1]


def to_blob(img0):
    r = cv2.resize(img0, (IN_W, IN_H))
    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])


def overlay_da(img_bgr, mask):
    vis = img_bgr.copy()
    vis[mask] = (0.55 * vis[mask].astype(np.float64) + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("--n", type=int, default=24, help="몽타주에 보여줄 프레임 수(전체에서 균등 샘플링)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cols", type=int, default=6)
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        BASE, "outputs", "montages",
        f"ensemble_preview_{os.path.basename(args.input_dir.rstrip('/'))}.png")

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
    print(f"6-앙상블 로드 완료 ({N_SEEDS}+1개)")

    files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    assert files, f"{args.input_dir}에 이미지 없음"
    n = min(args.n, len(files))
    idxs = np.linspace(0, len(files) - 1, n).astype(int)
    sample = [files[i] for i in idxs]
    print(f"전체 {len(files)}장 중 {n}장 균등 샘플링")

    thumbs = []
    for i, fname in enumerate(sample):
        img = cv2.imread(os.path.join(args.input_dir, fname))
        h, w = img.shape[:2]
        blob_np = to_blob(img)
        blob_t = torch.from_numpy(blob_np).float()
        probs = []
        for m in torch_models:
            with torch.no_grad():
                out_da, _ = m(blob_t)
            probs.append(torch.softmax(out_da, dim=1)[0, 1].numpy())
        da_out, _ = onnx_sess.run(["da", "ll"], {"images": blob_np})
        probs.append(softmax_fg_np(da_out[0]))
        avg = np.mean(probs, axis=0)
        pred = cv2.resize(avg, (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5

        vis = overlay_da(img, pred)
        thumb = cv2.resize(vis, (THUMB_W, THUMB_H))
        cv2.rectangle(thumb, (0, 0), (THUMB_W, 16), (0, 0, 0), -1)
        cv2.putText(thumb, fname, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        thumbs.append(thumb)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n}")

    cols = args.cols
    rows = (len(thumbs) + cols - 1) // cols
    grid = np.full((rows * THUMB_H, cols * THUMB_W, 3), 40, dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * THUMB_H:(r + 1) * THUMB_H, c * THUMB_W:(c + 1) * THUMB_W] = t

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, grid)
    print(f"\n저장: {out_path} (da=초록 오버레이, GT 없어서 TP/FP/FN 구분은 안 됨)")


if __name__ == "__main__":
    main()
