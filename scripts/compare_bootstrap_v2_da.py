#!/usr/bin/env python3
"""Stage 1(bootstrap_v2, 134장 사람 라벨 기반) 모델이 §2.12에서 발견한 da 커브
과다포함 편향을 실제로 고쳤는지 검증.
1) 74장 사람 GT 기준 정량 비교: 구모델(small,470, pth) vs 신모델(bootstrap_v2, onnx)의
   IoU / 과다포함(FP)·과소포함(FN) 비율.
2) 커브 프레임(§2.12에서 과다포함 확인된 프레임들) 구모델 vs 신모델 da 시각 비교.
"""
import math
import os
import sys
from argparse import Namespace

import cv2
import numpy as np
import onnxruntime as ort
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "_TwinLiteNetPlus_ref"))
from model.model import TwinLiteNetPlus  # noqa: E402

VANILLA_ONNX = os.path.expanduser("~/code/UMK/track_drive/track_drive/models/best.onnx")  # 실차 배포중인 순정 TwinLiteNet
VANILLA_W, VANILLA_H = 640, 360
VANILLA_DA_THRESH, VANILLA_LL_THRESH = 0.5, 0.7  # config.py DL_FG_THRESHOLD/DL_LL_FG_THRESHOLD

OLD_PTH = os.path.expanduser("~/Downloads/best(1).pth")  # small,470 - §2.12에서 확인된 커브 da 오염 있음
OLD_CONFIG = "small"

NEW_ONNX = os.path.expanduser("~/Downloads/twinlitenetplus_small_bootstrap_v2.onnx")

W, H = 640, 384
GT_IMAGES = os.path.join(BASE, "bootstrap", "images")
GT_DA = os.path.join(BASE, "bootstrap", "da_masks")
IMAGES_470 = os.path.join(BASE, "raw", "images_todo")

OUT_QUANT_VIS = os.path.join(BASE, "outputs", "montages", "bootstrap_v2_overpaint_check.png")
OUT_CURVE_VIS = os.path.join(BASE, "outputs", "montages", "bootstrap_v2_vs_old_curve_montage.png")
THUMB_W, THUMB_H = 320, 240

CURVE_FRAMES = [
    "frame_001884.png", "frame_001082.png", "frame_001372.png", "frame_000851.png",
    "frame_001693.png", "frame_001431.png", "frame_001430.png", "frame_001633.png",
]


def softmax_fg(logits_2ch):
    m = logits_2ch.max(axis=0, keepdims=True)
    e = np.exp(logits_2ch - m)
    return (e / e.sum(axis=0, keepdims=True))[1]


def infer_vanilla(sess, img0):
    h, w = img0.shape[:2]
    resized = cv2.resize(img0, (VANILLA_W, VANILLA_H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    da_out, ll_out = sess.run(out_names, {in_name: blob})
    da_prob = cv2.resize(softmax_fg(da_out[0]), (w, h))
    ll_prob = cv2.resize(softmax_fg(ll_out[0]), (w, h))
    return (da_prob >= VANILLA_DA_THRESH), (ll_prob >= VANILLA_LL_THRESH)


def load_old():
    model = TwinLiteNetPlus(Namespace(config=OLD_CONFIG))
    state = torch.load(OLD_PTH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def infer_old(model, img0):
    h, w = img0.shape[:2]
    resized = cv2.resize(img0, (W, H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])).float()
    with torch.no_grad():
        out_da, out_ll = model(blob)
    da_mask = torch.argmax(out_da, dim=1)[0].numpy().astype(np.uint8)
    ll_mask = torch.argmax(out_ll, dim=1)[0].numpy().astype(np.uint8)
    da_full = cv2.resize(da_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    ll_full = cv2.resize(ll_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return da_full, ll_full


def infer_new(sess, img0):
    h, w = img0.shape[:2]
    resized = cv2.resize(img0, (W, H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])
    da_out, ll_out = sess.run(["da", "ll"], {"images": blob})
    da_mask = np.argmax(da_out[0], axis=0).astype(np.uint8)
    ll_mask = np.argmax(ll_out[0], axis=0).astype(np.uint8)
    da_full = cv2.resize(da_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    ll_full = cv2.resize(ll_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return da_full, ll_full


def roi_free_stats(pred, gt):
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    gt_area = max(gt.sum(), 1)
    iou = tp.sum() / max((pred | gt).sum(), 1)
    return iou, fp.sum() / gt_area, fn.sum() / gt_area


def label_thumb(img, text):
    thumb = cv2.resize(img, (THUMB_W, THUMB_H))
    cv2.rectangle(thumb, (0, 0), (THUMB_W, 18), (0, 0, 0), -1)
    cv2.putText(thumb, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return thumb


ROI_Y0, ROI_Y1 = 250, 390  # track_drive config.py DL_ROI_Y0/Y1 - BEV 원근변환 전에 자르는 영역


def overlay_masks(img0, da, ll):
    overlay = img0.copy()
    overlay[da] = (0.5 * overlay[da].astype(np.float64) + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
    overlay[ll] = (0, 0, 255)
    cv2.rectangle(overlay, (0, ROI_Y0), (overlay.shape[1], ROI_Y1), (0, 255, 255), 1)  # 노란 선 = BEV 크롭 경계
    return overlay


def main():
    vanilla_sess = ort.InferenceSession(VANILLA_ONNX, providers=["CPUExecutionProvider"])
    old_model = load_old()
    new_sess = ort.InferenceSession(NEW_ONNX, providers=["CPUExecutionProvider"])

    # 1) 74장 GT 정량 비교
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(GT_IMAGES))
    old_stats, new_stats = [], []
    for n in names:
        img0 = cv2.imread(os.path.join(GT_IMAGES, n + ".png"))
        gt = cv2.imread(os.path.join(GT_DA, n + ".png"), cv2.IMREAD_GRAYSCALE) > 127
        old_da, _ = infer_old(old_model, img0)
        new_da, _ = infer_new(new_sess, img0)
        old_stats.append(roi_free_stats(old_da, gt))
        new_stats.append(roi_free_stats(new_da, gt))

    old_iou, old_fp, old_fn = np.mean(old_stats, axis=0)
    new_iou, new_fp, new_fn = np.mean(new_stats, axis=0)
    print("=== 74장 human GT 기준 정량 비교 ===")
    print(f"구모델(small,470, 커브 편향 있음)   : IoU={old_iou:.3f} FP/GT(과다포함)={old_fp:.3f} FN/GT(과소포함)={old_fn:.3f}")
    print(f"신모델(bootstrap_v2, 134장 사람라벨): IoU={new_iou:.3f} FP/GT(과다포함)={new_fp:.3f} FN/GT(과소포함)={new_fn:.3f}")

    # 2) 커브 프레임 시각 비교 (원래 470장 세트에서, GT 없음 - 육안 비교용)
    blocks = []
    for fname in CURVE_FRAMES:
        img0 = cv2.imread(os.path.join(IMAGES_470, fname))
        van_da, van_ll = infer_vanilla(vanilla_sess, img0)
        old_da, old_ll = infer_old(old_model, img0)
        new_da, new_ll = infer_new(new_sess, img0)

        t_van = label_thumb(overlay_masks(img0, van_da, van_ll),
                             f"{fname} 1.순정 da={van_da.sum()/van_da.size:.3f} ll={van_ll.sum()/van_ll.size:.3f}")
        t_old = label_thumb(overlay_masks(img0, old_da, old_ll),
                             f"2.small470(오염) da={old_da.sum()/old_da.size:.3f} ll={old_ll.sum()/old_ll.size:.3f}")
        t_new = label_thumb(overlay_masks(img0, new_da, new_ll),
                             f"3.bootstrap_v2 da={new_da.sum()/new_da.size:.3f} ll={new_ll.sum()/new_ll.size:.3f}")
        blocks.append(np.hstack([t_van, t_old, t_new]))  # 프레임 1개 = 순정+small470+bootstrap_v2 가로 3쌍 = 1블록
        print(f"{fname}: vanilla(da={van_da.sum()/van_da.size:.3f}) old(da={old_da.sum()/old_da.size:.3f}) new(da={new_da.sum()/new_da.size:.3f})")

    # A4 비율에 맞게 블록을 2열 그리드로 배치(세로로 8줄씩 안 늘어지게) - 8블록 -> 2열×4행
    BLOCKS_PER_ROW = 2
    block_h, block_w = blocks[0].shape[:2]
    n_rows = math.ceil(len(blocks) / BLOCKS_PER_ROW)
    grid = np.full((n_rows * block_h, BLOCKS_PER_ROW * block_w, 3), 30, dtype=np.uint8)
    for i, b in enumerate(blocks):
        r, c = divmod(i, BLOCKS_PER_ROW)
        grid[r * block_h:(r + 1) * block_h, c * block_w:(c + 1) * block_w] = b
    print(f"그리드 크기: {grid.shape[1]}x{grid.shape[0]} (가로:세로 = {grid.shape[1] / grid.shape[0]:.2f}:1)")

    os.makedirs(os.path.dirname(OUT_CURVE_VIS), exist_ok=True)
    cv2.imwrite(OUT_CURVE_VIS, grid)
    print("\n저장:", OUT_CURVE_VIS)


if __name__ == "__main__":
    main()
