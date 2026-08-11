#!/usr/bin/env python3
"""Before/After 파인튜닝 비교 몽타주.

이 컴퓨터(Windows/WSL)에는 원조 TwinLiteNet(구모델 onnx)이나 YOLOPv2 가중치가 없어서
(다른 머신에 있던 참조 자산), 기존 scripts/make_three_way_compare_montage.py와 같은
3-way 비교는 이 머신에서 재현 불가. 대신 이번 세션에 실제로 갖고 있는 것만으로:

  BEFORE = pretrained/medium.pth (원본 TwinLiteNetPlus 공식 pretrained, 파인튜닝 0회)
  AFTER  = finetune_out_medium_40ep/best.pth (이번 40-epoch 파인튜닝 결과, da 기준 best)

를 pseudo_dataset_v2 val 스플릿 샘플 프레임들에 나란히 돌려서 비교.

실행 위치: WSL venv (torch 설치돼 있는 곳).
"""
import os
import random
from argparse import Namespace

import cv2
import numpy as np
import torch

BASE_DIR = os.path.expanduser("~/fine-tune")
REPO_DIR = os.path.join(BASE_DIR, "TwinLiteNetPlus")
PSEUDO_DIR = os.path.join(BASE_DIR, "pseudo_dataset_v2")
IMAGES_DIR = os.path.join(PSEUDO_DIR, "images")

OUT_PATH = os.path.join(BASE_DIR, "before_after_finetune_montage.png")

BEFORE_PTH = os.path.join(REPO_DIR, "pretrained", "medium.pth")
AFTER_PTH = os.path.join(BASE_DIR, "finetune_out_medium_40ep", "best.pth")
AFTER_LL_PTH = os.path.join(BASE_DIR, "finetune_out_medium_40ep", "best_ll.pth")
CONFIG = "medium"
IN_W, IN_H = 640, 384  # letterbox 제거 후(plain resize) 학습 해상도

THUMB_W, THUMB_H = 320, 240
ROI_Y0, ROI_Y1 = 250, 390  # track_drive config.py DL_ROI_Y0/Y1과 동일
N_SAMPLES = 12
VAL_RATIO, SEED = 0.15, 42  # 학습 때 데이터 준비 스크립트와 동일 시드 (val 프레임 재현용)

import sys
sys.path.insert(0, REPO_DIR)
from model.model import TwinLiteNetPlus  # noqa: E402


def roi_cov(mask):
    roi = mask[ROI_Y0:ROI_Y1]
    return float(np.count_nonzero(roi)) / roi.size


def overlay_masks(img0, da_full, ll_full):
    overlay = img0.copy()
    overlay[da_full] = (0.5 * overlay[da_full].astype(np.float64) + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
    overlay[ll_full] = (0, 0, 255)
    cv2.rectangle(overlay, (0, ROI_Y0), (overlay.shape[1], ROI_Y1), (0, 255, 255), 1)
    return overlay


def label_thumb(img, text):
    thumb = cv2.resize(img, (THUMB_W, THUMB_H))
    cv2.rectangle(thumb, (0, 0), (THUMB_W, 18), (0, 0, 0), -1)
    cv2.putText(thumb, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return thumb


def load_model(pth_path):
    model = TwinLiteNetPlus(Namespace(config=CONFIG))
    state = torch.load(pth_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def infer(model, img0):
    h, w = img0.shape[:2]
    resized = cv2.resize(img0, (IN_W, IN_H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])).float()
    with torch.no_grad():
        out_da, out_ll = model(blob)
    da_mask = torch.argmax(out_da, dim=1)[0].numpy().astype(np.uint8)
    ll_mask = torch.argmax(out_ll, dim=1)[0].numpy().astype(np.uint8)
    da_full = cv2.resize(da_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    ll_full = cv2.resize(ll_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return da_full, ll_full


def build_val_set():
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(IMAGES_DIR) if f.endswith(".png"))
    shuffled = names[:]
    random.Random(SEED).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_RATIO))
    return sorted(shuffled[:n_val])


def main():
    assert os.path.isfile(BEFORE_PTH), f"{BEFORE_PTH} 없음"
    assert os.path.isfile(AFTER_PTH), f"{AFTER_PTH} 없음"

    val_names = build_val_set()
    picked = random.Random(7).sample(val_names, min(N_SAMPLES, len(val_names)))
    picked.sort()
    print(f"val {len(val_names)}장 중 {len(picked)}장 샘플링")

    before_model = load_model(BEFORE_PTH)
    after_model = load_model(AFTER_PTH)
    after_ll_model = load_model(AFTER_LL_PTH) if os.path.isfile(AFTER_LL_PTH) else None

    n_cols = 3 if after_ll_model is not None else 2
    rows = []
    for name in picked:
        fname = name + ".png"
        img0 = cv2.imread(os.path.join(IMAGES_DIR, fname))

        b_da, b_ll = infer(before_model, img0)
        a_da, a_ll = infer(after_model, img0)

        t_before = label_thumb(overlay_masks(img0, b_da, b_ll),
                                f"{fname} BEFORE(pretrained) da={roi_cov(b_da):.2f} ll={roi_cov(b_ll):.2f}")
        t_after = label_thumb(overlay_masks(img0, a_da, a_ll),
                               f"AFTER(40ep,best-da) da={roi_cov(a_da):.2f} ll={roi_cov(a_ll):.2f}")
        row = [t_before, t_after]

        line = (f"{fname}: before(da={roi_cov(b_da):.3f},ll={roi_cov(b_ll):.3f}) "
                f"after(da={roi_cov(a_da):.3f},ll={roi_cov(a_ll):.3f})")

        if after_ll_model is not None:
            al_da, al_ll = infer(after_ll_model, img0)
            t_after_ll = label_thumb(overlay_masks(img0, al_da, al_ll),
                                      f"AFTER(40ep,best-ll) da={roi_cov(al_da):.2f} ll={roi_cov(al_ll):.2f}")
            row.append(t_after_ll)
            line += f" after_ll(da={roi_cov(al_da):.3f},ll={roi_cov(al_ll):.3f})"

        rows.append(row)
        print(line)

    header_h = 26
    grid_w = THUMB_W * n_cols
    grid_h = header_h + THUMB_H * len(rows)
    grid = np.full((grid_h, grid_w, 3), 30, dtype=np.uint8)
    titles = ["1. BEFORE (원본 pretrained, 파인튜닝 0회)", "2. AFTER (40epoch, best.pth / da기준)"]
    if n_cols == 3:
        titles.append("3. AFTER (40epoch, best_ll.pth / ll기준)")
    for c, title in enumerate(titles):
        cv2.putText(grid, title, (c * THUMB_W + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for r, thumbs in enumerate(rows):
        for c, t in enumerate(thumbs):
            y0 = header_h + r * THUMB_H
            grid[y0:y0 + THUMB_H, c * THUMB_W:(c + 1) * THUMB_W] = t

    cv2.imwrite(OUT_PATH, grid)
    print("저장:", OUT_PATH, grid.shape)


if __name__ == "__main__":
    main()
