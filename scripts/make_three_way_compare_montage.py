#!/usr/bin/env python3
"""3-way 비교 몽타주: 원조 TwinLiteNet(best.onnx, 실차 배포 중) vs 파인튜닝 TwinLiteNet+
(small config, 470장 pseudo_dataset로 학습한 best(1).pth) vs YOLOPv2(사전학습, 파인튜닝 0회).

프레임 선정: 원래 실패/커브 등 어려운 프레임 8장 : 원래 잘 잡던 프레임 4장 (2:1 비율).
후보 풀은 raw/images_todo/(470장, 새 모델이 실제로 학습에 쓴 데이터)로 한정하고,
train/val 여부는 노트북 cell-12와 동일한 seed=42 셔플로 재현해서 각 썸네일에 표시
(val이면 새 모델이 그 프레임을 직접 학습하진 않은 것 - 일반화 성능 판단에 더 공정).

실행: .venv_yolo (torch/onnxruntime 둘 다 있음)로 실행할 것.
"""
import csv
import math
import os
import random
import sys
from argparse import Namespace

import cv2
import numpy as np
import onnxruntime as ort
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE, "raw", "images_todo")
OUT_PATH = os.path.join(BASE, "outputs", "montages", "three_way_compare_montage.png")

OLD_ONNX = os.path.expanduser("~/code/UMK/track_drive/track_drive/models/best.onnx")
OLD_W, OLD_H = 640, 360
OLD_DA_THRESH, OLD_LL_THRESH = 0.5, 0.7  # config.py DL_FG_THRESHOLD / DL_LL_FG_THRESHOLD 그대로

NEW_PTH = os.path.expanduser("~/Downloads/best(1).pth")
NEW_CONFIG = "small"
NEW_W, NEW_H = 640, 384  # BDD100K.py 패치 후(letterbox 제거, plain resize) 학습 해상도

TWINPLUS_REPO = os.path.join(BASE, "_TwinLiteNetPlus_ref")
sys.path.insert(0, TWINPLUS_REPO)

YOLOPV2_DIR = os.path.join(BASE, "YOLOPv2")
sys.path.insert(0, YOLOPV2_DIR)
from utils.utils import letterbox  # noqa: E402
YOLOPV2_WEIGHTS = os.path.join(YOLOPV2_DIR, "data", "weights", "yolopv2.pt")

THUMB_W, THUMB_H = 320, 240
ROI_Y0, ROI_Y1 = 250, 390
N_HARD, N_EASY = 8, 4
VAL_RATIO, SEED = 0.15, 42  # 노트북 cell-12와 동일 (train/val 태그 재현용)


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


def softmax_fg(logits_2ch):
    m = logits_2ch.max(axis=0, keepdims=True)
    e = np.exp(logits_2ch - m)
    return (e / e.sum(axis=0, keepdims=True))[1]


def select_frames():
    def load_csv(name):
        with open(os.path.join(BASE, name)) as f:
            return list(csv.DictReader(f))

    triage = {r["file"]: r for r in load_csv("triage_result.csv")}
    curve = {r["file"]: float(r["curve_score"]) for r in load_csv("curve_result.csv")}
    missing = {r["file"]: r for r in load_csv("missing_line_result.csv")}
    candidates = sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith(".png"))

    hard, easy = [], []
    for f in candidates:
        is_fail = triage.get(f, {}).get("failure_candidate") == "True"
        n_det = int(missing[f]["n_detected"]) if f in missing else 3
        cscore = curve.get(f, 0.0)
        if is_fail:
            hard.append((f, 3, cscore))  # 진짜 완전 실패 프레임 최우선
        elif n_det < 3 or cscore > 40:
            hard.append((f, 1 if n_det < 3 else 0, cscore))
        elif n_det == 3 and cscore < 15:
            easy.append((f, cscore))

    hard.sort(key=lambda x: (-x[1], -x[2]))
    easy.sort(key=lambda x: x[1])

    picked_hard = [f for f, _, _ in hard[:N_HARD]]
    picked_easy = [f for f, _ in easy[:N_EASY]]
    return picked_hard, picked_easy


def build_val_set():
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(IMAGES_DIR) if f.endswith(".png"))
    shuffled = names[:]
    random.Random(SEED).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_RATIO))
    return set(shuffled[:n_val])


def infer_old(sess, img0):
    h, w = img0.shape[:2]
    resized = cv2.resize(img0, (OLD_W, OLD_H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    da_out, ll_out = sess.run(out_names, {in_name: blob})
    da_prob = cv2.resize(softmax_fg(da_out[0]), (w, h))
    ll_prob = cv2.resize(softmax_fg(ll_out[0]), (w, h))
    return (da_prob >= OLD_DA_THRESH), (ll_prob >= OLD_LL_THRESH)


def infer_new(model, img0):
    h, w = img0.shape[:2]
    resized = cv2.resize(img0, (NEW_W, NEW_H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])).float()
    with torch.no_grad():
        out_da, out_ll = model(blob)
    da_mask = torch.argmax(out_da, dim=1)[0].numpy().astype(np.uint8)
    ll_mask = torch.argmax(out_ll, dim=1)[0].numpy().astype(np.uint8)
    da_full = cv2.resize(da_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    ll_full = cv2.resize(ll_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return da_full, ll_full


def infer_yolopv2(model, img0):
    h, w = img0.shape[:2]
    img, _, _ = letterbox(img0, 640, stride=32)
    inp = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    inp_t = torch.from_numpy(inp).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        [pred, anchor_grid], seg, ll = model(inp_t)
    da = torch.argmax(seg, dim=1)[0].numpy().astype(np.uint8)
    ll_mask = torch.round(ll)[0, 0].numpy().astype(np.uint8)
    da_full = cv2.resize(da, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    ll_full = cv2.resize(ll_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return da_full, ll_full


def main():
    assert os.path.isfile(NEW_PTH), f"{NEW_PTH} 없음"
    assert os.path.isfile(OLD_ONNX), f"{OLD_ONNX} 없음"
    assert os.path.isfile(YOLOPV2_WEIGHTS), f"{YOLOPV2_WEIGHTS} 없음"

    from model.model import TwinLiteNetPlus  # noqa: E402  (TWINPLUS_REPO 클론 뒤 import)

    hard, easy = select_frames()
    val_set = build_val_set()
    print(f"hard {len(hard)}장 / easy {len(easy)}장 선정 (목표 {N_HARD}:{N_EASY})")

    old_sess = ort.InferenceSession(OLD_ONNX, providers=["CPUExecutionProvider"])

    new_model = TwinLiteNetPlus(Namespace(config=NEW_CONFIG))
    state = torch.load(NEW_PTH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_model.load_state_dict(state)
    new_model.eval()

    yolop_model = torch.jit.load(YOLOPV2_WEIGHTS, map_location="cpu")
    yolop_model.eval()

    rows = []
    ordered = [(f, "HARD") for f in hard] + [(f, "EASY") for f in easy]
    for fname, tag in ordered:
        img0 = cv2.imread(os.path.join(IMAGES_DIR, fname))
        split = "val" if os.path.splitext(fname)[0] in val_set else "train"

        old_da, old_ll = infer_old(old_sess, img0)
        new_da, new_ll = infer_new(new_model, img0)
        yp_da, yp_ll = infer_yolopv2(yolop_model, img0)

        t_old = label_thumb(overlay_masks(img0, old_da, old_ll),
                             f"[{tag}/{split}] {fname} OLD da={roi_cov(old_da):.2f} ll={roi_cov(old_ll):.2f}")
        t_new = label_thumb(overlay_masks(img0, new_da, new_ll),
                             f"NEW(small,470) da={roi_cov(new_da):.2f} ll={roi_cov(new_ll):.2f}")
        t_yp = label_thumb(overlay_masks(img0, yp_da, yp_ll),
                            f"YOLOPv2 da={roi_cov(yp_da):.2f} ll={roi_cov(yp_ll):.2f}")

        rows.append([t_old, t_new, t_yp])
        print(f"{tag}/{split} {fname}: old(da={roi_cov(old_da):.3f},ll={roi_cov(old_ll):.3f}) "
              f"new(da={roi_cov(new_da):.3f},ll={roi_cov(new_ll):.3f}) "
              f"yolop(da={roi_cov(yp_da):.3f},ll={roi_cov(yp_ll):.3f})")

    header_h = 26
    grid_w = THUMB_W * 3
    grid_h = header_h + THUMB_H * len(rows)
    grid = np.full((grid_h, grid_w, 3), 30, dtype=np.uint8)
    for c, title in enumerate(["1. 구모델 TwinLiteNet (실차 배포중)", "2. 신모델 TwinLiteNet+ (small,470장)", "3. YOLOPv2 (사전학습, 파인튜닝 0)"]):
        cv2.putText(grid, title, (c * THUMB_W + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for r, thumbs in enumerate(rows):
        for c, t in enumerate(thumbs):
            y0 = header_h + r * THUMB_H
            grid[y0:y0 + THUMB_H, c * THUMB_W:(c + 1) * THUMB_W] = t

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, grid)
    print("저장:", OUT_PATH, grid.shape)


if __name__ == "__main__":
    main()
