#!/usr/bin/env python3
"""라벨링 파이프라인 몽타주: 1) 원조 TwinLiteNet 추론  2) 우리 da 라벨(pseudo_dataset_v2/da_masks)
3) YOLOPv2 기반 ll 라벨(pseudo_dataset_v2/ll_masks)  4) 최종 파인튜닝 모델(40epoch) 추론.
README 맨 위 시각화용.
"""
import os
import sys
from argparse import Namespace

import cv2
import numpy as np
import torch

BASE_DIR = os.path.expanduser("~/fine-tune")
PSEUDO_DIR = os.path.join(BASE_DIR, "pseudo_dataset_v2")
OUT_PATH = "/mnt/c/fine-tune/TwinLiteNet-KMU-finetune/outputs/montages/pipeline_montage.png"

OLD_REPO = os.path.join(BASE_DIR, "TwinLiteNet")
OLD_WEIGHT = os.path.join(OLD_REPO, "pretrained", "best.pth")
OLD_W, OLD_H = 640, 360

NEW_REPO = os.path.join(BASE_DIR, "TwinLiteNetPlus")
NEW_WEIGHT = os.path.join(BASE_DIR, "finetune_out_medium_40ep", "best_ll.pth")
NEW_CONFIG = "medium"
NEW_W, NEW_H = 640, 384

ROI_Y0, ROI_Y1 = 250, 390
PANEL_W, PANEL_H = 300, 225
FRAMES = ["frame_000237", "frame_000792", "frame_001708", "frame_001936", "frame_000851"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def overlay_masks(img0, da_full, ll_full):
    overlay = img0.copy()
    overlay[da_full] = (0.5 * overlay[da_full].astype(np.float64) + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
    overlay[ll_full] = (0, 0, 255)
    cv2.rectangle(overlay, (0, ROI_Y0), (overlay.shape[1], ROI_Y1), (0, 255, 255), 1)
    return overlay


def infer_old(model, img0):
    h, w = img0.shape[:2]
    img = cv2.resize(img0, (OLD_W, OLD_H))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    t = torch.from_numpy(img).unsqueeze(0).to(device).float() / 255.0
    with torch.no_grad():
        out = model(t)
    _, da_p = torch.max(out[0], 1)
    _, ll_p = torch.max(out[1], 1)
    da = da_p.byte().cpu().numpy()[0]
    ll = ll_p.byte().cpu().numpy()[0]
    da_full = cv2.resize(da, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    ll_full = cv2.resize(ll, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return da_full, ll_full


def infer_new(model, img0):
    h, w = img0.shape[:2]
    resized = cv2.resize(img0, (NEW_W, NEW_H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])).float().to(device)
    with torch.no_grad():
        out_da, out_ll = model(blob)
    da_mask = torch.argmax(out_da, dim=1)[0].cpu().numpy().astype(np.uint8)
    ll_mask = torch.argmax(out_ll, dim=1)[0].cpu().numpy().astype(np.uint8)
    da_full = cv2.resize(da_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    ll_full = cv2.resize(ll_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return da_full, ll_full


def label_panel(img, text):
    panel = cv2.resize(img, (PANEL_W, PANEL_H))
    cv2.rectangle(panel, (0, 0), (PANEL_W, 20), (0, 0, 0), -1)
    cv2.putText(panel, text, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def main():
    sys.path.insert(0, OLD_REPO)
    from model import TwinLite as net
    old_model = net.TwinLiteNet()
    old_model = torch.nn.DataParallel(old_model)
    old_model = old_model.to(device)
    old_model.load_state_dict(torch.load(OLD_WEIGHT, map_location=device))
    old_model.eval()

    sys.path.insert(0, NEW_REPO)
    from model.model import TwinLiteNetPlus
    new_model = TwinLiteNetPlus(Namespace(config=NEW_CONFIG))
    state = torch.load(NEW_WEIGHT, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_model.load_state_dict(state)
    new_model = new_model.to(device)
    new_model.eval()

    rows = []
    for stem in FRAMES:
        fname = stem + ".png"
        img0 = cv2.imread(os.path.join(PSEUDO_DIR, "images", fname))
        da_label = cv2.imread(os.path.join(PSEUDO_DIR, "da_masks", fname), cv2.IMREAD_GRAYSCALE) > 0
        ll_label = cv2.imread(os.path.join(PSEUDO_DIR, "ll_masks", fname), cv2.IMREAD_GRAYSCALE) > 0

        old_da, old_ll = infer_old(old_model, img0)
        new_da, new_ll = infer_new(new_model, img0)

        p1 = label_panel(overlay_masks(img0, old_da, old_ll), "1. TwinLiteNet (original)")
        p2 = label_panel(overlay_masks(img0, da_label, np.zeros_like(da_label)), "2. Our da label")
        p3 = label_panel(overlay_masks(img0, np.zeros_like(ll_label), ll_label), "3. YOLOPv2-based ll label")
        p4 = label_panel(overlay_masks(img0, new_da, new_ll), "4. Ours (finetuned, 40ep)")

        rows.append(np.hstack([p1, p2, p3, p4]))
        print(stem, "done")

    grid = np.vstack(rows)
    cv2.imwrite(OUT_PATH, grid)
    print("저장:", OUT_PATH, grid.shape)


if __name__ == "__main__":
    main()
