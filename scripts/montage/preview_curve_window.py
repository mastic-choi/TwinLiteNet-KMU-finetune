#!/usr/bin/env python3
"""후보 커브 구간의 대표 프레임 몇 장을 old/new overlay로 렌더링해서 미리보기."""
import os
import sys
from argparse import Namespace

import cv2
import numpy as np
import torch

BASE_DIR = os.path.expanduser("~/fine-tune")
DATASET_DIR = "/mnt/c/fine-tune/dataset"
OUT_DIR = "/mnt/c/fine-tune/TwinLiteNet-KMU-finetune/outputs/montages"

OLD_REPO = os.path.join(BASE_DIR, "TwinLiteNet")
OLD_WEIGHT = os.path.join(OLD_REPO, "pretrained", "best.pth")
OLD_W, OLD_H = 640, 360

NEW_REPO = os.path.join(BASE_DIR, "TwinLiteNetPlus")
NEW_WEIGHT = os.path.join(BASE_DIR, "finetune_out_medium_40ep", "best_ll.pth")
NEW_CONFIG = "medium"
NEW_W, NEW_H = 640, 384

ROI_Y0, ROI_Y1 = 250, 390
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


def main(frame_names):
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
    for fname in frame_names:
        img0 = cv2.imread(os.path.join(DATASET_DIR, fname))
        old_da, old_ll = infer_old(old_model, img0)
        new_da, new_ll = infer_new(new_model, img0)
        t_old = overlay_masks(img0, old_da, old_ll)
        t_new = overlay_masks(img0, new_da, new_ll)
        cv2.putText(t_old, f"OLD {fname}", (5, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(t_new, f"NEW {fname}", (5, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        rows.append(np.hstack([t_old, t_new]))

    grid = np.vstack(rows)
    out_path = os.path.join(OUT_DIR, "curve_preview.png")
    cv2.imwrite(out_path, grid)
    print("저장:", out_path, grid.shape)


if __name__ == "__main__":
    frames = ["frame_000851.png", "frame_000860.png", "frame_000870.png", "frame_000880.png", "frame_000890.png"]
    main(frames)
