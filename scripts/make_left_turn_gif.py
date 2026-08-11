#!/usr/bin/env python3
"""dataset/의 연속 프레임(커브 구간, frame_000800~000935)을 old(원조 TwinLiteNet) vs
new(우리 파인튜닝 TwinLiteNetPlus, 40epoch) 나란히 오버레이해서 실제 주행처럼 보이는
GIF로 만든다. 곡률 검출(scripts/detect_curve_frames_pth.py) 결과 기준으로 선정한 구간.
"""
import os
import sys
from argparse import Namespace

import cv2
import numpy as np
import torch
from PIL import Image

BASE_DIR = os.path.expanduser("~/fine-tune")
DATASET_DIR = "/mnt/c/fine-tune/dataset"
OUT_PATH = "/mnt/c/fine-tune/TwinLiteNet-KMU-finetune/outputs/montages/left_turn_old_vs_new.gif"

OLD_REPO = os.path.join(BASE_DIR, "TwinLiteNet")
OLD_WEIGHT = os.path.join(OLD_REPO, "pretrained", "best.pth")
OLD_W, OLD_H = 640, 360

NEW_REPO = os.path.join(BASE_DIR, "TwinLiteNetPlus")
NEW_WEIGHT = os.path.join(BASE_DIR, "finetune_out_medium_40ep", "best_ll.pth")
NEW_CONFIG = "medium"
NEW_W, NEW_H = 640, 384

ROI_Y0, ROI_Y1 = 250, 390
START, END = 400, 480  # 좌회전 구간
PANEL_W, PANEL_H = 400, 300
FPS = 12

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
    cv2.rectangle(panel, (0, 0), (PANEL_W, 22), (0, 0, 0), -1)
    cv2.putText(panel, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
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

    frames = []
    for idx in range(START, END + 1):
        fname = f"frame_{idx:06d}.png"
        path = os.path.join(DATASET_DIR, fname)
        if not os.path.isfile(path):
            continue
        img0 = cv2.imread(path)
        old_da, old_ll = infer_old(old_model, img0)
        new_da, new_ll = infer_new(new_model, img0)

        p_old = label_panel(overlay_masks(img0, old_da, old_ll), f"OLD TwinLiteNet  {fname}")
        p_new = label_panel(overlay_masks(img0, new_da, new_ll), "NEW finetuned (40ep)")
        combo = np.hstack([p_old, p_new])
        combo_rgb = cv2.cvtColor(combo, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(combo_rgb))

        if (idx - START) % 20 == 0:
            print(f"{idx - START}/{END - START} frames rendered")

    print(f"총 {len(frames)}프레임, GIF 저장 중...")
    duration_ms = int(1000 / FPS)
    frames[0].save(
        OUT_PATH, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=True
    )
    size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f"저장: {OUT_PATH} ({size_mb:.1f} MB, {len(frames)}프레임 @ {FPS}fps)")


if __name__ == "__main__":
    main()
