#!/usr/bin/env python3
"""dataset/(2123장, 연속 프레임) 전체에 원조 TwinLiteNet(구모델)과 우리 파인튜닝
TwinLiteNetPlus(신모델, 40epoch)를 돌려서 프레임별 ROI 커버리지를 CSV로 남긴다.
이후 몽타주/GIF용 프레임 선정에 사용.
"""
import csv
import os
import sys
from argparse import Namespace

import cv2
import numpy as np
import torch

BASE_DIR = os.path.expanduser("~/fine-tune")
DATASET_DIR = "/mnt/c/fine-tune/dataset"
OUT_CSV = os.path.join(BASE_DIR, "old_vs_new_gap.csv")

OLD_REPO = os.path.join(BASE_DIR, "TwinLiteNet")
OLD_WEIGHT = os.path.join(OLD_REPO, "pretrained", "best.pth")
OLD_W, OLD_H = 640, 360

NEW_REPO = os.path.join(BASE_DIR, "TwinLiteNetPlus")
NEW_WEIGHT = os.path.join(BASE_DIR, "finetune_out_medium_40ep", "best_ll.pth")
NEW_CONFIG = "medium"
NEW_W, NEW_H = 640, 384

ROI_Y0, ROI_Y1 = 250, 390

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def roi_cov(mask):
    roi = mask[ROI_Y0:ROI_Y1]
    return float(np.count_nonzero(roi)) / roi.size


def load_old_model():
    sys.path.insert(0, OLD_REPO)
    from model import TwinLite as net
    model = net.TwinLiteNet()
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(OLD_WEIGHT, map_location=device))
    model.eval()
    return model


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


def load_new_model():
    sys.path.insert(0, NEW_REPO)
    from model.model import TwinLiteNetPlus
    model = TwinLiteNetPlus(Namespace(config=NEW_CONFIG))
    state = torch.load(NEW_WEIGHT, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


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


def main():
    old_model = load_old_model()
    new_model = load_new_model()

    names = sorted(f for f in os.listdir(DATASET_DIR) if f.endswith(".png"))
    print(f"{len(names)}장 스캔 시작")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "old_da", "old_ll", "new_da", "new_ll"])
        for i, fname in enumerate(names):
            img0 = cv2.imread(os.path.join(DATASET_DIR, fname))
            old_da, old_ll = infer_old(old_model, img0)
            new_da, new_ll = infer_new(new_model, img0)
            writer.writerow([fname, f"{roi_cov(old_da):.4f}", f"{roi_cov(old_ll):.4f}",
                              f"{roi_cov(new_da):.4f}", f"{roi_cov(new_ll):.4f}"])
            if i % 200 == 0:
                print(f"{i}/{len(names)}")

    print("저장:", OUT_CSV)


if __name__ == "__main__":
    main()
