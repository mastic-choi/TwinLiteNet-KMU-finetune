#!/usr/bin/env python3
"""detect_curve_frames_pth.py에 방향(부호) 정보를 추가한 버전 - 좌회전/우회전 구분용.
signed_score>0, <0 이 각각 어느 방향인지는 실제 이미지로 확인 필요(카메라 좌표계 기준
아직 검증 안 됨) - 이 스크립트는 부호만 저장하고, 실제 방향 라벨은 미리보기로 확인한다.
"""
import csv
import math
import os
import sys

import cv2
import numpy as np
import torch

BASE_DIR = os.path.expanduser("~/fine-tune")
DATASET_DIR = "/mnt/c/fine-tune/dataset"
OUT_CSV = os.path.join(BASE_DIR, "turn_direction_result.csv")

OLD_REPO = os.path.join(BASE_DIR, "TwinLiteNet")
OLD_WEIGHT = os.path.join(OLD_REPO, "pretrained", "best.pth")
OLD_W, OLD_H = 640, 360
ROI_Y0, ROI_Y1 = 250, 390
LL_MIN_AREA = 80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def component_curve_signed(comp_mask):
    ys, xs = np.where(comp_mask)
    if len(ys) < 10:
        return 0.0, 0.0
    y0, y1 = ys.min(), ys.max()
    if y1 - y0 < 15:
        return 0.0, 0.0
    third = (y1 - y0) / 3.0
    top_band = ys <= y0 + third
    bot_band = ys >= y1 - third
    if top_band.sum() < 3 or bot_band.sum() < 3:
        return 0.0, 0.0

    def slope(band_ys, band_xs):
        A = np.vstack([band_ys, np.ones_like(band_ys)]).T
        a, _ = np.linalg.lstsq(A.astype(np.float64), band_xs.astype(np.float64), rcond=None)[0]
        return a

    a_top = slope(ys[top_band], xs[top_band])
    a_bot = slope(ys[bot_band], xs[bot_band])
    ang_top = math.degrees(math.atan(a_top))
    ang_bot = math.degrees(math.atan(a_bot))
    signed = ang_top - ang_bot
    return abs(signed), signed


def main():
    sys.path.insert(0, OLD_REPO)
    from model import TwinLite as net
    model = net.TwinLiteNet()
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(OLD_WEIGHT, map_location=device))
    model.eval()

    files = sorted(f for f in os.listdir(DATASET_DIR) if f.endswith(".png"))
    print(f"{len(files)}장 처리 시작")

    rows = []
    for i, fname in enumerate(files):
        bgr = cv2.imread(os.path.join(DATASET_DIR, fname))
        h, w = bgr.shape[:2]
        resized = cv2.resize(bgr, (OLD_W, OLD_H))
        img = resized[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        t = torch.from_numpy(img).unsqueeze(0).to(device).float() / 255.0
        with torch.no_grad():
            out = model(t)
        _, ll_p = torch.max(out[1], 1)
        ll = ll_p.byte().cpu().numpy()[0]
        ll_full = cv2.resize(ll, (w, h), interpolation=cv2.INTER_NEAREST)

        y0, y1 = max(0, min(ROI_Y0, h)), max(0, min(ROI_Y1, h))
        ll_mask = (ll_full[y0:y1] > 0).astype(np.uint8)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(ll_mask, connectivity=8)
        best_abs, best_signed = 0.0, 0.0
        for c in range(1, n):
            if stats[c, cv2.CC_STAT_AREA] < LL_MIN_AREA:
                continue
            a, s = component_curve_signed(labels == c)
            if a > best_abs:
                best_abs, best_signed = a, s

        rows.append({"file": fname, "curve_score": round(best_abs, 2), "signed_score": round(best_signed, 2)})
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(files)}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "curve_score", "signed_score"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
