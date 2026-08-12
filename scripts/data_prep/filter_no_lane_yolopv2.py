#!/usr/bin/env python3
"""신규 raw 프레임 폴더에 YOLOPv2로 ll(차선) 검출을 돌려서, 차선을 하나도 못 찾은
프레임은 제거한다(완전 삭제 대신 <입력폴더>_no_lane/으로 이동 - 되돌릴 수 있게,
lap_005 정지프레임 정리 때와 같은 컨벤션). skeleton_polyline_utils.mask_to_polylines()
로 폴리라인이 0개면 "차선 없음"으로 판단(기존 build_pseudo_label_dataset_v2.py의
n_ll_empty 집계와 동일한 기준).

사용: python scripts/data_prep/filter_no_lane_yolopv2.py <입력폴더>
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "YOLOPv2"))
from utils.utils import letterbox as yolopv2_letterbox  # noqa: E402
sys.path.insert(0, os.path.join(BASE, "scripts", "pseudo_label"))
from skeleton_polyline_utils import mask_to_polylines  # noqa: E402

YOLOPV2_WEIGHTS = os.path.join(BASE, "YOLOPv2", "data", "weights", "yolopv2.pt")


def infer_ll_yolopv2(model, bgr):
    h0, w0 = bgr.shape[:2]
    img, _, _ = yolopv2_letterbox(bgr, 640, stride=32)
    inp = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    inp_t = torch.from_numpy(inp).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        [pred, anchor_grid], seg, ll = model(inp_t)
    ll_mask = torch.round(ll)[0, 0].numpy().astype(np.uint8)
    return cv2.resize(ll_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    args = ap.parse_args()

    input_dir = args.input_dir.rstrip("/")
    backup_dir = input_dir + "_no_lane"
    os.makedirs(backup_dir, exist_ok=True)

    yolopv2_model = torch.jit.load(YOLOPV2_WEIGHTS, map_location="cpu")
    yolopv2_model.eval()

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"대상 {len(files)}장, YOLOPv2 ll 검출 중...")

    no_lane = []
    for i, fname in enumerate(files):
        img = cv2.imread(os.path.join(input_dir, fname))
        ll_mask = infer_ll_yolopv2(yolopv2_model, img)
        polylines = mask_to_polylines(ll_mask)
        if not polylines:
            no_lane.append(fname)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)} (지금까지 차선없음 {len(no_lane)}장)")

    print(f"\n차선 미검출: {len(no_lane)}장 / {len(files)}장 ({len(no_lane)/len(files)*100:.1f}%)")

    with open(os.path.join(backup_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "reason"])
        for fn in no_lane:
            w.writerow([fn, "yolopv2_no_lane_detected"])

    for fn in no_lane:
        os.replace(os.path.join(input_dir, fn), os.path.join(backup_dir, fn))

    remaining = len([f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"완료: {len(no_lane)}장을 {backup_dir}로 이동, {input_dir}에 {remaining}장 남음")


if __name__ == "__main__":
    main()
