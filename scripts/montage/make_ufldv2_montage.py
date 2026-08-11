#!/usr/bin/env python3
"""Ultra-Fast-Lane-Detection-v2(CULane ResNet18 사전학습,
https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) 20장 몽타주.
레포의 get_model()/merge_config()가 .cuda()/argparse-yaml을 하드코딩하고 있어서
CPU에서 그대로 못 씀 -> configs/culane_res18.py 값으로 cfg를 직접 구성하고
parsingNet을 .cuda() 없이 직접 생성해서 우회함. 이 모델은 da(주행가능영역) 헤드가
없고 차선 포인트 좌표만 냄(mask가 아니라 polyline 좌표 직접 회귀).
"""
import math
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.join(BASE, "Ultra-Fast-Lane-Detection-v2")
sys.path.insert(0, REPO_DIR)

# utils/common.py가 학습 전용 data/dali_data.py(nvidia.dali, CUDA 전용)를 무조건 import함 ->
# 추론(inference)엔 필요 없는데도 막힘 -> 더미 모듈로 미리 채워서 우회
import types  # noqa: E402
_dali_stub = types.ModuleType("data.dali_data")
_dali_stub.TrainCollect = object
sys.modules["data.dali_data"] = _dali_stub

from model.model_culane import parsingNet  # noqa: E402

IMAGES_DIR = os.path.join(BASE, "raw", "images_todo")
OUT_PATH = os.path.join(BASE, "outputs", "montages", "ufldv2_montage.png")
WEIGHTS = os.path.join(REPO_DIR, "weights", "culane_res18.pth")
THUMB_W, THUMB_H = 320, 240
ROI_Y0, ROI_Y1 = 250, 390

FRAMES = [
    'frame_000024.png', 'frame_000173.png', 'frame_000268.png', 'frame_000380.png',
    'frame_000457.png', 'frame_000543.png', 'frame_000674.png', 'frame_000750.png',
    'frame_000861.png', 'frame_000968.png', 'frame_001058.png', 'frame_001156.png',
    'frame_001225.png', 'frame_001389.png', 'frame_001476.png', 'frame_001595.png',
    'frame_001680.png', 'frame_001758.png', 'frame_001864.png', 'frame_001966.png',
]

# configs/culane_res18.py 값
CFG = dict(
    backbone='18', num_cell_row=200, num_row=72, num_cell_col=100, num_col=81,
    num_lanes=4, use_aux=False, train_height=320, train_width=1600, fc_norm=True,
    crop_ratio=0.6,
)
ROW_ANCHOR = np.linspace(0.42, 1, CFG["num_row"])
COL_ANCHOR = np.linspace(0, 1, CFG["num_col"])


def pred2coords(pred, row_anchor, col_anchor, local_width=1, ow=640, oh=480):
    _, num_grid_row, num_cls_row, _ = pred['loc_row'].shape
    _, num_grid_col, num_cls_col, _ = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row = pred['exist_row'].argmax(1).cpu()
    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col = pred['exist_col'].argmax(1).cpu()
    loc_row, loc_col = pred['loc_row'].cpu(), pred['loc_col'].cpu()

    coords = []
    for i in [1, 2]:  # row-based lane (좌/우 대략 수평에 가까운 차선)
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(list(range(
                        max(0, max_indices_row[0, k, i] - local_width),
                        min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1)))
                    out_tmp = (loc_row[0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * ow
                    tmp.append((int(out_tmp), int(row_anchor[k] * oh)))
            coords.append(tmp)
    for i in [0, 3]:  # col-based lane (대각/수직에 가까운 차선)
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(list(range(
                        max(0, max_indices_col[0, k, i] - local_width),
                        min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1)))
                    out_tmp = (loc_col[0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col - 1) * oh
                    tmp.append((int(col_anchor[k] * ow), int(out_tmp)))
            coords.append(tmp)
    return coords


def roi_cov_from_coords(coords, ow, oh, width=8):
    mask = np.zeros((oh, ow), dtype=np.uint8)
    for lane in coords:
        pts = np.array(lane, dtype=np.int32).reshape(-1, 1, 2)
        if len(pts) >= 2:
            cv2.polylines(mask, [pts], False, 1, thickness=width)
        elif len(pts) == 1:
            cv2.circle(mask, tuple(pts[0, 0]), width // 2, 1, -1)
    roi = mask[ROI_Y0:ROI_Y1] > 0
    return mask, float(np.count_nonzero(roi)) / roi.size


def main():
    net = parsingNet(pretrained=False, backbone=CFG["backbone"], num_grid_row=CFG["num_cell_row"],
                      num_cls_row=CFG["num_row"], num_grid_col=CFG["num_cell_col"], num_cls_col=CFG["num_col"],
                      num_lane_on_row=CFG["num_lanes"], num_lane_on_col=CFG["num_lanes"], use_aux=CFG["use_aux"],
                      input_height=CFG["train_height"], input_width=CFG["train_width"], fc_norm=CFG["fc_norm"])

    state_dict = torch.load(WEIGHTS, map_location='cpu')['model']
    compatible = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
    net.load_state_dict(compatible, strict=False)
    net.eval()

    resize_h = int(CFG["train_height"] / CFG["crop_ratio"])
    img_transforms = transforms.Compose([
        transforms.Resize((resize_h, CFG["train_width"])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    thumbs = []
    for fname in FRAMES:
        pil_img = Image.open(os.path.join(IMAGES_DIR, fname)).convert("RGB")
        ow, oh = pil_img.size
        inp = img_transforms(pil_img)
        inp = inp[:, -CFG["train_height"]:, :]  # LaneTestDataset과 동일하게 아래쪽만 crop
        inp = inp.unsqueeze(0)

        with torch.no_grad():
            pred = net(inp)

        coords = pred2coords(pred, ROW_ANCHOR, COL_ANCHOR, ow=ow, oh=oh)
        mask, ll_cov = roi_cov_from_coords(coords, ow, oh)

        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        overlay = bgr.copy()
        overlay[mask > 0] = (0, 0, 255)
        cv2.rectangle(overlay, (0, ROI_Y0), (ow, ROI_Y1), (0, 255, 255), 1)

        thumb = cv2.resize(overlay, (THUMB_W, THUMB_H))
        label = f"{fname} ll={ll_cov:.3f} (da 헤드 없음)"
        cv2.rectangle(thumb, (0, 0), (THUMB_W, 18), (0, 0, 0), -1)
        cv2.putText(thumb, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
        print(f"{fname}: ll={ll_cov:.3f} (lanes found: {len(coords)})")

    cols = 5
    rows_n = math.ceil(len(thumbs) / cols)
    grid = np.zeros((rows_n * THUMB_H, cols * THUMB_W, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * THUMB_H:(r + 1) * THUMB_H, c * THUMB_W:(c + 1) * THUMB_W] = t
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, grid)
    print(f"\n{len(thumbs)}장 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
