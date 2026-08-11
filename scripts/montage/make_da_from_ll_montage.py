#!/usr/bin/env python3
"""da는 ll 최외곽 두 선 사이라는 기하학적 관계를 이용한 da 생성 방법 비교 몽타주(수정판).
1차 버전은 geometric 영역의 axis-aligned bounding box를 SAM 프롬프트로 썼는데, 사다리꼴
모양(위는 좁고 아래는 화면 전체 폭)이라 bbox가 사실상 화면 전체가 돼서 SAM이 벽/천장까지
과다검출하는 버그가 있었음. 이번엔 bbox 대신 사다리꼴 "안쪽"을 지나는 점 6개(row별
중앙점)로 다중 포인트 프롬프트를 줌 - 그래도 SAM이 점들을 다 포함하는 큰 덩어리로 번지는
문제가 여전해서(과다포함 51%) bbox 버전보다 오히려 더 나쁨. 결론: da는 SAM 없이 기존
old/new 앙상블 모델(과다포함 문제 자체가 없음, IoU 0.852)로 가는 게 맞음 - 이 몽타주는
"왜 SAM 프롬프트 방식을 안 쓰기로 했는지"를 보여주는 근거 자료.

make_yolopv2_weakness_montage.py와 같은 12장(GT 없는 프레임이라 IoU 대신 ROI 커버리지로
표시)으로 geometric(노랑) / geometric points->SAM(빨강) / YOLOPv2 ll(자홍)을 비교."""
import math
import os
import sys

import cv2
import numpy as np
import torch
from ultralytics import SAM

sys.path.insert(0, "YOLOPv2")
from utils.utils import letterbox  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "scripts", "pseudo_label"))
from da_from_ll import geometric_da_from_ll_with_fallback  # noqa: E402
IMAGES_DIR = os.path.join(BASE, "raw", "images_todo")
OUT_PATH = os.path.join(BASE, "outputs", "montages", "da_from_ll_montage.png")
THUMB_W, THUMB_H = 320, 240
N_SAMPLE_POINTS = 6
ROI_Y0, ROI_Y1 = 250, 390

# make_yolopv2_weakness_montage.py와 동일한 12장(da+ll*3 기준 가장 약했던 프레임)
FRAMES = [
    'frame_001082.png', 'frame_001974.png', 'frame_000928.png', 'frame_001973.png',
    'frame_000861.png', 'frame_001225.png', 'frame_001595.png', 'frame_000380.png',
    'frame_001372.png', 'frame_001604.png', 'frame_001966.png', 'frame_001680.png',
]


def yolopv2_ll(model, img_path):
    img0 = cv2.imread(img_path)
    h0, w0 = img0.shape[:2]
    img, _, _ = letterbox(img0, 640, stride=32)
    inp = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    inp_t = torch.from_numpy(inp).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        [pred, anchor_grid], seg, ll = model(inp_t)
    ll_mask = torch.round(ll)[0, 0].numpy().astype(np.uint8)
    return cv2.resize(ll_mask, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)


def sample_points(geo_da, n=N_SAMPLE_POINTS):
    """geo_da(사다리꼴) '안쪽'을 지나는 점들만 반환 - bbox와 달리 화면 전체로 안 번짐."""
    h, _ = geo_da.shape
    ys_valid = [y for y in range(h) if geo_da[y].any()]
    if not ys_valid:
        return []
    idxs = np.linspace(min(ys_valid), max(ys_valid), n).astype(int)
    pts = []
    for y in idxs:
        xs = np.where(geo_da[y])[0]
        if len(xs) == 0:
            continue
        pts.append([float((xs.min() + xs.max()) / 2), float(y)])
    return pts


def roi_cov(mask):
    roi = mask[ROI_Y0:ROI_Y1]
    return float(np.count_nonzero(roi)) / roi.size


def main():
    model = torch.jit.load("YOLOPv2/data/weights/yolopv2.pt", map_location="cpu")
    model.eval()
    sam = SAM("sam2.1_b.pt")

    thumbs = []
    for fname in FRAMES:
        img_path = os.path.join(IMAGES_DIR, fname)
        img = cv2.imread(img_path)

        yll = yolopv2_ll(model, img_path)
        geo_da = geometric_da_from_ll_with_fallback(yll)

        pts = sample_points(geo_da)
        sam_da = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        if pts:
            res = sam(img_path, points=[pts], labels=[[1] * len(pts)], verbose=False)
            if res[0].masks is not None:
                sam_da = res[0].masks.data[0].cpu().numpy().astype(bool)

        overlay = img.copy()
        overlay[sam_da] = (0.5 * overlay[sam_da].astype(np.float64) + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        overlay[geo_da] = (0.5 * overlay[geo_da].astype(np.float64) + 0.5 * np.array([0, 255, 255])).astype(np.uint8)
        overlay[yll] = (255, 0, 255)
        cv2.rectangle(overlay, (0, ROI_Y0), (img.shape[1], ROI_Y1), (0, 255, 255), 1)
        for px, py in pts:
            cv2.circle(overlay, (int(px), int(py)), 4, (0, 0, 0), -1)
            cv2.circle(overlay, (int(px), int(py)), 2, (255, 255, 255), -1)

        thumb = cv2.resize(overlay, (THUMB_W, THUMB_H))
        geo_c, sam_c = roi_cov(geo_da), roi_cov(sam_da)
        label = f"{fname} geo={geo_c:.2f} sam={sam_c:.2f}"
        cv2.rectangle(thumb, (0, 0), (THUMB_W, 18), (0, 0, 0), -1)
        cv2.putText(thumb, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
        print(f"{fname}: geo_cov={geo_c:.3f} sam_cov={sam_c:.3f}")

    cols = 4
    rows_n = math.ceil(len(thumbs) / cols)
    grid = np.zeros((rows_n * THUMB_H, cols * THUMB_W, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * THUMB_H:(r + 1) * THUMB_H, c * THUMB_W:(c + 1) * THUMB_W] = t
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, grid)
    print(f"\n{len(thumbs)}장 -> {OUT_PATH}")
    print("범례: 노랑=geometric(ll 사이 채움), 빨강=geometric point(흑백 점)->SAM, 자홍=YOLOPv2 ll")


if __name__ == "__main__":
    main()
