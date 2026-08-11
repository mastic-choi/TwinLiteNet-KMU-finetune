#!/usr/bin/env python3
"""나머지(아직 CVAT에서 completed 안 된) 프레임에 대해:
1. old(best.onnx, 640x360) + new(부트스트랩 twinlitenetplus, 640x384) + YOLOPv2(pretrained,
   BDD100K) 세 모델 다 추론
2. da: old/new 중 ROI 커버리지가 더 높은 쪽 선택(YOLOPv2는 da도 있지만, bootstrap GT로
   직접 IoU 검증한 결과 우리 finetuned 모델이 훨씬 정확해서 da엔 안 씀 — PROGRESS.md 참고)
3. ll: old/new/YOLOPv2 세 후보 중 ROI 커버리지가 가장 높은 걸 선택 -> skeleton화 ->
   polyline 단순화 -> 고정 두께로 재래스터화해서 두께 노이즈 제거
   (YOLOPv2가 bootstrap GT 대비 IoU 0.41로 우리 모델 0.05~0.06보다 훨씬 정확했지만,
   곡선 구간에서는 격차가 줄어들어서 무조건 우선시하지 않고 프레임별 자동 선택 유지)
4. **SAM 정밀화는 da/ll 둘 다 안 씀** — bootstrap GT로 직접 검증한 결과 SAM은 완벽한
   입력을 줘도 정확도를 깎아먹는 것으로 확인됨(da IoU 1.0->0.75, ll IoU 0.41->0.28,
   GT 밖으로 평균 20%p 이상 삐져나옴) — 얇은 선이든 큰 덩어리든 "의미적 경계"(차선이
   정의하는 경계)가 "시각적 경계"와 다르면 SAM이 못 맞춤. PROGRESS.md 참고.

출력: ensemble_drafts/images, da_masks, ll_masks + 소스 로그 csv
"""
import csv
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch

from skeleton_polyline_utils import mask_to_polylines, polylines_to_mask

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGES_DIR = os.path.join(BASE, "raw", "images_todo")
OUT_DIR = os.path.join(BASE, "ensemble_drafts")
OUT_IMAGES = os.path.join(OUT_DIR, "images")
OUT_DA = os.path.join(OUT_DIR, "da_masks")
OUT_LL = os.path.join(OUT_DIR, "ll_masks")
LOG_CSV = os.path.join(OUT_DIR, "source_log.csv")

YOLOPV2_DIR = os.path.join(BASE, "YOLOPv2")
sys.path.insert(0, YOLOPV2_DIR)
from utils.utils import letterbox as yolopv2_letterbox  # noqa: E402

OLD_ONNX = os.path.expanduser("~/code/UMK/track_drive/track_drive/models/best.onnx")
NEW_ONNX = os.path.expanduser("~/Downloads/twinlitenetplus_small_finetuned_refined.onnx")
YOLOPV2_WEIGHTS = os.path.join(YOLOPV2_DIR, "data", "weights", "yolopv2.pt")

OLD_W, OLD_H = 640, 360
NEW_W, NEW_H = 640, 384
DA_THRESH, LL_THRESH = 0.5, 0.7
ROI_Y0, ROI_Y1 = 250, 390
LL_REFINE_WIDTH = 8            # skeleton -> polyline 재래스터화 고정 두께(px), bootstrap_refined와 동일값


def softmax_fg(logits_2ch):
    m = logits_2ch.max(axis=0, keepdims=True)
    e = np.exp(logits_2ch - m)
    return (e / e.sum(axis=0, keepdims=True))[1]


def infer(sess, bgr, model_w, model_h):
    h, w = bgr.shape[:2]
    resized = cv2.resize(bgr, (model_w, model_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])
    da_out, ll_out = sess.run([o.name for o in sess.get_outputs()], {sess.get_inputs()[0].name: blob})
    da_prob = cv2.resize(softmax_fg(da_out[0]), (w, h))
    ll_prob = cv2.resize(softmax_fg(ll_out[0]), (w, h))
    return da_prob, ll_prob


def roi_cov(mask):
    roi = mask[ROI_Y0:ROI_Y1]
    return float(np.count_nonzero(roi)) / roi.size


def yolopv2_infer_ll(model, bgr):
    """YOLOPv2(BDD100K pretrained)로 ll만 추론. 우리 이미지(640x480, 4:3)는 letterbox
    패딩이 거의 0이라(480이 stride=32 배수) 레포의 하드코딩 crop[12:372] 없이 letterbox
    해상도 그대로 round해서 원본 크기로 resize(make_yolopv2_montage.py와 동일 방식)."""
    h0, w0 = bgr.shape[:2]
    img, _, _ = yolopv2_letterbox(bgr, 640, stride=32)
    inp = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    inp_t = torch.from_numpy(inp).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        [pred, anchor_grid], seg, ll = model(inp_t)
    ll_mask = torch.round(ll)[0, 0].numpy().astype(np.uint8)
    return cv2.resize(ll_mask, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)


def refine_ll(ll_mask):
    """앙상블로 고른 ll 마스크 -> skeleton화 -> polyline 단순화 -> 고정 두께 재래스터화.
    polyline을 하나도 못 뽑으면(빈 마스크 등) 원본을 그대로 반환."""
    h, w = ll_mask.shape
    polylines = mask_to_polylines(ll_mask)
    if not polylines:
        return ll_mask, 0
    refined = polylines_to_mask(polylines, h, w, line_width=LL_REFINE_WIDTH).astype(bool)
    return refined, len(polylines)


def build_one(sess_old, sess_new, yolopv2_model, fname):
    img_path = os.path.join(IMAGES_DIR, fname)
    bgr = cv2.imread(img_path)

    da_old, ll_old = infer(sess_old, bgr, OLD_W, OLD_H)
    da_new, ll_new = infer(sess_new, bgr, NEW_W, NEW_H)
    ll_mask_yolopv2 = yolopv2_infer_ll(yolopv2_model, bgr)

    da_mask_old, ll_mask_old = da_old >= DA_THRESH, ll_old >= LL_THRESH
    da_mask_new, ll_mask_new = da_new >= DA_THRESH, ll_new >= LL_THRESH

    da_cov_old, da_cov_new = roi_cov(da_mask_old), roi_cov(da_mask_new)
    ll_cov_old, ll_cov_new = roi_cov(ll_mask_old), roi_cov(ll_mask_new)
    ll_cov_yolopv2 = roi_cov(ll_mask_yolopv2)

    # da: old/new만 후보(YOLOPv2 da는 bootstrap GT 검증 결과 우리 모델보다 부정확해서 제외)
    da_mask = da_mask_new if da_cov_new >= da_cov_old else da_mask_old
    da_src = "new" if da_cov_new >= da_cov_old else "old"

    # ll: old/new/YOLOPv2 3-way, ROI 커버리지 최고인 걸 선택(곡선에서 YOLOPv2가 약해질 수
    # 있어서 무조건 우선하지 않고 프레임별 자동 선택 - PROGRESS.md 참고)
    ll_candidates = {"old": (ll_mask_old, ll_cov_old), "new": (ll_mask_new, ll_cov_new),
                      "yolopv2": (ll_mask_yolopv2, ll_cov_yolopv2)}
    ll_src = max(ll_candidates, key=lambda k: ll_candidates[k][1])
    ll_mask = ll_candidates[ll_src][0]

    ll_mask_refined, n_polylines = refine_ll(ll_mask)
    da_mask_refined = da_mask | ll_mask_refined  # 차선은 도로 위에 있으니 da에 포함

    cv2.imwrite(os.path.join(OUT_IMAGES, fname), bgr)
    cv2.imwrite(os.path.join(OUT_DA, fname), (da_mask_refined.astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(OUT_LL, fname), (ll_mask_refined.astype(np.uint8) * 255))

    return {
        "file": fname, "da_src": da_src, "ll_src": ll_src,
        "da_cov_old": round(da_cov_old, 4), "da_cov_new": round(da_cov_new, 4),
        "ll_cov_old": round(ll_cov_old, 4), "ll_cov_new": round(ll_cov_new, 4),
        "ll_cov_yolopv2": round(ll_cov_yolopv2, 4),
        "ll_n_polylines": n_polylines,
    }


def main(frame_list):
    for d in (OUT_IMAGES, OUT_DA, OUT_LL):
        os.makedirs(d, exist_ok=True)

    sess_old = ort.InferenceSession(OLD_ONNX, providers=["CPUExecutionProvider"])
    sess_new = ort.InferenceSession(NEW_ONNX, providers=["CPUExecutionProvider"])
    yolopv2_model = torch.jit.load(YOLOPV2_WEIGHTS, map_location="cpu")
    yolopv2_model.eval()

    rows = []
    for i, f in enumerate(frame_list):
        rows.append(build_one(sess_old, sess_new, yolopv2_model, f))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(frame_list)}")

    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_ll_empty = sum(1 for r in rows if r["ll_n_polylines"] == 0)
    print(f"\n완료 {len(rows)}장 -> {OUT_DIR}")
    print("da 소스:", {s: sum(1 for r in rows if r["da_src"] == s) for s in ("old", "new")})
    print("ll 소스:", {s: sum(1 for r in rows if r["ll_src"] == s) for s in ("old", "new", "yolopv2")})
    print(f"ll polyline 0개(전부 다 차선 못 찾음): {n_ll_empty}/{len(rows)} — 이 프레임들은 사람이 처음부터 확인 필요")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        frames = sys.argv[1:]
    else:
        frames = sorted(os.listdir(IMAGES_DIR))
    main(frames)
