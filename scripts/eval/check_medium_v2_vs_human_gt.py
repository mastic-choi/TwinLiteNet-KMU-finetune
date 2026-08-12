#!/usr/bin/env python3
"""현재 배포 후보 모델(twinlitenetplus_medium_v2.onnx, outputs/models/best.onnx와
동일 가중치 - medium config, pseudo_dataset_v2 1081장 학습, ll IOU 기준 best)을
사람이 직접 라벨링한 bootstrap_v2(156장, da+ll 전부 사람 검증)에 돌려서 da/ll 둘 다
IoU/과다포함(FP)/과소포함(FN)을 직접 측정한다. check_da_overpaint.py(§2.12, da만,
74장 대상)의 후속판 - 지금은 ll도 사람 라벨이 있고 corpus도 156장으로 늘어서 범위를
넓힘.
"""
import os

import cv2
import numpy as np
import onnxruntime as ort

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ONNX_PATH = os.path.join(BASE, "outputs", "models", "best.onnx")
IN_W, IN_H = 640, 384

GT_DIR = os.path.join(BASE, "bootstrap_v2")
GT_IMAGES = os.path.join(GT_DIR, "images")
GT_DA = os.path.join(GT_DIR, "da_masks")
GT_LL = os.path.join(GT_DIR, "ll_masks")

OUT_PATH = os.path.join(BASE, "outputs", "montages", "medium_v2_vs_human_gt_156.png")
THUMB_W, THUMB_H = 320, 240
N_SHOW = 16


def softmax_fg(logits_2ch):
    m = logits_2ch.max(axis=0, keepdims=True)
    e = np.exp(logits_2ch - m)
    return (e / e.sum(axis=0, keepdims=True))[1]


def infer(sess, img0):
    h, w = img0.shape[:2]
    resized = cv2.resize(img0, (IN_W, IN_H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])
    da_out, ll_out = sess.run(["da", "ll"], {"images": blob})
    da_prob = softmax_fg(da_out[0])
    ll_prob = softmax_fg(ll_out[0])
    da_mask = cv2.resize(da_prob, (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5
    ll_mask = cv2.resize(ll_prob, (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5
    return da_mask, ll_mask


def fp_fn_iou(pred, gt):
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    gt_area = max(gt.sum(), 1)
    return (fp.sum() / gt_area, fn.sum() / gt_area, tp.sum() / max((pred | gt).sum(), 1))


def overlay(img0, pred, gt):
    vis = img0.copy()
    tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt
    vis[tp] = (0.5 * vis[tp].astype(np.float64) + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
    vis[fp] = (0.5 * vis[fp].astype(np.float64) + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    vis[fn] = (0.5 * vis[fn].astype(np.float64) + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    return vis


def main():
    assert os.path.isfile(ONNX_PATH), f"{ONNX_PATH} 없음"
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    names = sorted(os.path.splitext(f)[0] for f in os.listdir(GT_IMAGES))
    print(f"대상: {len(names)}장 (bootstrap_v2, 사람 da+ll 검증 전체)")

    da_fp, da_fn, da_iou = [], [], []
    ll_fp, ll_fn, ll_iou = [], [], []
    thumbs = []

    rng = np.random.RandomState(42)
    show_idx = set(rng.choice(len(names), size=min(N_SHOW, len(names)), replace=False).tolist())

    for i, n in enumerate(names):
        img0 = cv2.imread(os.path.join(GT_IMAGES, n + ".png"))
        gt_da = cv2.imread(os.path.join(GT_DA, n + ".png"), cv2.IMREAD_GRAYSCALE) > 127
        gt_ll = cv2.imread(os.path.join(GT_LL, n + ".png"), cv2.IMREAD_GRAYSCALE) > 127

        pred_da, pred_ll = infer(sess, img0)

        fp, fn, iou = fp_fn_iou(pred_da, gt_da)
        da_fp.append(fp); da_fn.append(fn); da_iou.append(iou)

        fp, fn, iou = fp_fn_iou(pred_ll, gt_ll)
        ll_fp.append(fp); ll_fn.append(fn); ll_iou.append(iou)

        if i in show_idx:
            vis_da = overlay(img0, pred_da, gt_da)
            vis_ll = overlay(img0, pred_ll, gt_ll)
            top = cv2.resize(vis_da, (THUMB_W, THUMB_H))
            bot = cv2.resize(vis_ll, (THUMB_W, THUMB_H))
            cv2.rectangle(top, (0, 0), (THUMB_W, 16), (0, 0, 0), -1)
            cv2.putText(top, f"{n} da IoU={da_iou[-1]:.2f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.rectangle(bot, (0, 0), (THUMB_W, 16), (0, 0, 0), -1)
            cv2.putText(bot, f"{n} ll IoU={ll_iou[-1]:.2f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            thumbs.append(np.vstack([top, bot]))

    print(f"\n=== da (n={len(names)}) ===")
    print(f"mean IoU={np.mean(da_iou):.3f}  FP/GT={np.mean(da_fp)*100:.1f}%  FN/GT={np.mean(da_fn)*100:.1f}%")
    print(f"=== ll (n={len(names)}) ===")
    print(f"mean IoU={np.mean(ll_iou):.3f}  FP/GT={np.mean(ll_fp)*100:.1f}%  FN/GT={np.mean(ll_fn)*100:.1f}%")

    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    grid = np.zeros((rows * THUMB_H * 2, cols * THUMB_W, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * THUMB_H * 2:(r + 1) * THUMB_H * 2, c * THUMB_W:(c + 1) * THUMB_W] = t
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, grid)
    print(f"\n몽타주 저장: {OUT_PATH} (위=da, 아래=ll, 초록=TP 빨강=FP(과다포함) 파랑=FN(과소포함))")


if __name__ == "__main__":
    main()
