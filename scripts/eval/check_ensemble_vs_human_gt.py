#!/usr/bin/env python3
"""딥앙상블 부트스트랩(medium x5, seed 0~4, §2.17) + 기존 배포모델(medium_v2,
pseudo_dataset_v2 1081장 학습, outputs/models/best.onnx)까지 총 6개 모델의 da
소프트보팅 결과를 사람 GT(bootstrap_v2, 156장)와 비교. check_medium_v2_vs_human_gt.py
와 동일 포맷(TP=초록/FP=빨강/FN=파랑 오버레이)으로 만들어서 단일 medium_v2 모델
(da IoU 0.904, FP 6.2%/FN 4.2%, §2.19)과 직접 비교.
목적: §2.19에서 발견한 "화면 하단 중앙 고정 물체 da 오검출"이 앙상블 소프트보팅으로
줄었는지 확인 + 커브 프레임 위주로 시각 비교(da가 커브에서 트랙 밖으로 넘치는
편향이 §2.12에서 이미 확인됐던 문제라 커브에서 개선 여부가 제일 중요).

da: 6개 모델(5개 신규 부트스트랩 + 기존 배포모델 1개) 확률 평균(soft-vote) 후 0.5 임계값.
ll: 참고용(seed0 모델 하나로만 표시, 앙상블 대상 아님 - §2.17 피드백은 da 한정).
몽타주 프레임 선정: curve_result.csv(전체 2123장 곡률 점수) 기준 상위 곡선 프레임 위주.
"""
import os
import sys
from argparse import Namespace

import cv2
import numpy as np
import onnxruntime as ort
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "_TwinLiteNetPlus_ref"))
from model.model import TwinLiteNetPlus  # noqa: E402

ENSEMBLE_DIR = os.path.join(BASE, "outputs", "ensemble_bootstrap_v1")
N_SEEDS = 5
CONFIG = "medium"
IN_W, IN_H = 640, 384

DEPLOYED_ONNX = os.path.join(BASE, "outputs", "models", "best.onnx")

GT_DIR = os.path.join(BASE, "bootstrap_v2")
GT_IMAGES = os.path.join(GT_DIR, "images")
GT_DA = os.path.join(GT_DIR, "da_masks")
GT_LL = os.path.join(GT_DIR, "ll_masks")

CURVE_CSV = os.path.join(BASE, "curve_result.csv")

OUT_PATH = os.path.join(BASE, "outputs", "montages", "ensemble6_vs_human_gt_curves.png")
THUMB_W, THUMB_H = 320, 240
N_SHOW = 16

BOTTOM_OBJ_Y = (400, 480)
BOTTOM_OBJ_X = (220, 420)


def load_torch_models():
    models = []
    for seed in range(N_SEEDS):
        pth = os.path.join(ENSEMBLE_DIR, f"seed{seed}", "best.pth")
        assert os.path.isfile(pth), f"{pth} 없음 - 학습이 아직 안 끝났을 수 있음"
        m = TwinLiteNetPlus(Namespace(config=CONFIG))
        state = torch.load(pth, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    print(f"신규 부트스트랩 앙상블 {len(models)}개 로드 완료 (156장 학습)")
    return models


def load_deployed_onnx():
    assert os.path.isfile(DEPLOYED_ONNX), f"{DEPLOYED_ONNX} 없음"
    sess = ort.InferenceSession(DEPLOYED_ONNX, providers=["CPUExecutionProvider"])
    print("기존 배포모델(medium_v2, pseudo_dataset_v2 1081장 학습) 로드 완료")
    return sess


def softmax_fg_np(logits_2ch):
    m = logits_2ch.max(axis=0, keepdims=True)
    e = np.exp(logits_2ch - m)
    return (e / e.sum(axis=0, keepdims=True))[1]


def infer_da_prob_torch(model, blob):
    with torch.no_grad():
        out_da, _ = model(blob)
    return torch.softmax(out_da, dim=1)[0, 1].numpy()


def infer_da_prob_onnx(sess, blob_np):
    da_out, _ = sess.run(["da", "ll"], {"images": blob_np})
    return softmax_fg_np(da_out[0])


def infer_ll_mask(model, blob):
    with torch.no_grad():
        _, out_ll = model(blob)
    return torch.argmax(out_ll, dim=1)[0].numpy().astype(bool)


def to_blob(img0):
    resized = cv2.resize(img0, (IN_W, IN_H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])


def fp_fn_iou(pred, gt):
    tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt
    gt_area = max(gt.sum(), 1)
    return (fp.sum() / gt_area, fn.sum() / gt_area, tp.sum() / max((pred | gt).sum(), 1))


def overlay(img0, pred, gt):
    vis = img0.copy()
    tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt
    vis[tp] = (0.5 * vis[tp].astype(np.float64) + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
    vis[fp] = (0.5 * vis[fp].astype(np.float64) + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    vis[fn] = (0.5 * vis[fn].astype(np.float64) + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    return vis


TRIAGE_CSV = os.path.join(BASE, "triage_result.csv")
CONE_CSV = os.path.join(BASE, "cone_detect_result.csv")
MISSING_CSV = os.path.join(BASE, "missing_line_result.csv")


def _load_csv(path):
    import csv
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[os.path.splitext(row["file"])[0]] = row
    return rows


def load_diverse_names(candidate_names, n_per_cat=4):
    """실패후보/콘/커브/일반(3선 다 잡힘) 4개 카테고리에서 골고루 뽑아 다양한 상황을
    한 몽타주에서 보게 함 - 커브 단독 선정 대신."""
    candidate_names = sorted(candidate_names)
    triage = _load_csv(TRIAGE_CSV) if os.path.isfile(TRIAGE_CSV) else {}
    cone = _load_csv(CONE_CSV) if os.path.isfile(CONE_CSV) else {}
    missing = _load_csv(MISSING_CSV) if os.path.isfile(MISSING_CSV) else {}
    curve_scores = {}
    if os.path.isfile(CURVE_CSV):
        with open(CURVE_CSV) as f:
            next(f)
            for line in f:
                fn, score = line.strip().rsplit(",", 1)
                curve_scores[os.path.splitext(fn)[0]] = float(score)

    fail = [n for n in candidate_names if triage.get(n, {}).get("failure_candidate") == "True"]
    curves = sorted([n for n in candidate_names if n in curve_scores],
                     key=lambda n: -curve_scores[n])
    cones = [n for n in candidate_names if int(cone.get(n, {}).get("n_cones", 0)) > 0]
    easy = [n for n in candidate_names if missing.get(n, {}).get("n_detected") == "3"]

    chosen, used = [], set()
    for label, pool in [("실패후보", fail), ("커브", curves), ("콘", cones), ("일반성공", easy)]:
        picked = 0
        for n in pool:
            if n in used:
                continue
            chosen.append(n)
            used.add(n)
            picked += 1
            if picked >= n_per_cat:
                break
        print(f"  {label}: {picked}장 선정")

    # 카테고리 합쳐서 n_per_cat*4에 못 미치면 나머지에서 랜덤으로 채움
    remaining = [n for n in candidate_names if n not in used]
    rng = np.random.RandomState(42)
    while len(chosen) < n_per_cat * 4 and remaining:
        pick = remaining.pop(int(rng.randint(len(remaining))))
        chosen.append(pick)

    return chosen[:n_per_cat * 4]


def main():
    torch_models = load_torch_models()
    onnx_sess = load_deployed_onnx()

    names = sorted(os.path.splitext(f)[0] for f in os.listdir(GT_IMAGES))
    print(f"대상: {len(names)}장 (bootstrap_v2)")
    show_names = set(load_curve_ranked_names(names))
    print(f"몽타주 표시: 커브 점수 상위 {len(show_names)}장 - {sorted(show_names)}")

    da_fp, da_fn, da_iou = [], [], []
    ll_fp, ll_fn, ll_iou = [], [], []
    bottom_fp_pixels, bottom_gt_pixels = 0, 0
    thumbs = []

    for i, n in enumerate(names):
        img0 = cv2.imread(os.path.join(GT_IMAGES, n + ".png"))
        h, w = img0.shape[:2]
        gt_da = cv2.imread(os.path.join(GT_DA, n + ".png"), cv2.IMREAD_GRAYSCALE) > 127
        gt_ll = cv2.imread(os.path.join(GT_LL, n + ".png"), cv2.IMREAD_GRAYSCALE) > 127

        blob_np = to_blob(img0)
        blob_torch = torch.from_numpy(blob_np).float()

        probs = [infer_da_prob_torch(m, blob_torch) for m in torch_models]
        probs.append(infer_da_prob_onnx(onnx_sess, blob_np))
        avg_prob = np.mean(probs, axis=0)
        pred_da = cv2.resize(avg_prob, (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5
        pred_ll = cv2.resize(infer_ll_mask(torch_models[0], blob_torch).astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        fp, fn, iou = fp_fn_iou(pred_da, gt_da)
        da_fp.append(fp); da_fn.append(fn); da_iou.append(iou)
        fp, fn, iou = fp_fn_iou(pred_ll, gt_ll)
        ll_fp.append(fp); ll_fn.append(fn); ll_iou.append(iou)

        y0, y1 = BOTTOM_OBJ_Y
        x0, x1 = BOTTOM_OBJ_X
        region_pred, region_gt = pred_da[y0:y1, x0:x1], gt_da[y0:y1, x0:x1]
        bottom_fp_pixels += (region_pred & ~region_gt).sum()
        bottom_gt_pixels += max((region_pred | region_gt).sum(), 1)

        if n in show_names:
            vis_da = overlay(img0, pred_da, gt_da)
            vis_ll = overlay(img0, pred_ll, gt_ll)
            top = cv2.resize(vis_da, (THUMB_W, THUMB_H))
            bot = cv2.resize(vis_ll, (THUMB_W, THUMB_H))
            cv2.rectangle(top, (0, 0), (THUMB_W, 16), (0, 0, 0), -1)
            cv2.putText(top, f"{n} da(6ensemble) IoU={da_iou[-1]:.2f}", (2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.rectangle(bot, (0, 0), (THUMB_W, 16), (0, 0, 0), -1)
            cv2.putText(bot, f"{n} ll(seed0) IoU={ll_iou[-1]:.2f}", (2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            thumbs.append((n, np.vstack([top, bot])))

        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(names)}")

    print(f"\n=== da 소프트보팅 앙상블 6개(신규5+기존배포1) (n={len(names)}) ===")
    print(f"mean IoU={np.mean(da_iou):.3f}  FP/GT={np.mean(da_fp)*100:.1f}%  FN/GT={np.mean(da_fn)*100:.1f}%")
    print("(비교: 기존 단일 medium_v2 - IoU=0.904 FP=6.2% FN=4.2% / 신규5개만 - IoU=0.901 FP=4.5% FN=5.9%, §2.19)")
    print(f"\n=== ll (seed0 참고, 앙상블 대상 아님, n={len(names)}) - FP비율은 GT가 거의 없는 프레임에서 왜곡될 수 있어 참고만 ===")
    print(f"mean IoU={np.mean(ll_iou):.3f}")
    print(f"\n=== 화면 하단 고정 물체 영역 targeted FP율 ===")
    print(f"6-앙상블: {bottom_fp_pixels/bottom_gt_pixels*100:.1f}%  (기존 단일모델: 25.8%, 신규5개만: 15.9%)")

    thumbs.sort(key=lambda t: t[0])
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    grid = np.zeros((rows * THUMB_H * 2, cols * THUMB_W, 3), dtype=np.uint8)
    for i, (_, t) in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * THUMB_H * 2:(r + 1) * THUMB_H * 2, c * THUMB_W:(c + 1) * THUMB_W] = t
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, grid)
    print(f"\n몽타주 저장: {OUT_PATH} (커브 위주, 위=da 6-앙상블, 아래=ll(seed0), 초록=TP 빨강=FP 파랑=FN)")


if __name__ == "__main__":
    main()
