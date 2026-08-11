#!/usr/bin/env python3
"""지금 타협한 수준의 ll-clip 로직을 val split(160장) 전체 da 라벨에 적용해서,
40epoch 파인튜닝 모델의 예측을 (1) 원본 da 라벨 대비, (2) 크롭된 da 라벨 대비
각각 mIoU로 비교. 학습에 실제로 쓰인 IOUEval 공식 그대로 사용."""
import os
import sys
from argparse import Namespace

import cv2
import numpy as np
import torch

BASE_DIR = os.path.expanduser("~/fine-tune")
NEW_REPO = os.path.join(BASE_DIR, "TwinLiteNetPlus")
PSEUDO_DIR = os.path.join(BASE_DIR, "pseudo_dataset_v2")
CONFIG = "medium"
IN_W, IN_H = 640, 384
VAL_RATIO, SEED = 0.15, 42

FINETUNED = os.path.join(BASE_DIR, "finetune_out_medium_40ep", "best_ll.pth")

sys.path.insert(0, NEW_REPO)
from model.model import TwinLiteNetPlus  # noqa: E402
from IOUEval import SegmentationMetric  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pseudo_label"))
from make_da_clip_dataset_montage import clip_da_with_ll  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_val_names():
    import random
    img_dir = os.path.join(PSEUDO_DIR, "images")
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".png"))
    shuffled = names[:]
    random.Random(SEED).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_RATIO))
    return sorted(shuffled[:n_val])


def load_model(weight_path):
    model = TwinLiteNetPlus(Namespace(config=CONFIG))
    state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def to_class_idx(mask_bool):
    m = cv2.resize(mask_bool.astype(np.uint8) * 255, (IN_W, IN_H), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy((m > 0).astype(np.int64)).unsqueeze(0)


def main():
    names = build_val_names()
    print(f"val {len(names)}장 평가")

    model = load_model(FINETUNED)

    DA_orig = SegmentationMetric(2)
    DA_clip = SegmentationMetric(2)
    orig_ious, clip_ious = [], []
    n_changed = 0

    with torch.no_grad():
        for n in names:
            img0 = cv2.imread(os.path.join(PSEUDO_DIR, "images", n + ".png"))
            da_native = cv2.imread(os.path.join(PSEUDO_DIR, "da_masks", n + ".png"), cv2.IMREAD_GRAYSCALE) > 0
            ll_native = cv2.imread(os.path.join(PSEUDO_DIR, "ll_masks", n + ".png"), cv2.IMREAD_GRAYSCALE) > 0

            da_clipped = clip_da_with_ll(da_native, ll_native)
            if not np.array_equal(da_native, da_clipped):
                n_changed += 1

            resized = cv2.resize(img0, (IN_W, IN_H))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])).float().to(device)
            out_da, _ = model(blob)
            da_pred = torch.argmax(out_da, dim=1).cpu()

            da_gt_orig = to_class_idx(da_native)
            da_gt_clip = to_class_idx(da_clipped)

            DA_orig.reset()
            DA_orig.addBatch(da_pred[0].numpy(), da_gt_orig[0].numpy())
            DA_clip.reset()
            DA_clip.addBatch(da_pred[0].numpy(), da_gt_clip[0].numpy())

            orig_ious.append(DA_orig.meanIntersectionOverUnion())
            clip_ious.append(DA_clip.meanIntersectionOverUnion())

    print(f"\nda 라벨이 실제로 바뀐 프레임: {n_changed}/{len(names)}")
    print(f"모델 예측 vs 원본 da 라벨   : mIoU = {np.mean(orig_ious):.4f}")
    print(f"모델 예측 vs 크롭된 da 라벨 : mIoU = {np.mean(clip_ious):.4f}")


if __name__ == "__main__":
    main()
