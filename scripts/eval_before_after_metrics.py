#!/usr/bin/env python3
"""pretrained(파인튜닝 0회) vs 40epoch 파인튜닝 모델을, 학습 때 쓴 것과 동일한
IOUEval 공식으로 val split(160장) 전체에 대해 정량 비교한다. README 성능 표용.
"""
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

PRETRAINED = os.path.join(NEW_REPO, "pretrained", "medium.pth")
FINETUNED = os.path.join(BASE_DIR, "finetune_out_medium_40ep", "best_ll.pth")

sys.path.insert(0, NEW_REPO)
from model.model import TwinLiteNetPlus  # noqa: E402
from IOUEval import SegmentationMetric  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_val_names():
    img_dir = os.path.join(PSEUDO_DIR, "images")
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".png"))
    import random
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


def load_label_onehot(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (IN_W, IN_H), interpolation=cv2.INTER_NEAREST)
    fg = (m > 0).astype(np.int64)
    return torch.from_numpy(fg).unsqueeze(0)  # [1,H,W] class-index gt


def evaluate(model, names):
    DA = SegmentationMetric(2)
    LL = SegmentationMetric(2)
    with torch.no_grad():
        for n in names:
            img0 = cv2.imread(os.path.join(PSEUDO_DIR, "images", n + ".png"))
            resized = cv2.resize(img0, (IN_W, IN_H))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])).float().to(device)

            out_da, out_ll = model(blob)
            da_pred = torch.argmax(out_da, dim=1).cpu()  # [1,H,W]
            ll_pred = torch.argmax(out_ll, dim=1).cpu()

            da_gt = load_label_onehot(os.path.join(PSEUDO_DIR, "da_masks", n + ".png"))
            ll_gt = load_label_onehot(os.path.join(PSEUDO_DIR, "ll_masks", n + ".png"))

            DA.reset()
            DA.addBatch(da_pred[0].numpy(), da_gt[0].numpy())
            LL.reset()
            LL.addBatch(ll_pred[0].numpy(), ll_gt[0].numpy())

            da_mious = DA.meanIntersectionOverUnion()
            ll_acc = LL.lineAccuracy()
            ll_iou = LL.IntersectionOverUnion()

            yield n, da_mious, ll_acc, ll_iou


def main():
    names = build_val_names()
    print(f"val {len(names)}장 평가")

    results = {}
    for label, weight in [("pretrained", PRETRAINED), ("finetuned_40ep", FINETUNED)]:
        model = load_model(weight)
        da_list, acc_list, iou_list = [], [], []
        for n, da_miou, ll_acc, ll_iou in evaluate(model, names):
            da_list.append(da_miou)
            acc_list.append(ll_acc)
            iou_list.append(ll_iou)
        results[label] = {
            "da_mIoU": float(np.mean(da_list)),
            "ll_Acc": float(np.mean(acc_list)),
            "ll_IOU": float(np.mean(iou_list)),
        }
        print(label, results[label])

    print("\n=== 요약 ===")
    for label, r in results.items():
        print(f"{label}: da_mIoU={r['da_mIoU']:.4f}  ll_Acc={r['ll_Acc']:.4f}  ll_IOU={r['ll_IOU']:.4f}")


if __name__ == "__main__":
    main()
