#!/usr/bin/env python3
"""nano/small/medium/large 4개 config의 파라미터 수 + medium(우리가 실제 쓰는 config)의
RX 9070 XT(ROCm) 추론 속도(fps)를 측정. README "모델 크기/속도" 표용."""
import os
import sys
import time
from argparse import Namespace

import torch

BASE_DIR = os.path.expanduser("~/fine-tune")
NEW_REPO = os.path.join(BASE_DIR, "TwinLiteNetPlus")
sys.path.insert(0, NEW_REPO)
from model.model import TwinLiteNetPlus  # noqa: E402
from utils import netParams  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

for cfg in ["nano", "small", "medium", "large"]:
    model = TwinLiteNetPlus(Namespace(config=cfg))
    n_params = netParams(model)
    print(f"{cfg:8s} params={n_params:,}")

# 실제 쓰는 medium config로 fps 측정 (batch=1, 640x384, GPU)
model = TwinLiteNetPlus(Namespace(config="medium")).to(device).eval()
dummy = torch.zeros(1, 3, 384, 640).to(device)

with torch.no_grad():
    for _ in range(10):  # warmup
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    N = 100
    for _ in range(N):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

fps = N / elapsed
print(f"\nmedium config 추론 속도 (batch=1, {device}): {fps:.1f} fps ({elapsed/N*1000:.2f} ms/frame)")
