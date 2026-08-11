#!/usr/bin/env python3
"""40epoch 파인튜닝 결과(best_ll.pth, da/ll 둘 다 이 체크포인트가 best.pth와 동급 이상)를
onnx로 export하고 PyTorch 출력과 오차 검증까지 수행. track_drive의 TwinLiteNetEngine과
바로 호환(DL_INPUT_NAME='images', DL_OUTPUT_NAMES=('da','ll'))."""
import os
import sys
from argparse import Namespace

import numpy as np
import onnxruntime as ort
import torch

BASE_DIR = os.path.expanduser("~/fine-tune")
NEW_REPO = os.path.join(BASE_DIR, "TwinLiteNetPlus")
WEIGHT_PATH = os.path.join(BASE_DIR, "finetune_out_medium_40ep", "best_ll.pth")
CONFIG = "medium"
ONNX_OUT = os.path.join(BASE_DIR, "best.onnx")

sys.path.insert(0, NEW_REPO)
from model.model import TwinLiteNetPlus

model = TwinLiteNetPlus(Namespace(config=CONFIG))
state = torch.load(WEIGHT_PATH, map_location="cpu")
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
model.load_state_dict(state)
model.eval()

dummy = torch.zeros(1, 3, 384, 640)
torch.onnx.export(
    model, dummy, ONNX_OUT,
    input_names=["images"], output_names=["da", "ll"],
    dynamic_axes={"images": {0: "batch"}, "da": {0: "batch"}, "ll": {0: "batch"}},
    opset_version=12,
)
print("내보내기 완료:", ONNX_OUT)

sess = ort.InferenceSession(ONNX_OUT, providers=["CPUExecutionProvider"])
x = dummy.numpy()
onnx_da, onnx_ll = sess.run(["da", "ll"], {"images": x})

with torch.no_grad():
    torch_da, torch_ll = model(dummy)

da_diff = np.abs(onnx_da - torch_da.numpy()).max()
ll_diff = np.abs(onnx_ll - torch_ll.numpy()).max()
print(f"da 최대 오차: {da_diff:.6f} | ll 최대 오차: {ll_diff:.6f} (1e-3 이하면 정상)")
assert da_diff < 1e-3 and ll_diff < 1e-3, "ONNX 변환 결과가 PyTorch와 다름"
print("검증 통과")
print("onnx 파일 크기:", os.path.getsize(ONNX_OUT) / 1024, "KB")
if os.path.isfile(ONNX_OUT + ".data"):
    print("외부 데이터 파일도 생성됨:", ONNX_OUT + ".data",
          os.path.getsize(ONNX_OUT + ".data") / 1024, "KB")
