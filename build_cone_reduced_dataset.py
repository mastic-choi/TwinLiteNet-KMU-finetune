#!/usr/bin/env python3
"""cone_detect_result.csv 기반으로 라바콘 프레임이 과대표집된 dataset/를
정리한다. 라바콘 검출 프레임은 대부분 같은 통과 구간(라바콘 존)의 연속
프레임이라 근접 중복이 심함 — 구간(run)별로 골고루 솎아내고, 각 구간에서
신뢰도가 가장 높은(잘 검출된) 프레임은 항상 포함시켜 '오검출 경계 사례'만
남지 않게 한다. 콘이 없는 프레임은 원래도 과대표집이 아니므로 그대로 둔다.

결과: dataset_balanced/ (원본 파일명 유지, 심볼릭 링크 아님 — 복사)
"""
import csv
import os
import shutil

from ultralytics import YOLO

BASE = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE, "dataset")
CSV_PATH = os.path.join(BASE, "cone_detect_result.csv")
OUT_DIR = os.path.join(BASE, "dataset_balanced")
MONTAGE_PATH = os.path.join(BASE, "cone_kept_montage.png")
MODEL_PATH = os.path.expanduser("~/code/xycar_ws/src/yolo_ros/cone_best.pt")

RUN_GAP_THRESHOLD = 5   # 이 이상 프레임 번호가 벌어지면 다른 라바콘 구간(통과)으로 취급
STRIDE = 5              # 각 구간에서 몇 프레임마다 하나씩 남길지(근접 중복 제거)


def frame_idx(fname):
    return int(fname.replace("frame_", "").replace(".png", ""))


def group_runs(indices):
    indices = sorted(indices)
    runs = [[indices[0]]]
    for i in indices[1:]:
        if i - runs[-1][-1] > RUN_GAP_THRESHOLD:
            runs.append([i])
        else:
            runs[-1].append(i)
    return runs


def main():
    rows = {r["file"]: r for r in csv.DictReader(open(CSV_PATH))}
    cone_files = {f: r for f, r in rows.items() if int(r["n_cones"]) > 0}
    no_cone_files = [f for f in rows if f not in cone_files]

    by_idx = {frame_idx(f): f for f in cone_files}
    runs = group_runs(list(by_idx.keys()))
    print(f"라바콘 구간(run) {len(runs)}개, 총 {len(cone_files)}장")

    kept_cone_files = set()
    for run in runs:
        run_files = [by_idx[i] for i in run]
        # 구간 내 stride 샘플링(근접 중복 제거, 시간순 다양성 유지)
        strided = run_files[::STRIDE]
        # 구간 내 최고 신뢰도 프레임(=잘 검출된 예시)은 항상 포함
        best = max(run_files, key=lambda f: float(cone_files[f]["max_conf"]))
        kept_cone_files.update(strided)
        kept_cone_files.add(best)

    print(f"솎아낸 뒤 남는 라바콘 프레임: {len(kept_cone_files)}장 "
          f"({len(kept_cone_files) / len(cone_files):.1%} of 원래 콘 프레임)")

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    for f in no_cone_files:
        shutil.copy(os.path.join(DATASET_DIR, f), os.path.join(OUT_DIR, f))
    for f in kept_cone_files:
        shutil.copy(os.path.join(DATASET_DIR, f), os.path.join(OUT_DIR, f))

    total = len(no_cone_files) + len(kept_cone_files)
    print(f"\n최종 dataset_balanced/: {total}장 "
          f"(콘 없음 {len(no_cone_files)} + 콘 {len(kept_cone_files)}, "
          f"콘 비중 {len(kept_cone_files) / total:.1%})")
    print("저장 위치:", OUT_DIR)

    # ── 몽타주: 남긴 콘 프레임에 박스 그려서 확인용 ──
    import math
    import cv2
    import numpy as np

    model = YOLO(MODEL_PATH)
    kept_sorted = sorted(kept_cone_files, key=frame_idx)
    THUMB_W, THUMB_H = 320, 240
    thumbs = []
    for f in kept_sorted:
        fp = os.path.join(DATASET_DIR, f)
        result = model.predict(fp, conf=0.5, imgsz=640, verbose=False)[0]
        plotted = result.plot()  # BGR, 박스+신뢰도 그려진 이미지
        thumb = cv2.resize(plotted, (THUMB_W, THUMB_H))
        label = f"{f} conf={cone_files[f]['max_conf']}"
        cv2.rectangle(thumb, (0, THUMB_H - 16), (THUMB_W, THUMB_H), (0, 0, 0), -1)
        cv2.putText(thumb, label, (3, THUMB_H - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)

    cols = 6
    rows_n = math.ceil(len(thumbs) / cols)
    grid = np.zeros((rows_n * THUMB_H, cols * THUMB_W, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * THUMB_H:(r + 1) * THUMB_H, c * THUMB_W:(c + 1) * THUMB_W] = t
    cv2.imwrite(MONTAGE_PATH, grid)
    print("몽타주 저장:", MONTAGE_PATH)


if __name__ == "__main__":
    main()
