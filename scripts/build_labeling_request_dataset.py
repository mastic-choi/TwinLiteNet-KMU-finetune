#!/usr/bin/env python3
"""다른 라벨러에게 요청할 "old TwinLiteNet이 약한 프레임" 모음 데이터셋 생성.
- 실패 후보: triage_result.csv의 failure_candidate=True (ll 미검출/da 미검출/글레어)
- 콘: cone_detect_result.csv n_cones>0, 신뢰도 높은 순
- 커브: curve_result.csv curve_score 높은 순 (detect_curve_frames.py)

이미 raw/images_todo/(470장, 작업 중)에 있는 프레임은 제외 - 중복 라벨링 방지.
perceptual hash로 근접 중복(연속 프레임)도 걸러냄(diet_dataset.py와 동일 방식).
카테고리별 상한을 둬서 특정 유형이 너무 많이 뽑히지 않게 함.
"""
import csv
import os
import shutil

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE, "dataset")
ALREADY_WORKING = set(os.listdir(os.path.join(BASE, "raw", "images_todo")))
OUT_DIR = os.path.join(BASE, "labeling_request_hard_frames")

CAP_FAILURE = 60
CAP_CONE = 40
CAP_CURVE = 40
DHASH_SIZE = 8
DHASH_MIN_DIST = 6  # 이보다 가까우면(비슷하면) 근접 중복으로 보고 스킵


def dhash(img_path, hash_size=DHASH_SIZE):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (hash_size + 1, hash_size))
    diff = small[:, 1:] > small[:, :-1]
    return diff.flatten()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def load_csv(name):
    with open(os.path.join(BASE, name)) as f:
        return list(csv.DictReader(f))


def main():
    failure_rows = [r for r in load_csv("triage_result.csv") if r["failure_candidate"] == "True"]
    cone_rows = sorted(
        [r for r in load_csv("cone_detect_result.csv") if int(r["n_cones"]) > 0],
        key=lambda r: -float(r["max_conf"]))
    curve_rows = sorted(load_csv("curve_result.csv"), key=lambda r: -float(r["curve_score"]))

    candidates = []  # (file, reason)
    for r in failure_rows:
        candidates.append((r["file"], "failure"))
    for r in cone_rows:
        candidates.append((r["file"], "cone"))
    for r in curve_rows:
        if float(r["curve_score"]) < 15:
            break
        candidates.append((r["file"], "curve"))

    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
    kept_hashes = []
    kept = []
    cap = {"failure": CAP_FAILURE, "cone": CAP_CONE, "curve": CAP_CURVE}
    count = {"failure": 0, "cone": 0, "curve": 0}
    seen_files = set()

    for fname, reason in candidates:
        if fname in ALREADY_WORKING or fname in seen_files:
            continue
        if count[reason] >= cap[reason]:
            continue
        fp = os.path.join(DATASET_DIR, fname)
        if not os.path.isfile(fp):
            continue
        h = dhash(fp)
        if any(hamming(h, kh) < DHASH_MIN_DIST for kh in kept_hashes):
            continue

        kept_hashes.append(h)
        seen_files.add(fname)
        count[reason] += 1
        kept.append((fname, reason))
        shutil.copy(fp, os.path.join(OUT_DIR, "images", fname))

    with open(os.path.join(OUT_DIR, "reasons.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "reason"])
        writer.writerows(sorted(kept))

    print(f"총 {len(kept)}장 -> {OUT_DIR}")
    print("카테고리별:", count)


if __name__ == "__main__":
    main()
