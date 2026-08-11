#!/usr/bin/env python3
"""diet_dataset.py와 동일한 로직(콘 축소 재구성 + dHash 근접중복 제거)이되,
"성공 프레임 추가 다운샘플링" 단계를 SUCCESS_STRIDE=1(다운샘플 없음)로 둬서
dedup 상한까지 최대한 살린 버전 - 최종 1027장.
da/ll pseudo label을 자동 생성할 것이므로(사람 라벨링 병목 없음) 굳이 더 줄일
이유가 없다는 판단(2026-08-11, 사용자 확인) - PROGRESS.md §3 참고.
"""
import csv
import math
import os
import shutil
from collections import defaultdict

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE, "dataset")
OUT_DIR = os.path.join(BASE, "dataset_diet_v2")
MISSING_CSV = os.path.join(BASE, "missing_line_result.csv")
CONE_CSV = os.path.join(BASE, "cone_detect_result.csv")

HASH_SIZE = 8
DUP_HAMMING_THRESH = 6
CONE_RUN_GAP = 5
CONE_STRIDE = 5
SUCCESS_RUN_GAP = 5
SUCCESS_STRIDE = 1  # diet_dataset.py는 3 - 여기선 dedup 상한까지 전부 유지
CONE_TARGET_RATIO = 0.15


def frame_idx(fname):
    return int(fname.replace("frame_", "").replace(".png", ""))


def dhash(gray):
    resized = cv2.resize(gray, (HASH_SIZE + 1, HASH_SIZE), interpolation=cv2.INTER_AREA)
    return (resized[:, 1:] > resized[:, :-1]).flatten()


def group_runs(indices, gap):
    indices = sorted(indices)
    if not indices:
        return []
    runs = [[indices[0]]]
    for i in indices[1:]:
        if i - runs[-1][-1] > gap:
            runs.append([i])
        else:
            runs[-1].append(i)
    return runs


def main():
    missing_rows = {r["file"]: r for r in csv.DictReader(open(MISSING_CSV))}
    cone_rows = {r["file"]: r for r in csv.DictReader(open(CONE_CSV))}

    no_cone = [f for f, r in cone_rows.items() if int(r["n_cones"]) == 0]
    cone_idx = {frame_idx(f): f for f, r in cone_rows.items() if int(r["n_cones"]) > 0}
    runs = group_runs(list(cone_idx.keys()), CONE_RUN_GAP)
    kept_cone = set()
    for run in runs:
        run_files = [cone_idx[i] for i in run]
        kept_cone.update(run_files[::CONE_STRIDE])
        best = max(run_files, key=lambda f: float(cone_rows[f]["max_conf"]))
        kept_cone.add(best)
    balanced_set = set(no_cone) | kept_cone
    files = sorted(balanced_set, key=frame_idx)
    print(f"1단계(콘축소 재구성): {len(files)}장")

    def is_failure(f):
        r = missing_rows.get(f)
        return r is not None and int(r["n_detected"]) < 3

    def is_cone(f):
        r = cone_rows.get(f)
        return r is not None and int(r["n_cones"]) > 0

    def priority(f):
        if is_failure(f):
            return 2
        if is_cone(f):
            return 1
        return 0

    hashes = {}
    for f in files:
        gray = cv2.imread(os.path.join(RAW_DIR, f), cv2.IMREAD_GRAYSCALE)
        hashes[f] = dhash(gray)

    n = len(files)
    reps_arr = np.zeros((n, 64), dtype=bool)
    n_reps = 0
    rep_cluster_ids = []
    clusters = defaultdict(list)
    for f in files:
        h = hashes[f]
        cid = None
        if n_reps > 0:
            dist = np.count_nonzero(reps_arr[:n_reps] != h, axis=1)
            min_idx = int(np.argmin(dist))
            if dist[min_idx] <= DUP_HAMMING_THRESH:
                cid = rep_cluster_ids[min_idx]
        if cid is None:
            reps_arr[n_reps] = h
            cid = n_reps
            rep_cluster_ids.append(cid)
            n_reps += 1
        clusters[cid].append(f)

    kept = set()
    for cid, members in clusters.items():
        if len(members) == 1:
            kept.update(members)
            continue
        forced = [f for f in members if priority(f) == 2]
        has_priority = any(priority(f) > 0 for f in members)
        keep_n = max(2, math.ceil(len(members) * 0.3)) if has_priority else max(1, math.ceil(len(members) * 0.15))
        members_sorted = sorted(members, key=lambda f: (-priority(f), frame_idx(f)))
        chosen = set(forced)
        for f in members_sorted:
            if len(chosen) >= max(keep_n, len(forced)):
                break
            chosen.add(f)
        kept.update(chosen)
    print(f"2단계(dHash 근접중복 제거 후): {len(kept)}장")

    success_files = [f for f in kept if priority(f) == 0]
    other_files = kept - set(success_files)
    by_idx = {frame_idx(f): f for f in success_files}
    runs = group_runs(list(by_idx.keys()), SUCCESS_RUN_GAP)
    success_kept = set()
    for run in runs:
        run_files = [by_idx[i] for i in run]
        strided = run_files[::SUCCESS_STRIDE]
        if not strided:
            strided = run_files[:1]
        success_kept.update(strided)
    final_kept = other_files | success_kept
    print(f"3단계(성공프레임 stride={SUCCESS_STRIDE}): {len(success_files)} -> {len(success_kept)}장")

    cone_final = [f for f in final_kept if is_cone(f)]
    non_cone_final = final_kept - set(cone_final)
    by_idx_cone = {frame_idx(f): f for f in cone_final}
    cone_runs = group_runs(list(by_idx_cone.keys()), SUCCESS_RUN_GAP)
    stride = 1
    cone_kept = set(cone_final)
    while cone_final and len(cone_kept) / (len(non_cone_final) + len(cone_kept)) > CONE_TARGET_RATIO and stride < 10:
        stride += 1
        cone_kept = set()
        for run in cone_runs:
            run_files = [by_idx_cone[i] for i in run]
            strided = run_files[::stride]
            if not strided:
                strided = run_files[:1]
            best = max(run_files, key=lambda f: float(cone_rows[f]["max_conf"]))
            cone_kept.update(strided)
            cone_kept.add(best)
    if stride > 1:
        print(f"콘 비중 초과 -> stride={stride}로 추가 다이어트: {len(cone_final)} -> {len(cone_kept)}장")
    final_kept = non_cone_final | cone_kept

    print(f"\n최종: {len(final_kept)}장")
    n_cone_final = sum(1 for f in final_kept if is_cone(f))
    n_fail_final = sum(1 for f in final_kept if is_failure(f))
    n_succ_final = len(final_kept) - n_cone_final - n_fail_final
    print(f"  실패후보: {n_fail_final} / 라바콘: {n_cone_final} / 일반성공: {n_succ_final}")

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    for f in final_kept:
        shutil.copy(os.path.join(RAW_DIR, f), os.path.join(OUT_DIR, f))
    print("저장 위치:", OUT_DIR)


if __name__ == "__main__":
    main()
