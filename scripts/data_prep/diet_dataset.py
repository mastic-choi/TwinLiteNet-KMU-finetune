#!/usr/bin/env python3
"""dataset_balanced/(콘 중복 이미 정리됨)를 추가로 다이어트한다.

1. perceptual hash(dHash)로 "시각적으로 거의 같은" 프레임을 클러스터링 —
   정지 상태에서 연속 촬영됐거나, 다른 바퀴에서 같은 지점을 지나 생긴
   시간상 멀리 떨어진 중복까지 잡아낸다(프레임 번호 근접 여부와 무관).
2. 클러스터 안에서는 우선순위(실패 후보 > 라바콘 > 그냥 성공)가 높은
   프레임을 최대한 보존하고, 나머지는 대표만 남긴다.
3. dedup 후에도 남아있는 "3개 선 다 잡힌(n_detected==3) + 콘 없음" 성공
   프레임은 구간(run)별 stride 샘플링으로 한 번 더 줄인다.
4. 라바콘 프레임은 이번 단계에서 더 줄이지 않는다(이미 build_cone_reduced_
   dataset.py에서 정리됨) — 최종 콘 비중이 과도해지면 경고만 출력.
"""
import csv
import math
import os
import shutil
from collections import defaultdict

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE, "dataset")          # dataset_balanced/가 삭제돼도 원본에서 재구성
OVERLAY_DIR = os.path.join(BASE, "dataset_overlay")
OUT_DIR = os.path.join(BASE, "dataset_diet")
MONTAGE_DIR = os.path.join(BASE, "diet_montages")
MISSING_CSV = os.path.join(BASE, "missing_line_result.csv")
CONE_CSV = os.path.join(BASE, "cone_detect_result.csv")

HASH_SIZE = 8          # dHash -> 64bit
DUP_HAMMING_THRESH = 6  # 이 이하 해밍거리면 "거의 같은 장면"으로 취급
SUCCESS_RUN_GAP = 5
SUCCESS_STRIDE = 3      # 남은 성공 프레임 구간에서 몇 장마다 하나 남길지

# build_cone_reduced_dataset.py와 동일한 로직/값 — dataset_balanced/ 없이도
# 원본 dataset/ + cone_detect_result.csv에서 그때와 같은 콘 서브셋을 재구성
CONE_RUN_GAP = 5
CONE_STRIDE = 5

MONTAGE_CHUNK = 120     # 몽타주 1장당 최대 타일 수(너무 커지지 않게 분할)
THUMB_W, THUMB_H = 200, 150


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


def rebuild_balanced_file_list(cone_rows):
    """dataset_balanced/ 없이 dataset/ + cone_detect_result.csv로 그때와 같은
    (콘 없음 전부 + 콘 있음 구간별 stride+최고신뢰도) 서브셋을 재구성한다."""
    all_files = {r for r in cone_rows}
    no_cone = [f for f, r in cone_rows.items() if int(r["n_cones"]) == 0]
    cone_idx = {frame_idx(f): f for f, r in cone_rows.items() if int(r["n_cones"]) > 0}
    runs = group_runs(list(cone_idx.keys()), CONE_RUN_GAP)
    kept_cone = set()
    for run in runs:
        run_files = [cone_idx[i] for i in run]
        kept_cone.update(run_files[::CONE_STRIDE])
        best = max(run_files, key=lambda f: float(cone_rows[f]["max_conf"]))
        kept_cone.add(best)
    return set(no_cone) | kept_cone


def main():
    missing_rows = {r["file"]: r for r in csv.DictReader(open(MISSING_CSV))}
    cone_rows = {r["file"]: r for r in csv.DictReader(open(CONE_CSV))}

    balanced_set = rebuild_balanced_file_list(cone_rows)
    files = sorted(balanced_set, key=frame_idx)
    print(f"입력 {len(files)}장 (dataset_balanced 재구성, 원본 dataset/에서 읽음)")

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

    # ── 1. dHash 계산 ──
    hashes = {}
    for f in files:
        gray = cv2.imread(os.path.join(RAW_DIR, f), cv2.IMREAD_GRAYSCALE)
        hashes[f] = dhash(gray)

    # ── 2. 그리디 클러스터링(프레임 번호 순서 무관하게 해밍거리로 비교) ──
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

    n_dup_clusters = sum(1 for m in clusters.values() if len(m) > 1)
    n_dup_frames = sum(len(m) for m in clusters.values() if len(m) > 1)
    print(f"유사 클러스터 {len(clusters)}개 (그 중 {n_dup_clusters}개가 2장 이상 묶임, "
          f"총 {n_dup_frames}장이 중복군에 속함)")

    kept = set()
    for cid, members in clusters.items():
        if len(members) == 1:
            kept.update(members)
            continue
        forced = [f for f in members if priority(f) == 2]  # 실패 후보는 무조건 보존
        has_priority = any(priority(f) > 0 for f in members)
        keep_n = max(2, math.ceil(len(members) * 0.3)) if has_priority else max(1, math.ceil(len(members) * 0.15))
        members_sorted = sorted(members, key=lambda f: (-priority(f), frame_idx(f)))
        chosen = set(forced)
        for f in members_sorted:
            if len(chosen) >= max(keep_n, len(forced)):
                break
            chosen.add(f)
        kept.update(chosen)

    print(f"dedup 후: {len(kept)}장 (제거 {len(files) - len(kept)}장)")

    # ── 3. 남은 "성공(3선 다 잡힘) + 콘 없음" 프레임 추가 stride 다운샘플 ──
    success_files = [f for f in kept if priority(f) == 0]
    other_files = kept - set(success_files)
    print(f"  그 중 성공/콘없음(추가 다이어트 대상): {len(success_files)}장, "
          f"실패후보+콘(보존): {len(other_files)}장")

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
    print(f"성공 프레임 추가 다이어트: {len(success_files)} -> {len(success_kept)}장")

    # ── 4. 다른 카테고리가 더 많이 깎여 콘 비중이 상대적으로 부풀었으면
    #    콘도 같은 방식(구간별 stride)으로 추가로 줄인다 — 이번엔 target
    #    비중(<=15%)을 만족할 때까지 stride를 키운다. 각 구간의 최고
    #    신뢰도 프레임(잘 검출된 예시)은 항상 남긴다.
    CONE_TARGET_RATIO = 0.15
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
        print(f"콘 비중 초과 감지 -> 콘도 구간별 stride={stride}로 추가 다이어트: "
              f"{len(cone_final)} -> {len(cone_kept)}장 (최고 신뢰도 프레임은 항상 보존)")

    final_kept = non_cone_final | cone_kept
    print(f"\n최종 선택: {len(final_kept)}장 (원본 dataset_balanced 대비 {len(final_kept)/len(files):.1%})")

    n_cone_final = sum(1 for f in final_kept if is_cone(f))
    n_fail_final = sum(1 for f in final_kept if is_failure(f))
    n_succ_final = len(final_kept) - n_cone_final - n_fail_final
    print(f"  실패후보(선 미검출): {n_fail_final}장 ({n_fail_final/len(final_kept):.1%})")
    print(f"  라바콘: {n_cone_final}장 ({n_cone_final/len(final_kept):.1%})")
    print(f"  일반 성공: {n_succ_final}장 ({n_succ_final/len(final_kept):.1%})")

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    for f in final_kept:
        shutil.copy(os.path.join(RAW_DIR, f), os.path.join(OUT_DIR, f))
    print("저장 위치:", OUT_DIR)

    # ── 몽타주: dataset_overlay/(이미 da/ll 박스 그려둔 이미지) 재사용, 결과가
    #    많으니 MONTAGE_CHUNK장 단위로 나눠 여러 장으로 저장 ──
    if os.path.exists(MONTAGE_DIR):
        shutil.rmtree(MONTAGE_DIR)
    os.makedirs(MONTAGE_DIR)

    def category(f):
        if is_failure(f):
            return "FAIL"
        if is_cone(f):
            return "CONE"
        return "OK"

    ordered = sorted(final_kept, key=frame_idx)
    n_chunks = math.ceil(len(ordered) / MONTAGE_CHUNK)
    cols = 10
    for c in range(n_chunks):
        chunk = ordered[c * MONTAGE_CHUNK:(c + 1) * MONTAGE_CHUNK]
        thumbs = []
        for f in chunk:
            overlay_fp = os.path.join(OVERLAY_DIR, f)
            img = cv2.imread(overlay_fp)
            if img is None:
                img = cv2.imread(os.path.join(RAW_DIR, f))
            thumb = cv2.resize(img, (THUMB_W, THUMB_H))
            label = f"{category(f)} {f.replace('frame_', '').replace('.png', '')}"
            cv2.rectangle(thumb, (0, THUMB_H - 14), (THUMB_W, THUMB_H), (0, 0, 0), -1)
            cv2.putText(thumb, label, (2, THUMB_H - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
            thumbs.append(thumb)

        rows_n = math.ceil(len(thumbs) / cols)
        grid = np.zeros((rows_n * THUMB_H, cols * THUMB_W, 3), dtype=np.uint8)
        for i, t in enumerate(thumbs):
            r, cc = divmod(i, cols)
            grid[r * THUMB_H:(r + 1) * THUMB_H, cc * THUMB_W:(cc + 1) * THUMB_W] = t
        out_fp = os.path.join(MONTAGE_DIR, f"diet_montage_{c + 1:02d}.png")
        cv2.imwrite(out_fp, grid)

    print(f"몽타주 {n_chunks}장 저장: {MONTAGE_DIR}")


if __name__ == "__main__":
    main()
