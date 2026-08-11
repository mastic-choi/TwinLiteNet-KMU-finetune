#!/usr/bin/env python3
"""pseudo_dataset(완전 자동 생성, §2.9)에 사람이 CVAT에서 실제로 완료한 라벨(bootstrap/)을
병합 — 해당 프레임은 자동 라벨 대신 사람 라벨을 우선 사용한다.

- da: bootstrap/da_masks 그대로 (사람이 직접 그린 정답)
- ll: bootstrap_refined/ll_masks 사용 (사람이 그린 위치 + skeleton 정제로 폭 8px 고정)
  -> pseudo_dataset의 나머지 프레임(YOLOPv2+skeleton, 동일 8px)과 두께 컨벤션을 맞춰서
  같은 데이터셋 안에서 GT 두께가 프레임마다 들쭉날쭉해지는 걸 방지.
- images: 이미 같은 원본이라 안 바꿔도 되지만 혹시 모르니 bootstrap 쪽으로 통일.

주의: bootstrap/는 예전에 export_cvat_bootstrap.py로 "completed" 확정된 74장 스냅샷이라,
그 뒤로 팀원이 CVAT에서 더 완료했을 수 있는 프레임은 여기 반영 안 됨 — 최신 Jobs CSV
받으면 다시 이 스크립트 방식으로 갱신할 것(PROGRESS.md §3 item 1).
"""
import os
import shutil

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PSEUDO_DIR = os.path.join(BASE, "pseudo_dataset")
BOOTSTRAP_DIR = os.path.join(BASE, "bootstrap")
BOOTSTRAP_REFINED_DIR = os.path.join(BASE, "bootstrap_refined")


def main():
    human_frames = sorted(os.listdir(os.path.join(BOOTSTRAP_DIR, "images")))
    n_replaced = 0
    for fname in human_frames:
        pseudo_img = os.path.join(PSEUDO_DIR, "images", fname)
        if not os.path.isfile(pseudo_img):
            print(f"경고: {fname}이 pseudo_dataset에 없음, 스킵")
            continue

        shutil.copy(os.path.join(BOOTSTRAP_DIR, "images", fname), pseudo_img)
        shutil.copy(os.path.join(BOOTSTRAP_DIR, "da_masks", fname),
                    os.path.join(PSEUDO_DIR, "da_masks", fname))
        shutil.copy(os.path.join(BOOTSTRAP_REFINED_DIR, "ll_masks", fname),
                    os.path.join(PSEUDO_DIR, "ll_masks", fname))
        n_replaced += 1

    total = len(os.listdir(os.path.join(PSEUDO_DIR, "images")))
    print(f"사람 라벨로 교체: {n_replaced}/{len(human_frames)}장")
    print(f"pseudo_dataset 전체 {total}장 중 사람 검증 {n_replaced}장 + 자동 생성 {total - n_replaced}장")


if __name__ == "__main__":
    main()
