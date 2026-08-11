#!/usr/bin/env python3
"""merge_human_labels.py의 v2 - pseudo_dataset_v2(1027장, da=bootstrap_v2 모델,
ll=YOLOPv2+skeleton)에 사람 라벨 134장(기존74+신규60) 병합.

- da: bootstrap_v2/da_masks 그대로 (전부 사람이 직접 그린 정답)
- ll: **주의(버그 수정)** - bootstrap_v2/ll_masks는 기존 74장 몫이 원본 브러시
  두께(bootstrap/, §2.7에서 문제됐던 들쭉날쭉 버전)로 잘못 복사돼 있었음. 기존
  74장은 skeleton 정제판(bootstrap_refined/ll_masks, 8px 고정)을 쓰고, 신규
  60장만 bootstrap_v2/ll_masks(YOLOPv2+skeleton, 이미 8px)를 그대로 씀 - 두
  소스 다 동일한 8px 두께 컨벤션이라 섞여도 일관됨.
"""
import os
import shutil

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PSEUDO_DIR = os.path.join(BASE, "pseudo_dataset_v2")
BOOTSTRAP_V2_DIR = os.path.join(BASE, "bootstrap_v2")
BOOTSTRAP_REFINED_DIR = os.path.join(BASE, "bootstrap_refined")


def main():
    original_74 = set(os.listdir(os.path.join(BOOTSTRAP_REFINED_DIR, "images")))
    human_frames = sorted(os.listdir(os.path.join(BOOTSTRAP_V2_DIR, "images")))
    n_replaced, n_missing = 0, 0
    for fname in human_frames:
        pseudo_img = os.path.join(PSEUDO_DIR, "images", fname)
        if not os.path.isfile(pseudo_img):
            print(f"경고: {fname}이 pseudo_dataset_v2에 없음 (dedup 단계에서 제외됐을 수 있음) -> 별도 추가 필요")
            n_missing += 1
            continue

        shutil.copy(os.path.join(BOOTSTRAP_V2_DIR, "images", fname), pseudo_img)
        shutil.copy(os.path.join(BOOTSTRAP_V2_DIR, "da_masks", fname),
                    os.path.join(PSEUDO_DIR, "da_masks", fname))
        ll_src = (os.path.join(BOOTSTRAP_REFINED_DIR, "ll_masks", fname) if fname in original_74
                  else os.path.join(BOOTSTRAP_V2_DIR, "ll_masks", fname))
        shutil.copy(ll_src, os.path.join(PSEUDO_DIR, "ll_masks", fname))
        n_replaced += 1

    # dedup(1027장 선별)에서 빠진 사람 라벨 프레임은 별도로 추가 - 사람이 직접 그린
    # 귀한 라벨이라 스킵하지 않고 pseudo_dataset_v2에 새로 편입시킴
    n_added = 0
    for fname in human_frames:
        pseudo_img = os.path.join(PSEUDO_DIR, "images", fname)
        if os.path.isfile(pseudo_img):
            continue
        shutil.copy(os.path.join(BOOTSTRAP_V2_DIR, "images", fname), pseudo_img)
        shutil.copy(os.path.join(BOOTSTRAP_V2_DIR, "da_masks", fname),
                    os.path.join(PSEUDO_DIR, "da_masks", fname))
        ll_src = (os.path.join(BOOTSTRAP_REFINED_DIR, "ll_masks", fname) if fname in original_74
                  else os.path.join(BOOTSTRAP_V2_DIR, "ll_masks", fname))
        shutil.copy(ll_src, os.path.join(PSEUDO_DIR, "ll_masks", fname))
        n_added += 1

    total = len(os.listdir(os.path.join(PSEUDO_DIR, "images")))
    print(f"사람 라벨로 교체: {n_replaced}장 / dedup에서 빠져서 새로 추가: {n_added}장")
    print(f"pseudo_dataset_v2 최종 {total}장 (사람 검증 {n_replaced + n_added} + 자동생성 {total - n_replaced - n_added})")


if __name__ == "__main__":
    main()
