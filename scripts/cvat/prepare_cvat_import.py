#!/usr/bin/env python3
"""raw/da_masks_draft + raw/ll_masks_draft(이미 만들어둔 470장 초안 이진 마스크)를
CVAT의 'Segmentation mask 1.1' import 포맷으로 합쳐서 zip으로 만든다.

CVAT에 이 zip을 upload annotations 하면 우리가 만든 초안이 그대로 pre-annotation으로
깔려서, 팀원들은 처음부터 그리지 않고 브러시로 틀린 부분만 고치면 된다.

포맷(공식 문서 기준, https://docs.cvat.ai/docs/dataset_management/formats/format-smask/):
  archive.zip/
    labelmap.txt
    ImageSets/Segmentation/default.txt   # 확장자 없는 파일명 목록
    SegmentationClass/<file>.png         # 3채널 RGB, 픽셀색=클래스

한 픽셀에 클래스 하나만 들어갈 수 있어서(da/ll이 겹치는 영역), 차선은 도로 위에
그려진 것이므로 겹치는 픽셀은 lane_line으로 우선 표시한다(더 좁고 중요한 클래스).
나중에 CVAT에서 export한 걸 다시 두 개의 이진 마스크로 되돌릴 때
(export_cvat_masks.py, 아직 안 만듦) drivable_area = drivable_area 색 OR lane_line 색
으로 복원하면 원래 의미(차선 밑도 도로)가 유지된다.
"""
import os
import shutil
import zipfile

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGES_DIR = os.path.join(BASE, "raw", "images_todo")
DA_DIR = os.path.join(BASE, "raw", "da_masks_draft")
LL_DIR = os.path.join(BASE, "raw", "ll_masks_draft")
OUT_DIR = os.path.join(BASE, "cvat_import")
ZIP_PATH = os.path.join(BASE, "cvat_import_masks.zip")

BG_COLOR = (0, 0, 0)        # RGB
DA_COLOR = (0, 128, 0)      # RGB - drivable_area
LL_COLOR = (128, 0, 0)      # RGB - lane_line

LABELMAP = """\
background:0,0,0::
drivable_area:0,128,0::
lane_line:128,0,0::
"""


def main():
    files = sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith(".png"))
    assert files, f"{IMAGES_DIR}에 png 없음"
    print(f"{len(files)}장 CVAT import용 마스크 변환")

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    seg_class_dir = os.path.join(OUT_DIR, "SegmentationClass")
    imagesets_dir = os.path.join(OUT_DIR, "ImageSets", "Segmentation")
    os.makedirs(seg_class_dir, exist_ok=True)
    os.makedirs(imagesets_dir, exist_ok=True)

    with open(os.path.join(OUT_DIR, "labelmap.txt"), "w") as f:
        f.write(LABELMAP)

    basenames = []
    for f in files:
        base = os.path.splitext(f)[0]
        basenames.append(base)

        da = cv2.imread(os.path.join(DA_DIR, f), cv2.IMREAD_GRAYSCALE)
        ll = cv2.imread(os.path.join(LL_DIR, f), cv2.IMREAD_GRAYSCALE)
        assert da is not None and ll is not None, f"{f} 마스크 로드 실패"

        h, w = da.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:] = BG_COLOR
        rgb[da > 0] = DA_COLOR
        rgb[ll > 0] = LL_COLOR  # ll이 da 위에 그려진 것이므로 겹치면 ll 우선 표시

        # cv2는 BGR로 쓰므로 RGB 배열을 뒤집어서 저장
        cv2.imwrite(os.path.join(seg_class_dir, f), rgb[:, :, ::-1])

    with open(os.path.join(imagesets_dir, "default.txt"), "w") as f:
        f.write("\n".join(basenames) + "\n")

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, filenames in os.walk(OUT_DIR):
            for fn in filenames:
                fp = os.path.join(root, fn)
                arcname = os.path.relpath(fp, OUT_DIR)
                zf.write(fp, arcname)

    print("완료:", ZIP_PATH)
    print(f"  labelmap: background(검정) / drivable_area(초록) / lane_line(빨강)")


if __name__ == "__main__":
    main()
