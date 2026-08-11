# TwinLiteNet+ KMU Track Fine-tune

국민대(KMU) 자율주행 소형 RC카 프로젝트 [UMK/track_drive](https://github.com/mastic-choi/UMK)의
차선/주행가능영역 인식 모델을 우리 트랙 도메인에 맞게 파인튜닝하는 프로젝트.

## 배경

`track_drive`는 원조 [TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet)을 da
(drivable area, 주행가능영역) / ll(lane line, 차선) 듀얼헤드 세그멘테이션 모델로 쓰고
있는데, 실차 영상에서 두 가지 문제가 확인됨:
1. 노란색 차선을 거의 못 잡음
2. 역광/글레어 구간에서 완전 미검출

같은 트랙·같은 차량으로만 주행하는 좁은 도메인이라, 범용 모델보다 우리 트랙 데이터로
직접 파인튜닝하는 게 효율적이라 판단해서 시작한 프로젝트. 베이스 모델은 원조
TwinLiteNet의 후속작인 [TwinLiteNetPlus](https://github.com/chequanghuy/TwinLiteNetPlus)
(CAAM 어텐션 모듈 추가, nano/small/medium/large 4단계 크기 지원)를 씀.

## 우리가 한 것 (학습 방법론 요약)

전체 히스토리는 세션별로 [`PROGRESS.md`](./PROGRESS.md)에 시간순으로 상세 기록돼
있음 — 여기서는 핵심 흐름만 요약.

### 1. 데이터 수집 & 다이어트
- 실차에서 원본 프레임 2123장 수집
- perceptual hash(dHash)로 근접 중복(정지 반복 촬영, 같은 지점 재방문) 제거
- 라바콘 검출기로 콘 프레임 과대표집 구간을 stride 다운샘플
- 처음엔 "사람이 라벨링 가능한 현실적 상한"으로 470장까지 축소했다가, 라벨링이
  완전 자동화되면서 그 제약이 사라져 **dHash dedup 상한(1068장)까지 확장**

### 2. 라벨링 파이프라인
- 팀원들이 [CVAT](https://www.cvat.ai/)에서 협업 라벨링 (COCO RLE mask 포맷 채택 —
  polygon보다 CVAT 네이티브 Mask 편집 도구와 궁합이 좋음)
- da(주행가능영역): 사람이 직접 폴리곤/마스크로 라벨링
- ll(차선): 사람이 브러시로 그리면 두께가 프레임마다 들쭉날쭉(4~48px)해서 노이즈가
  됨 → **skeleton화 + polyline 단순화 + 고정 두께(8px) 재래스터화**로 정제

### 3. 완전 자동 의사 라벨링(pseudo-labeling)
사람 라벨링이 병목이 되지 않도록, da/ll 모두 자동 생성 파이프라인 확립:
- **da**: 우리가 파인튜닝한 모델의 예측
- **ll**: 여러 SOTA 차선 검출 모델(Grounding DINO+SAM2, UFLD-v2, CLRNet, GANet 등)을
  실제로 검증해본 결과, **YOLOPv2**(BDD100K 사전학습, 파인튜닝 0회)가 우리 도메인에서
  압도적으로 좋았음 — CULane/TuSimple 벤치마크 성능이 실제 도메인 전이 성능을 전혀
  예측하지 못한다는 걸 확인(UFLD-v2는 벤치마크 1위권인데 우리 트랙에서 거의 전멸).
  YOLOPv2 출력을 skeleton 정제해서 사용.
- SAM(Segment Anything)은 da/ll 둘 다 시도했지만 최종 폐기 — "시각적으로 이어진
  영역을 통째로 채우는" 방식이라, da/ll 경계가 시각적 경계가 아니라 차선이 정의하는
  의미적 경계인 우리 트랙에서는 구조적으로 안 맞음(완벽한 GT를 그대로 넣어도 SAM이
  IoU를 오히려 깎아먹는 것까지 확인).

### 4. 발견하고 고친 버그들
- **[핵심] letterbox vs plain resize 좌표 불일치**: TwinLiteNetPlus 원본 학습 코드가
  이미지는 letterbox(여백 패딩)로, 라벨은 plain resize로 서로 다르게 처리해서
  이미지-라벨 좌표가 어긋나 있었음(BDD100K 16:9 비율에선 우연히 안 드러났지만 우리
  이미지 4:3 비율에선 좌우 64px씩 어긋남). 실차 코드(`dl_lane.py`)가 plain resize를
  쓰는 것과도 안 맞는 쪽이라 학습 파이프라인을 실차 방식에 맞춰 수정.
- **pseudo label 오염 대물림**: da pseudo label을 만드는 데 쓴 모델이 위 letterbox
  버그가 있던 상태로 학습된 거라, 그 모델의 편향(커브 구간 도로 폭 과대평가)이
  pseudo label에 박히고, 그걸로 학습한 다음 모델이 편향을 그대로 물려받는 문제를
  발견. **2단계 부트스트랩**으로 해결: 사람이 직접 라벨링한 데이터만으로 먼저 작은
  모델을 학습(편향 없는 "깨끗한" 소스) → 그 모델로 나머지 프레임의 da pseudo label을
  생성.
- **체크포인트 재개 버그**: 학습 중 GPU 할당량이 끊겨 `--resume`으로 재개할 때마다
  "지금까지 최고 기록" 추적 변수가 checkpoint에 저장이 안 돼 있어서 리셋되던 버그 —
  재개 직후 첫 epoch이 실제 성능과 무관하게 무조건 "새 best"로 찍혀서 더 좋은
  체크포인트를 덮어쓸 위험이 있었음. checkpoint에 저장/복원하도록 수정 + da 기준
  best와 별도로 ll 기준 best도 같이 저장하도록 개선.

### 5. 검증 방법론
- 학습 로그 숫자만 보지 않고, 항상 실제 이미지에 다시 돌려서 old(원조 모델) vs
  new(파인튜닝) vs 다른 후보 모델을 나란히 비교하는 몽타주로 검증
- 사람이 라벨링한 프레임(GT)에는 IoU/과다포함(FP)/과소포함(FN) 비율을 직접 계산
- "커버리지(coverage)가 높아 보인다" 같은 단순 통계만으로 품질을 판단하지 않음 —
  실제로 커버리지는 높은데 GT 대비 30~50% 과다예측인 모델도 있었음

### 6. 학습 환경
- 1차: Google Colab(무료 GPU) — 할당량 소진 문제로 계정을 옮겨가며 진행, 매 에폭
  Drive 자동 백업 + 자동 재개 로직 추가
- 2차: 로컬 Windows PC(AMD RX 9070 XT, ROCm) — Colab 무료 GPU로 medium config
  학습이 너무 오래 걸려서 로컬로 이전

## 레포 구성

```
finetune_twinlitenetplus.ipynb                 # Colab 학습 노트북 (Google Drive 연동)
finetune_twinlitenetplus_local_windows_rocm.ipynb  # 로컬(Windows+ROCm) 학습 노트북
PROGRESS.md                                     # 세션별 상세 진행 기록 (읽기 순서: 목표→시간순 기록→다음 할 일)
scripts/                                        # 데이터 파이프라인/검증 스크립트
  build_pseudo_label_dataset*.py                #   da/ll pseudo label 자동 생성
  build_bootstrap_v2.py, build_diet_v2.py       #   부트스트랩/데이터셋 확장
  merge_human_labels*.py                        #   사람 라벨 병합
  check_da_overpaint.py, compare_*.py           #   모델 비교/검증 몽타주
  skeleton_polyline_utils.py                    #   ll 두께 정제 유틸
*.py (최상위)                                    # 초기 세션 데이터 triage/export 스크립트
```

데이터셋 원본/중간산출물/모델 가중치는 용량 문제로 이 레포에 포함하지 않음
(`.gitignore` 참고) — 별도 zip/Drive로 관리.

## 관련 레포

- 베이스 모델: [chequanghuy/TwinLiteNetPlus](https://github.com/chequanghuy/TwinLiteNetPlus)
- 원조 모델: [chequanghuy/TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet)
- ll pseudo label 소스: [CAIC-AD/YOLOPv2](https://github.com/CAIC-AD/YOLOPv2)
- 실차 배포 대상: [mastic-choi/UMK](https://github.com/mastic-choi/UMK) (`track_drive` 패키지)
