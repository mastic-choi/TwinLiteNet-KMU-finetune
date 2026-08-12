<div align="center">
<h1>TwinLiteNet+ KMU Track Fine-tune 🏎️</h1>
<p><a href="https://auto-contest.kookmin.ac.kr/">국민대학교 제9회 자율주행 경진대회</a> 참여작 —
<a href="https://github.com/mastic-choi/UMK">UMK/track_drive</a>의 차선·주행가능영역
인식 모델을, 미래관(신관) 4층 자율주행스튜디오 트랙에 맞춰 파인튜닝한다.</p>
</div>

## Background

`track_drive`는 원조 [TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet)을 da(주행가능영역)
/ ll(차선) 듀얼헤드 세그멘테이션 모델로 쓰는데, 실차 영상에서 두 가지 문제가 확인됐다:

1. 노란색 차선을 거의 못 잡음
2. 역광/글레어 구간, 그리고 **커브·좌회전 구간에서 주행가능영역을 트랙 경계 밖까지 잘못 칠함**

같은 트랙·같은 차량으로만 주행하는 좁은 도메인이라 범용 모델보다 우리 데이터로 직접
파인튜닝하는 게 효율적이라 판단해서 시작한 프로젝트. 베이스 모델은 원조 TwinLiteNet의
후속작인 [TwinLiteNetPlus](https://github.com/chequanghuy/TwinLiteNetPlus)(CAAM 어텐션
모듈 추가, nano/small/medium/large 4단계 크기 지원)를 썼다.

### 차량 사양

자이카 Y모델 기본 사양: AMD Ryzen(64bit, 8-core/16thread) + AMD Radeon RX Vega 8,
Traxxas 1:10 스케일 섀시, 170° 어안렌즈 640×480 카메라. 대회 후반부에
**reComputer Super J4012**(NVIDIA Jetson Orin NX 16GB, JetPack 6 / Ubuntu 22.04)가
추가로 제공되어 **본선 경기는 이 PC로 진행**함.

## Results

### 실제 주행 비교 (원조 TwinLiteNet vs 우리 파인튜닝 모델)

`dataset/`의 연속 프레임을 그대로 이어 붙여 실제 주행처럼 재생되는 GIF. 왼쪽이 원조
TwinLiteNet, 오른쪽이 우리가 40epoch 파인튜닝한 모델(medium config). 파란색=주행가능영역,
빨간색=차선, 노란 박스=실차 제어에 실제로 쓰는 ROI.

**커브 구간(S자, frame_000800~000935)** — 원조 모델은 커브 내내 주행가능영역이 트랙
경계 밖 회색 바닥까지 계속 삐져나감.

![커브 구간 비교](outputs/montages/curve_old_vs_new.gif)

**커브 구간 2(frame_000400~000480)** — 첫 번째와 다른 지점의 커브 구간. 회전
바깥쪽으로 과다포함되는 문제가 가장 뚜렷하게 보임.

![커브 구간 2 비교](outputs/montages/curve2_old_vs_new.gif)

**직진 구간(frame_000933~000985, 대조군)** — 직진에서는 원조 모델도 원래 크게 나쁘지
않다는 걸 보여주는 대조군. 문제는 특히 커브에서 두드러짐.

![직진 구간 비교](outputs/montages/straight_old_vs_new.gif)

### 라벨링 파이프라인 → 최종 모델

왼쪽부터: **① 원조 TwinLiteNet 추론 → ② 우리가 만든 da 라벨(사람 라벨링 + 파인튜닝
모델 pseudo-label) → ③ YOLOPv2 기반 ll 라벨(skeleton 정제) → ④ 이 라벨들로 학습한
최종 파인튜닝 모델의 추론 결과**. 즉 ②③이 학습 데이터, ④가 그 데이터로 학습한 결과물.

![라벨링 파이프라인](outputs/montages/pipeline_montage.png)

### 파인튜닝 전/후 정량 비교 (validation set 160장, GT 대비 동일 기준 측정)

원조 TwinLiteNet은 우리 트랙 GT로 학습/검증된 적이 없어서 여기엔 없음(대신 위 주행
비교 GIF가 old vs new 비교 자료) — 이 표는 **같은 베이스 모델(TwinLiteNetPlus
medium)의 파인튜닝 전/후**를 우리 GT 기준 IoU로 직접 비교한 것.

| Model | Drivable Area mIoU | Lane Line Acc | Lane Line IoU |
|:-----:|:-------------------:|:-------------:|:--------------:|
| TwinLiteNetPlus (pretrained, BDD100K, 파인튜닝 0회) | 0.815 | 0.669 | 0.132 |
| **TwinLiteNetPlus (우리 트랙 파인튜닝, medium, 40epoch)** | **0.935 (+0.120)** :arrow_up: | **0.802 (+0.133)** :arrow_up: | **0.552 (+0.420, 4.2×)** :arrow_up: |

특히 ll IoU가 4배 이상 뛴 게 이 프로젝트의 원래 목표(원조 모델이 차선을 거의 못 잡던
문제)와 정확히 일치하는 지표.

### 모델 크기 (config별 파라미터 수)

TwinLiteNetPlus는 nano/small/medium/large 4단계를 지원 — 우리는 정확도와 속도를
같이 고려해 **medium**을 최종 채택.

| Config | Params |
|:------:|:------:|
| nano | 33,379 |
| small | 121,552 |
| **medium (채택)** | **478,876** |
| large | 1,943,911 |

### 원조 TwinLiteNet vs TwinLiteNetPlus — 크기/속도

| Model | Params | Input | Speed (RX 9070 XT, ROCm, batch=1) |
|:-----:|:------:|:-----:|:----------------------------------:|
| TwinLiteNet (원조) | 439,633 | 640×360 | 46.3 fps |
| **TwinLiteNetPlus (medium, 우리 채택)** | 478,876 | 640×384 | **91.5 fps** |

파라미터 수는 비슷한데(약 9% 많음) 추론 속도는 약 2배 — CAAM 어텐션 구조가 원조보다
효율적인 것으로 보임(엄밀한 알고리즘 단위 프로파일링은 안 해봄, 같은 GPU/배치에서의
end-to-end 측정치).

## 학습된 모델

- **`outputs/models/best.onnx`** (+ `best.onnx.data`) — medium config, 40epoch, ll
  IOU 기준 best 체크포인트. `track_drive`의 `TwinLiteNetEngine`과 바로 호환
  (입력 `images`, 출력 `da`/`ll`, 입력 크기 640×384).
- 실차 반영 시 `perception/dl_lane.py`의 `DL_INPUT_H`를 360 → **384**로 바꿔야 함
  (letterbox 버그 수정 후 학습 해상도가 변경됨 — 자세한 배경은 PROGRESS.md 참고).
- 실차에 바로 쓰기 전에 반드시 실제 주행 테스트로 검증할 것 — 지금까지는 정적
  이미지/ROI 커버리지 기준 검증만 마친 상태.

## 우리가 한 것 (요약)

세션별 시행착오와 트러블슈팅 히스토리는 전부 [`PROGRESS.md`](./PROGRESS.md)에
시간순으로 기록돼 있음 — 여기서는 최종적으로 자리잡은 방법론만 요약.

- **데이터**: 실차 원본 2123장 → dHash 근접중복 제거로 1068장 확보
- **라벨링**: da는 사람이 CVAT에서 직접 라벨링(일부) + 우리 모델의 pseudo-label(대부분),
  ll은 YOLOPv2(BDD100K 사전학습, 파인튜닝 0회) 추론을 skeleton화+polyline 단순화로
  정제해서 사용 — 여러 SOTA 차선 모델을 직접 비교한 결과 YOLOPv2가 우리 도메인에서
  압도적으로 좋았음(공개 벤치마크 순위와 무관)
- **학습**: TwinLiteNetPlus medium config, ll 손실 가중치 2배 + ll 디코더 LR 2.5배
  (ll이 da보다 어려운 태스크라 학습 압력을 더 줌)
- **검증**: 학습 로그 숫자만 보지 않고 항상 실제 이미지에 다시 돌려서 old/new 모델을
  나란히 비교 — "커버리지가 높아 보인다" 같은 인상만으로 판단하지 않고 사람 GT 대비
  IoU/과다포함/과소포함을 직접 계산
- **학습 환경**: Google Colab(무료 GPU) → 로컬 Windows(AMD RX 9070 XT, ROCm)로 이전
  (medium config가 Colab에서 너무 오래 걸려서). 원래는 국민대 소프트웨어융합대학
  **FOSCAR** 동아리의 RTX 컴퓨터를 지원받아 쓸 계획이었으나 무산되어, 팀원
  최준수의 개인 PC(RX 9070 XT)로 학습을 진행함

## 레포 구성

```
finetune_twinlitenetplus.ipynb                     # Colab 학습 노트북 (Google Drive 연동)
finetune_twinlitenetplus_local_windows_rocm.ipynb   # 로컬(Windows+ROCm) 학습 노트북
kuac_lane_prelabel_shared.ipynb                     # SAM2+YOLOPv2 기반 pre-labeling Colab 노트북
PROGRESS.md                                          # 세션별 상세 진행 기록 + 시행착오 히스토리
outputs/
  models/best.onnx(.data)                            #   최종 학습 결과물
  montages/                                           #   비교 GIF/몽타주 전체
scripts/                                             # 기능별로 분류된 파이프라인 스크립트
  data_prep/                                          #   원본 프레임 수집·다이어트·triage
  cvat/                                                #   CVAT import/export, 사람 라벨 병합
  pseudo_label/                                        #   da/ll 자동 라벨링 파이프라인 + 공용 유틸
  eval/                                                #   모델 비교·검증·곡률/각도 검출
  montage/                                             #   비교 몽타주/GIF 생성
  export/                                              #   ONNX export
```

데이터셋 원본/중간산출물은 용량 문제로 이 레포에 포함하지 않음(`.gitignore` 참고) —
별도 zip/Drive로 관리.

## 관련 레포

- 베이스 모델: [chequanghuy/TwinLiteNetPlus](https://github.com/chequanghuy/TwinLiteNetPlus)
- 원조 모델: [chequanghuy/TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet)
- ll pseudo label 소스: [CAIC-AD/YOLOPv2](https://github.com/CAIC-AD/YOLOPv2)
- 실차 배포 대상: [mastic-choi/UMK](https://github.com/mastic-choi/UMK) (`track_drive` 패키지)

## Special Thanks

이 프로젝트의 딥앙상블(deep ensemble) 라벨링 아이디어를 제안해주신 국민대학교
**이현기** 교수님께 감사드립니다. ([hyungi-lee.github.io](https://hyungi-lee.github.io/))
