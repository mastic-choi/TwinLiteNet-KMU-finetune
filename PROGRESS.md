# TwinLiteNet+ 트랙 파인튜닝 — 진행 상황 (2026-08-11 갱신, 6회차)

이 문서는 세션이 끊겨도 다음 LLM/사람이 바로 이어서 작업할 수 있게 만든 상태 기록이다.
읽는 순서: 1) 목표 → 2) 지금까지 한 일(시간순) → 3) 알아낸 버그/교훈 → 4) 다음 할 일 → 5) 파일/경로 참조.

## 1. 목표

UMK `track_drive` 패키지가 쓰는 원조 TwinLiteNet(da/ll 듀얼헤드)이 (1) 노란선을 거의 못 잡고
(2) 역광/글레어에서 완전 미검출되는 문제가 실차 영상에서 확인됨(`track_drive/track_drive/README.md` §2.18).
같은 트랙·같은 차량으로만 주행하므로, TwinLiteNet+를 우리 트랙 데이터로 파인튜닝해서 `.onnx`로
내보내 교체하는 게 목표. 작업 노트북: `/Users/mastic-choi/code/fine-tune/finetune_twinlitenetplus.ipynb`
(Google Drive `MyDrive/umk_twinlite/finetune_twinlitenetplus.ipynb`에 올려서 Colab으로 실행 중).

## 2. 지금까지 한 일 (시간순)

### 2.1 원본 프레임 수집 및 다이어트 (로컬 Mac)
- 실차에서 뽑은 원본 프레임 2123장이 `/Users/mastic-choi/code/fine-tune/dataset/`에 있음.
- 기존 배포 모델 `~/code/UMK/track_drive/track_drive/models/best.onnx`(원조 TwinLiteNet, 640×360
  입력)로 전체 프레임에 da/ll 추론 돌려서 `triage_result.csv` 생성 (`triage_dataset.py`) —
  `config.py`의 실제 임계값(`DL_FG_THRESHOLD=0.5`, `DL_LL_FG_THRESHOLD=0.7`)과 ROI
  (`DL_ROI_Y0=250, DL_ROI_Y1=390`)를 그대로 반영해서 계산.
- 오버레이 이미지 전체 생성: `dataset_overlay/`(da=파랑 블렌드, ll=빨강, 노란 박스=ROI) —
  `make_overlay_dataset.py`.
- ROI를 좌/중앙/우 3구간으로 나눠 "차선 3개 중 몇 개 잡히는지" 검사 → `missing_line_result.csv`
  (`find_missing_line_frames.py`). `n_detected<3`이면 실패 후보.
- 라바콘 검출기 `~/code/xycar_ws/src/yolo_ros/cone_best.pt`(YOLOv8, 클래스 1개 `cone`)로 전체
  프레임 콘 검출 → `cone_detect_result.csv`(`detect_cones.py`). 라바콘 프레임이 30개 통과
  구간·542장에 과대표집된 것 확인.
- 콘 프레임 구간별 stride=5 + 구간 최고신뢰도 프레임 보존으로 143장까지 축소
  (`build_cone_reduced_dataset.py`) → `dataset_balanced/`(1724장, 나중에 로컬에서 삭제됨,
  재구성 로직은 `diet_dataset.py`의 `rebuild_balanced_file_list()`에 들어있음).
- perceptual hash(dHash)로 근접 중복(정지 반복 촬영/같은 지점 재방문) 제거 + "3선 다 잡힌 쉬운
  성공" 프레임 stride 다운샘플 + 콘 비중 재초과 시 콘도 추가 축소 (`diet_dataset.py`) →
  최종 **470장**, `dataset_diet/`. 최종 구성: 실패후보 133 / 콘 71 / 일반성공 266
  (`missing_line_result.csv` 기준 분류).

**⚠️ 알아낸 버그**: 처음에 `triage_result.csv` 기준으로 뽑은 "진짜 실패 23장" 중 **20장이
최종 470장에서 빠졌음** — `diet_dataset.py`의 "실패 후보 무조건 보존" 로직이
`missing_line_result.csv` 기준만 참조하고 `triage_result.csv`의 `failure_candidate`
컬럼은 아예 안 봤음. 그 20장 다수가 콘도 같이 찍힌 프레임이라 콘 stride 축소 단계에서
보호 못 받고 같이 빠짐. → 나머지 396장 작업 시 이 갭을 인지하고 있을 것.

### 2.2 초안 마스크 생성 (재추론 없이)
- `dataset_overlay/`의 색상(정확히 아는 블렌드 공식)에서 역산해서 da/ll 이진 마스크 복원
  (`extract_draft_masks.py`) → `raw/images_todo/`, `raw/da_masks_draft/`,
  `raw/ll_masks_draft/` (각 470장, PNG).
- Drive 업로드는 `umk_twinlite_raw.zip`(3개 폴더 압축, 209MB) 하나로 — 브라우저로 개별
  파일/폴더를 대량 드래그하면 누락되거나 **중복 폴더가 생기는 사고**가 실제로 발생했음
  (아래 5.3 참고). zip 하나 올리고 Colab에서 `!unzip`하는 방식이 훨씬 안전.

### 2.3 CVAT 협업 라벨링 세팅
- `app.cvat.ai` 무료 계정, Organization/Project(`UMK`) 생성, 라벨 3개
  (`drivable_area`, `lane_line`, `background` — **background도 실제 라벨로 등록 필요**,
  안 하면 import 시 `Label 'background' is not registered` 에러남).
- Task `KMU_lane`에 470장 업로드. **Segment size를 실수로 3으로 설정**해서 job이 157개로
  잘게 쪼개짐(원래 인원수 기준으로 크게 잡았어야 했는데 그대로 둠 — 이미 진행 중이라 안 고침).
- 초안 마스크를 pre-annotation으로 넣는 방법 3번 시도:
  1. **Segmentation mask 1.1**(raster PNG) import → 픽셀 계단을 그대로 폴리곤 점으로 따라가서
     점이 너무 많음(비효율).
  2. **COCO 1.0 polygon**(`cv2.findContours` + `approxPolyDP`로 미리 단순화) → 점 92% 감소
     성공했지만, 라벨별로 폴리곤이 여러 개(배경까지 포함) 생겨서 여전히 복잡.
  3. **COCO 1.0 RLE(iscrowd=1)** → CVAT이 폴리곤이 아니라 네이티브 **Mask(브러시 편집)**
     객체로 가져옴, 이미지당 정확히 2개 객체(da 1개 + ll 1개)만 생김. **최종 채택.**
     `pycocotools.mask.encode()`로 RLE 인코딩, 픽셀 단위 왕복 검증 완료.
  - 관련 스크립트: `prepare_cvat_import.py`(1번), `prepare_coco_import.py`(2번),
    `prepare_coco_rle_import.py`(3번, 최종).
- 되돌리는(export) 스크립트: `export_cvat_masks.py`(구버전, Segmentation mask 포맷용,
  안 씀) / **`export_cvat_bootstrap.py`**(현재 COCO RLE export용, 실제 사용 중).

### 2.4 팀 라벨링 진행 + 부트스트랩(active learning) 미니 파인튜닝
- 팀원들이 CVAT에서 브러시로 초안 보정 중. **CVAT 무료 플랜은 AI agent calls(SAM 등)에
  월별 한도가 있고 이미 소진됨** — 지금은 SAM 없이 수동 Polygon/Brush로 작업해야 함(한도랑
  무관한 순수 수동 도구).
- Task를 Task 단위로 "Export dataset"(COCO 1.0) + Jobs 목록을 CSV로 export
  (`cvat-jobs-*.csv`, 컬럼: State/Start Frame/Stop Frame 등)해서, **State가 "completed"인
  job의 프레임만** 골라 `bootstrap/images`, `bootstrap/da_masks`, `bootstrap/ll_masks`
  (74장)로 변환 (`export_cvat_bootstrap.py`). "completed" 아닌 프레임은 아직 사람 검증
  안 된 원래 초안일 수 있어서 반드시 제외(안 그러면 모델이 원래 실패 패턴을 "정답"으로
  학습하게 됨).
- 노트북 cell-12/14/15가 `raw/*.jpg`를 하드코딩 참조하던 버그 발견 → `bootstrap/*.png`로
  전부 수정(로컬 파일 + Colab 양쪽 다 반영함). `BDD100K.py`(레포 자체 코드)는 이미지가
  `.png`여도 문제없음(경로 문자열 치환이 우연히 no-op이 되면서 정확한 경로가 나옴, 확인함).
- 74장(train 63/val 11)으로 미니 파인튜닝 실행(`CONFIG='small'`, `MAX_EPOCHS=40`,
  `pretrained/small.pth`에서 시작). 결과: **da mIoU 0.93대에서 수렴**(~epoch 24부터
  평평), **ll IOU는 0.28~0.31대에서 정체**(데이터가 적고 차선이 얇은 클래스라 한계).
  `finetune_out/best.pth`(581KB, `small` config가 원래 12만 파라미터짜리라 정상 크기).
- ONNX export 시 최신 torch(2.11)가 `onnxscript` 필요해서 `pip install onnxscript` 추가
  설치 필요했음. export 결과가 **가중치를 `.onnx.data`로 분리 저장**함(torch 최신 버전
  exporter 특성) — onnx 로드하려면 `.onnx`와 `.onnx.data` 둘 다 같은 폴더에 있어야 함.
  현재 `/Users/mastic-choi/Downloads/twinlitenetplus_small_finetuned.onnx`(+`.onnx.data`).

### 2.5 부트스트랩 모델 성능 검증
- 원래 실패했던 프레임 + 정상 프레임 섞어서 old(`best.onnx`) vs new(부트스트랩) 비교 몽타주
  제작 (`make_bootstrap_montage.py`, `make_before_after_montage.py`).
- **정량 비교**(ROI 안에서만): da 평균 +0.125, ll 평균 +0.029 — 대체로 개선. 단, 프레임별
  편차 큼 — 원래 완전 실패하던 프레임(예: `frame_001082`)은 da/ll 둘 다 0에서 실질적인
  값으로 극적으로 좋아짐. 반면 원래 이미 잘 되던 일부 프레임은 da가 오히려 떨어지기도 함
  (`001915`: 0.834→0.401, `001902`: 0.482→0.351).
- 시각적으로는 "old가 ll을 더 잘 잡는 것처럼" 보이는데, 이는 old의 빨간 선이 **ROI 밖
  (화면 위쪽)까지 길게 뻗어있어서** 생기는 착시 — 실제 기능적 ROI 안에서는 ll도 대부분
  new가 더 높음. da/ll 둘 다 "한쪽이 무조건 낫다"라고 말할 수 없고 **프레임마다 다름**.

### 2.6 CVAT AI agent(SAM) 한도 소진 → 로컬 SAM으로 전환
- 팀원이 CVAT에서 SAM(AI Tools) 쓰다가 "Waiting for a response..."에서 안 넘어감 →
  확인해보니 **CVAT 무료 플랜은 AI agent calls(SAM 등)에 월별 한도가 있고 이미 소진됨**
  (공식 Fair Usage Policy 확인, 정확한 숫자는 비공개, 무료 우회 방법 없음 — 다음 달 리셋 또는
  유료 전환만 가능). 이번 달은 CVAT 안에서 SAM 못 씀.
- 대안으로 **오픈소스 SAM을 로컬(M4 Mac, 16GB, MPS 가속 가능)에서 직접 돌리기**로 결정.
  `ultralytics` 패키지가 SAM2를 바로 지원해서 별도로 Meta 원본 레포 설치 안 해도 됨 —
  `.venv_yolo`에 `from ultralytics import SAM; SAM('sam2.1_b.pt')`로 자동 다운로드/로드
  확인 완료(장당 추론 ~1초, `points=[[x,y]], labels=[1]`로 포인트 프롬프트).
  같은 venv에 `onnxruntime`도 추가 설치함(원래 없었음).
- **"SAM으로 da/ll 다 자동 클릭" 대신, 앙상블(old+new 모델)과 SAM을 역할 분담**하는 방식으로
  설계함 — SAM은 큼직한 덩어리(도로)는 잘 다듬지만 얇고 구불구불한 선(차선)은 점 프롬프트 방식과
  안 맞아서 오히려 나쁠 수 있음:
  - **da**: old/new 앙상블로 고른 마스크의 최대 연결 덩어리 중심점을 SAM에 프롬프트로 줘서
    경계 정밀화. SAM 결과 면적이 앙상블 대비 0.2~5배 범위 밖이면(터무니없이 다르면) 못 믿는
    것으로 보고 앙상블 결과로 폴백.
  - **ll**: SAM 적용 안 함, 앙상블(old/new 중 ROI 커버리지 높은 쪽) 결과 그대로 사용.
  - 구현: `make_ensemble_sam_drafts.py` — 인자로 프레임 파일명 리스트 받음(없으면
    `raw/images_todo/` 전체). 출력: `ensemble_drafts/images|da_masks|ll_masks/` +
    `source_log.csv`(프레임별 da/ll이 old/new 중 뭘 썼는지, SAM 성공 여부 기록).
- **검증**: 아는 22장(실패후보 16 + 정상 8)으로 먼저 돌려서 old vs ENSEMBLE+SAM 비교 몽타주
  제작(`ensemble_sam_montage.png`) — 결과 좋음:
  - 원래 완전 실패(예: `001082`)하던 프레임 → 이제 도로/차선 다 잡힘
  - 원래 조각조각 끊겨있던 프레임(`000928`, `000924`) → SAM이 매끄럽게 이어붙임
  - 원래 이미 잘 되던 프레임 → 순수 앙상블만 썼을 때는 일부(`001915`, `001902`) da가
    떨어지는 부작용이 있었는데, **SAM 정밀화를 더하니 그 부작용이 사라짐**(비슷하거나 더
    깨끗해짐). 22장 전부 SAM 정밀화 성공(sanity check 통과).
- 사용자가 새 CVAT export(`task_2502410_annotations_2026_08_10_15_02_12_coco 1.0.zip`,
  객체 1160개, 이전 부트스트랩 시점 1133개보다 늘어남 — 그 사이 팀원들이 더 보정한 것으로
  추정)를 줌. 근데 **몇 개 job이 새로 completed 됐는지 알려주는 Jobs CSV는 아직 안 줌** —
  이게 있어야 "이미 사람이 검증한 프레임"과 "아직 초안뿐인 프레임"을 구분해서, 앙상블+SAM
  결과를 후자에만 적용하고 전자는 안 건드릴 수 있음(안 그러면 팀원이 방금 고친 걸 초안으로
  덮어씌울 위험). **다음 세션에서 이 CSV부터 받아서 진행할 것.**

### 2.7 ll 라벨 두께 노이즈 발견 → polyline 전환 결정

- 최신 CVAT COCO export(`task_2502410_..._coco 1.0.zip`, 470장/1160 오브젝트)를 직접 열어
  분석: ll(lane_line) 마스크의 두께가 오브젝트/프레임마다 **4~48px로 들쭉날쭉**(평균
  11.6px, 중앙값 8.4px, std 7.6px). 카메라 원근상 완만히 변해야 할 두께가 이렇게 널뛰는
  건 팀원마다·프레임마다 브러시 크기를 다르게 써서 생긴 라벨 노이즈로 추정 — da(큰 덩어리)는
  두께 개념이 없어 브러시 편차 영향이 적지만(mIoU 0.93), ll은 얇은 선이라 편차가 그대로
  라벨 노이즈로 들어감. **ll IoU가 0.28~0.31에서 정체되는 원인일 가능성.**
- 공개 차선 데이터셋(TuSimple: cubic spline, CULane: point annotation, BDD100K lane
  marking: 2D polyline+Bezier)을 조사한 결과, **업계 표준은 차선을 픽셀 브러시가 아니라
  polyline(점)으로 라벨링**하는 것으로 확인. da(BDD100K drivable area)는 반대로 지금처럼
  mask/polygon이 표준 — 지금 da 파이프라인은 그대로 유지.
- **조사한 자동 라벨링 대안 방법론 비교**(SAM 한도 소진 이후 대안 검토 겸):
  1. **파운데이션 모델**(Grounding DINO + SAM 2): 텍스트 프롬프트("lane mark")로 세그멘테이션,
     제로샷 성능 좋음 — 단 SAM은 얇고 구불구불한 구조에 약하다는 게 이미 §2.6에서 확인됨.
  2. **공개 SOTA 차선 모델**(Ultra-Fast-Lane-Detection-v2, CLRNet, YOLOPv2): CULane/TuSimple로
     사전학습, polyline 좌표 직접 추출 가능.
  3. **인터랙티브 오토 라벨링 툴**(AnyLabeling, CVAT+SAM backend): 클릭 몇 번으로 자동 영역
     지정 + human-in-the-loop 검수.
  - 추천되는 일반적 3-step 파이프라인: ① 의사 라벨(pseudo-label) 생성(SOTA 모델 추론) →
    ② 후처리(다항식/스플라인 피팅으로 끊긴 선 연결+노이즈 제거, confidence>0.8만 채택) →
    ③ human-in-the-loop 검수(AnyLabeling/CVAT로 불러와 오탐만 수정).
  - 즉시 써볼 수 있는 오픈소스 툴: **X-AnyLabeling**(SAM2+auto-detection 내장 데스크톱 앱),
    **CVAT**(Docker + Grounding DINO/SAM 연결해서 데이터셋 일괄 오토 어노테이션).
  - **우리 프로젝트 결정**: 2번(CLRNet/UFLD-v2 등 공개 SOTA 모델)은 채택 안 함 —
    TuSimple/CULane은 우리 트랙(라바콘 있는 실내 폐쇄 코스, 다른 카메라 각도)과 도메인이
    많이 달라서, 애초에 이 프로젝트를 시작한 이유(원조 TwinLiteNet의 도메인 갭)가 그대로
    재발할 위험. **3-step 구조 자체는 채택**하되, 의사 라벨 소스를 외부 SOTA 모델이 아니라
    **우리가 이미 도메인 적응시킨 부트스트랩 TwinLiteNet+ 모델의 ll 예측**으로 대체하고,
    후처리도 "스플라인 피팅" 대신 **skeleton화 + polyline 단순화 + 고정 두께 재래스터화**로
    구현(아래 2.8). 목적(끊긴 선 연결, 두께/노이즈 정리)은 같지만 외부 의존성/도메인 갭 없이
    우리 데이터만으로 처리.

### 2.8 skeleton 기반 ll 라벨 정제 실험 (완료된 74장 대상 검증)

- `skeleton_polyline_utils.py` 작성: 이진 마스크 → 컴포넌트별 skeleton 추출
  (`skimage.morphology.skeletonize`) → 가장 긴 경로(BFS 두 번, tree-diameter 방식,
  짧은 곁가지는 자동 무시) → `cv2.approxPolyDP`로 polyline 단순화 → `cv2.polylines`로
  고정 두께 재래스터화. 노이즈(면적 1% 미만 또는 15px 미만) 컴포넌트는 자동 제외.
- `refine_ll_masks.py`: 위 유틸로 `bootstrap/`(팀원이 브러시로 완료한 74장)의 ll_masks를
  재생성, da_masks는 `da |= ll_refined`로 재합성(export_cvat_bootstrap.py 관례 유지).
  결과 → `bootstrap_refined/`. 두께 편차 std가 **8.4px → 2.4px로 대폭 감소** 확인
  (재래스터화 폭 8px 기준).
- `make_refine_montage.py`로 원본(파랑) vs 정제 결과(빨강, 겹침=보라) 74장 전체 비교
  몽타주(`outputs/montages/ll_refine_montage.png`) 제작, 육안 확인 결과 대부분 프레임에서
  원래 브러시 선을 거의 그대로 따라가면서 두께만 일관되게 정리됨. 발견된 한계 2가지:
  1. 아주 작고 멀리 떨어진 점선 조각이 노이즈 필터에 걸려 가끔 누락됨(빈도 낮음, 심각하지 않음).
  2. 두 차선이 갈라지는 분기점 근처에서 최장경로 하나만 택하다 보니 경로가 살짝 어긋나는
     경우가 드물게 있음.
  - **주의**: 이 정제는 사람이 이미 그린 위치 정보를 그대로 보존하고 두께만 정리하는
    후처리일 뿐, 라벨 위치 자체의 정확도를 개선하지는 않음(그건 원래 브러시 작업 품질에
    종속).
- **재부트스트랩 학습 준비 완료** — 로컬에서 학습하지 않고 기존과 동일하게 Colab 노트북으로
  진행(모델이 작아 로컬 M4 MPS로도 가능하지만, 노트북/Drive 워크플로우를 그대로 유지하기로
  함). `finetune_twinlitenetplus.ipynb`를 수정:
  - cell 12(데이터 정리): `bootstrap/` 대신 `{DRIVE_ROOT}/bootstrap_refined.zip`을 자동
    압축 해제해서 사용하도록 변경(기존 `bootstrap/`은 안 건드리고 비교용으로 유지).
  - cell 21(학습 실행): `--savedir`를 `./finetune_out` → `./finetune_out_refined`로 분리
    (기존 브러시 라벨 학습 결과를 덮어쓰지 않고 나중에 비교 가능하게).
  - cell 23/25(Drive 백업 + onnx export)도 `finetune_out_refined` 기준으로 같이 수정,
    onnx 파일명도 `twinlitenetplus_small_finetuned_refined.onnx`로 구분.
  - 로컬에 `bootstrap_refined.zip`(32MB) 생성 완료 → Drive 업로드 + Colab 재학습까지 완료.

### 2.9 refined 학습 결과 검증 + 대규모 모델 서베이 + 완전 자동 의사 라벨링 파이프라인 확정

- **refined 학습 결과, 로그 숫자만으로 오판할 뻔함**: `finetune_out_refined` 학습 로그의
  ll IOU가 0.19대로 기존(0.28~0.31)보다 낮게 나와서 "polyline 해도 안 좋아지는데?"라는
  의심이 들었음. 원인 확인: (1) 학습이 40에폭 중 16에폭까지밖에 안 끝난 상태였고 아직
  IOU가 계속 오르는 중이었음, (2) 더 중요하게 — **GT 자체가 달라져서(얇아짐) IoU 숫자를
  직접 비교하면 안 되는 상황**이었음(같은 품질이어도 얇은 GT 기준 IoU가 구조적으로 더
  낮게 나옴). `make_refined_compare_montage.py`로 old/finetuned(brush)/finetuned(refined)
  3-way를 원래 성공 20장+실패 10장에 실제로 돌려서 **ROI 커버리지로 재검증**한 결과
  refined가 brush와 동등하거나 근소 우위(성공 프레임 평균 da/ll 둘 다 개선, 실패 프레임은
  `frame_001082` 한 장의 da 이상치 때문에 평균이 끌림 — 그 프레임 제외하면 역시 근소 우위).
  **교훈**: GT가 바뀌는 실험은 학습 로그 숫자를 절대 직접 비교하면 안 되고, 항상 별도
  스크립트로 실제 이미지에 돌려서 확인할 것.

- **자동 라벨링 대안으로 조사한 모델들 — 결과 요약**(전부 GitHub 클론 후 직접 실행/검증):

  | 모델 | 결과 | 비고 |
  |---|---|---|
  | Grounding DINO + SAM2 | ❌ | ll에 엉뚱한 위치를 두꺼운 덩어리로 잘못 잡음(§ 이전 기록) |
  | **YOLOPv2**(BDD100K pretrained) | ✅ **채택** | 파인튜닝 0회인데도 bootstrap GT 대비 ll IoU **0.412**(우리 모델 0.055의 7배+). da는 우리 모델(0.852)보다 부정확해서 da엔 안 씀. 곡선에서는 격차가 줄어듦(직선 ~25배 -> 곡선 ~2~3배 차이) |
  | Ultra-Fast-Lane-Detection-v2 | ❌ | CULane 전용이라 우리 도메인에서 거의 전멸(20장 중 17장 완전 미검출) |
  | CLRNet | ❌ | CUDA 전용 NMS 커스텀 연산(setup.py에 하드코딩) — Mac에서 빌드 자체 불가 |
  | GANet | ❌ | CLRNet보다 더 심함 — deform_conv/nms/roi_align 등 CUDA 전용 연산 6개+ |
  | CondLaneNet | ❌ | GANet과 같은 계열(오래된 mmdetection 포크), 역시 CUDA 전용 |
  | CLRerNet(ONNX) | 보류 | PINTO_model_zoo에 ONNX 버전 있으나(1.9GB), NMS+lane 디코딩 로직이 빠진 raw 출력이라 직접 재구현 필요(`clrernet_decode.py` 작성함) + 다운로드 자체가 너무 느려서(시간당 ~1GB) 중단. 필요시 재개 가능 |
  | PETRv2 | ❌ | nuScenes 6-카메라 서라운드+캘리브레이션 전제 구조라 모노카메라 1대로는 아예 성립 불가 |
  | LaneGAP | ❌ | MapTRv2와 같은 계열, PETRv2와 동일한 이유로 불가 |
  | SegFormer(Cityscapes) | ❌ | da 커버리지 수치는 높아 보였지만(0.87) **실제 GT랑 IoU 재보니 0.747, GT보다 30~50% 과다예측** — "도로처럼 보이는 건 다 칠하는" 착시였음. 우리 모델(0.852)보다 부정확 |
  | Mask2Former(Cityscapes) | ❌ | SegFormer와 동일한 한계(class에 "차선" 자체가 없음, da도 우리 모델보다 부정확) |

- **SAM은 da/ll 어디에도 못 씀 — 프롬프트 방식을 다 바꿔봐도 마찬가지였음**. bootstrap
  GT로 직접 검증:
  - da: **완벽한 GT를 그대로 SAM에 넣어도** IoU 1.0 → 0.75로 떨어지고 GT 밖으로 평균
    20%+ 삐져나옴 (point prompt, `sam_refine_da()`)
  - ll: YOLOPv2 ll(원본 IoU 0.412)을 SAM point prompt에 넣으면 오히려 0.278로 하락,
    한 프레임은 SAM이 바닥 전체로 폭주(GT 대비 11.7배 과다)
  - da: "da는 ll 두 최외곽선 사이"라는 기하학적 관계를 GT로 확인(사실로 확인됨) →
    이 영역을 SAM에 박스/점 프롬프트로 줘봤지만(`make_da_from_ll_montage.py`) 벽/천장까지
    번지는 동일한 문제 반복(과다포함 25~51%). 순수 geometric(SAM 없음)은 과다포함이
    거의 없지만(3%) ll 검출 누락분만큼 과소포함(IoU 0.375)
  - **결론**: SAM은 "시각적으로 이어진 영역"을 통째로 채우는 방식이라, da/ll의 경계가
    "의미적 경계"(차선이 정의)이지 "시각적 경계"(색상/질감 차이)가 아닌 우리 트랙에서는
    프롬프트를 뭘 쓰든 구조적으로 안 맞음. **앞으로 이 프로젝트에서 SAM은 안 쓰는 걸로
    최종 결론.**

- **최종 확정한 완전 자동 의사 라벨링 파이프라인**: `build_pseudo_label_dataset.py` 작성.
  - da: 우리 파인튜닝 모델(`twinlitenetplus_small_finetuned_refined.onnx`) 예측 그대로
  - ll: YOLOPv2 예측 → skeleton화 → polyline 단순화 → 고정 두께(8px) 재래스터화
  - da에 ll을 합집합으로 포함(차선은 도로 위 — 기존 관례 유지)
  - **CVAT 라벨링 진행 상태와 무관하게 완전 자동**이라 Jobs CSV 안 받아도 바로 실행 가능
    (기존 `make_ensemble_sam_drafts.py`의 old/new 앙상블+SAM 방식은 이제 안 씀 — 이
    스크립트가 그 자리를 대체)
  - `raw/images_todo/` 전체 470장에 대해 실행 완료 → `pseudo_dataset/images|da_masks|ll_masks`.
    YOLOPv2가 차선을 하나도 못 찾은 프레임 0/470. 12장 스팟 체크 몽타주로 품질 확인함(대체로
    양호, 일부 프레임 da 가장자리가 약간 지저분함 — 우리 모델 자체의 기존 한계).
  - **주의**: 이건 사람 검증 없는 순수 자동 생성 라벨임 — CVAT에서 팀원이 실제로 완료한
    프레임(§2.4의 74장, 지금은 더 늘었을 것)의 사람 라벨과 섞어 쓸 땐 사람 라벨을
    우선해야 함(아직 이 병합 로직은 미구현, 다음 할 일 참고).

### 2.10 사람 라벨 병합 + 470장 학습 실행 + 학습 코드 추가 개선 + 신규 라벨링 요청 데이터셋

- **버그 수정**: `pseudo_dataset` 만들 때 CVAT에서 사람이 이미 완료한 74장(`bootstrap/`)까지
  자동 라벨로 덮어썼던 걸 발견 → `merge_human_labels.py` 작성해서 그 74장만 사람 라벨
  (da는 `bootstrap/da_masks` 그대로, ll은 `bootstrap_refined/ll_masks`로 두께 컨벤션
  통일)로 다시 교체. 470장 중 74장 사람 검증 + 396장 자동 생성 상태로 zip 재생성함.
- **470장 학습 시작**: `finetune_out_470`으로 Colab에서 학습 진행 중. 로그 사용 시 주의 —
  da mIoU는 13에폭 만에 0.93까지 수렴했는데 ll IOU는 0.28대에서 아직 오르는 중(40에폭
  다 끝나야 판단 가능).
- **노트북에 추가 반영한 개선 5가지**(다음 학습부터 적용, 지금 도는 학습엔 소급 안 됨):
  1. `CONFIG`: small → **medium**
  2. `loss.py` 패치 — da/ll 손실을 1:1로 더하던 걸 **ll 손실에 2.0배 가중치**
     (`TotalLoss`가 원래 `tversky_da_loss+tversky_ll_loss`로 동일 가중치였음 — da가
     훨씬 쉬운 태스크라 먼저 수렴해버리면 ll 쪽 학습 압력이 상대적으로 약해지는 문제)
  3. 증강 확장 — 기존 글레어 증강에 **모션블러**(`motion_aug`), **바닥 반사 스트릭**
     (`reflect_aug`, `frame_001082`류 케이스 겨냥), **가짜 얇은 선 distractor**
     (`add_distractor_lines`, 체크무늬 타일 그라우트선 등과 헷갈리지 않게 하는 하드
     네거티브) 추가, 매 증강 이미지마다 랜덤 조합
  4. **매 에폭 Drive 자동 백업**(`--drive_backup_dir`) — 기존엔 학습 다 끝나야 Drive에
     백업됐는데, 세션 끊기면 Colab 로컬 디스크가 날아가서 중간 진행상황이 통째로
     유실되는 문제가 있었음(470장은 에폭당 ~6분, 40에폭이면 4시간이라 세션 끊길 위험
     실제로 큼). 이제 매 에폭마다 `checkpoint.pth.tar`/`best.pth`를 Drive에 복사 →
     끊겨도 `--resume`으로 재개 가능.
  5. **ll 디코더 전용 LR 상향**(`ll_lr_mult=2.5`) — `model.py`의 `TwinLiteNetPlus`는
     encoder/CAAM은 공유하고 `up_1_da/up_2_da/out_da`(da 디코더)와
     `up_1_ll/up_2_ll/out_ll`(ll 디코더)이 완전히 분리된 구조. 전체 LR을 올리면 이미
     잘 되는 da까지 흔들릴 수 있어서, `named_parameters()`에서 `'_ll' in name`으로
     ll 디코더 파라미터만 골라 별도 optimizer param group으로 LR 2.5배를 줌. 단
     `utils.py`의 `poly_lr_scheduler`가 원래 매 에폭 모든 param group의 lr을 동일값으로
     덮어써서 이것도 같이 패치해야 함(`param_group.get('lr_mult', 1.0)` 반영하도록) —
     로컬에서 실제로 여러 에폭 시뮬레이션해서 비율이 유지되는 것까지 검증 완료.
- **다른 라벨러용 "약점 프레임" 데이터셋 생성**: `detect_curve_frames.py`(신규 — old
  모델 ll 출력의 연결 성분별 위/아래 1/3 구간 기울기 차이로 곡률 점수 계산, 전체 2123장
  대상 `curve_result.csv` 생성) + 기존 `triage_result.csv`(실패후보)/`cone_detect_result.csv`
  (콘)를 조합해서 `build_labeling_request_dataset.py` 작성. 이미 작업 중인 470장은
  제외, dHash로 근접 중복 제거, 카테고리별 상한(실패 60/콘 40/커브 40)을 둬서
  `labeling_request_hard_frames/`(실제로는 실패후보 20+콘 40+커브 40=100장, 실패후보는
  전체 23장 중 이미 470장에 포함된 3장 제외하니 정확히 20장 남음) 생성 완료. CVAT에
  새 Task로 올려서 다른 사람에게 배정하면 됨. + `labeling_handoff/`(LABELING_GUIDE.md +
  이 100장 + reasons.csv 묶음, 팀원에게 그대로 전달용) → `labeling_handoff.zip`으로 압축.

### 2.11 **[핵심 버그 발견]** letterbox vs plain resize 불일치로 이미지-라벨 좌표 어긋남

- 28에폭 체크포인트(`checkpoint.pth.tar`)를 받아서 bootstrap 12장 실제 GT로 직접 IoU를
  재보니 처음엔 0.038(YOLOPv2 0.412는커녕 옛날 74장 모델 0.055보다도 낮음)이 나와서
  당황했음 — 근데 이게 진짜 모델 문제가 아니라 **검증 스크립트의 좌표 정렬 버그**였고,
  그걸 추적하다가 **학습 파이프라인 자체의 훨씬 근본적인 버그**를 발견함:
  - `BDD100K.py`(TwinLiteNetPlus 레포 원본, BDD100K 720x1280 이미지 기준으로 짜여있음)의
    `Dataset.__getitem__`이 이미지는 `letterbox()`(종횡비 유지 + 여백 패딩)로 640x384에
    맞추는데, **라벨(마스크)은 그냥 `cv2.resize(label, (640, 360))`으로 독립적으로
    눌러 리사이즈**함. BDD100K의 16:9 이미지는 letterbox해도 우연히 위아래로만 12px씩
    패딩되니까(그래서 `loss.py`/`utils.py`에 `[12:-12]` 크롭이 있었던 것) 이 둘이 어쩌다
    맞아떨어졌던 것뿐임. **우리 이미지(640x480, 4:3)는 letterbox하면 좌우로 64px씩
    패딩되는데 라벨은 여전히 그 사실을 모르고 눌러 리사이즈되니까, 이미지와 라벨의
    같은 x좌표가 서로 다른 실제 위치를 가리키게 됨.** 사용자가 "우리 모델이 그린 차선이
    차선 밖에 그려져 있는 느낌"이라고 육안으로 지적한 것과 정확히 일치하는 증상.
  - **실차 코드로 교차검증**: `~/code/UMK/track_drive/track_drive/perception/dl_lane.py`
    84번째 줄 주석에 "전처리는 letterbox 없이 640x360으로 그냥 리사이즈"라고 명시돼
    있음 → **letterbox를 쓴 학습 파이프라인 쪽이 실차 방식과도 안 맞는 쪽(잘못된 쪽)**
    이라는 게 확정됨. (참고로 이 세션 내내 우리가 만든 평가/몽타주 스크립트들은 전부
    plain resize를 써왔어서 결과적으로 다 실차 방식과는 맞았음 — 학습 파이프라인만
    예외였음.)
  - 실제로 letterbox 대신 이미지/라벨 좌표를 맞춰서(loss.py가 실제로 보는 좌표계 그대로)
    다시 IoU를 재니 0.038 → **0.363**로 뛰었고, 이건 학습 로그의 IOU(0.349, 21에폭)랑
    거의 일치 — 즉 로그 숫자 자체는 처음부터 진짜였고, 문제는 그 좌표계가 실차와
    어긋나 있었다는 것.
- **수정**: `finetune_twinlitenetplus.ipynb` cell 17에 3개 파일 패치 추가
  - `BDD100K.py`: `letterbox(image, ...)` → `cv2.resize(image, ...)`, 라벨 리사이즈
    높이도 360 → 384(H_)로 통일 (letterbox 호출 4곳, 라벨 리사이즈 3곳)
  - `loss.py`, `utils.py`: 이제 필요없어진 `[12:-12]`/`[:,12:-12]` 크롭 전부 제거
    (loss.py 3곳, utils.py 3곳 — `val_one()`의 것까지 포함, 안 쓰지만 일관성 위해)
  - **결론: 지금 도는(또는 돌았던) 학습은 이 버그가 있는 상태로 진행된 거라 처음부터
    다시 돌려야 함.** 어긋난 좌표로 28에폭 학습된 가중치는 이어서 써도 의미 없음.
- **부수적으로 발견/수정한 멱등성 버그 2개**(둘 다 "레포 재클론 없이 셀 재실행"할 때
  터짐 — 이 노트북을 여러 번 고쳐 쓰는 과정에서 실제로 사용자가 두 번 다 직접 겪음):
  1. cell 17의 loss.py/utils.py/BDD100K.py 패치가 원래 "패치 전 원본 텍스트"를 assert로
     찾는 방식이라, 이미 패치된 파일에 다시 실행하면 AssertionError. → 각 패치 블록에
     "이미 패치됐으면 스킵" 가드 추가, 3번 연속 실행해도 에러 안 나는 것까지 검증함.
     (이 과정에서 크롭 제거 시 들여쓰기 공백을 안 맞춰서 다음 줄과 붙어버리는 문법
     에러도 하나 더 발견해서 같이 고침 — `ast.parse()`로 최종 검증.)
  2. cell 14(증강)가 재실행마다 이미 생성된 `_aug*.png` 파일까지 "원본"으로 착각해서
     또 증강 → 원본 400장이 5200장까지 폭증하는 사고가 실제로 발생함. → 실행 시작할
     때마다 기존 `_aug*` 파일부터 지우고 시작하도록 수정, 로컬 시뮬레이션으로 검증함.
- **폴더 정리**: 이번 세션에 만든 스크립트 18개를 `scripts/`로 이동(`__file__` 기반
  경로 쓰던 16개는 한 단계 위 프로젝트 루트를 가리키도록 다 고쳐서 재이동 후에도
  정상 동작 확인함). 중복 폴더(`pseudo_dataset 2/`, `pseudo_dataset 3/` — Finder 중복
  추정, 412MB), zip으로 이미 다 들어있는 원본 폴더(`labeling_handoff/`,
  `labeling_request_hard_frames/`), 로컬 검증용 임시 클론(`TwinLiteNetPlus_ref/`,
  `TwinLiteNetPlus_fresh_test/`), 다운로드만 하고 안 쓴 CLRerNet/UFLD-v2 원본
  아카이브(`clrernet_onnx/`, `ufldv2_onnx/`, 1.6GB) 삭제.

### 2.12 470장 재학습(small config) 완료 + da pseudo label 오염(letterbox 버그 유전) 발견

- §2.11 버그 수정 후 **470장 재학습을 실제로 진행** — 다만 Colab GPU 무료 할당량이
  중간에 두 번 소진됨(계정 하나로는 완주 불가):
  1. `finetune.py`에 **자동 재개 지원 추가**(노트북 cell-21) — Drive
     백업(`finetune_out_470_live/checkpoint.pth.tar`)이 있으면 자동으로 `--resume`을
     걸어 이어서 학습하도록 수정.
  2. 그래도 할당량이 다시 소진돼서 **학교 계정(`choijunsoo@kookmin.ac.kr`)으로 갈아탐** —
     Drive 폴더 공유의 "바로가기를 내 드라이브에 추가"가 잘 안 돼서, 결국 로컬에 있던
     `pseudo_dataset.zip`을 학교 계정 Drive에 새로 업로드 + `checkpoint.pth.tar`도
     `finetune_out_470_live/`에 수동으로 옮겨서 이어감.
  3. **medium config는 Colab에서 시간이 너무 오래 걸려서 `CONFIG='small'`로 되돌려서
     완주**함 — `medium`은 나중에 집 개인 PC로 따로 돌릴 예정(우선순위 낮춤, 아래
     "다음 할 일" 참고).
  4. **[버그 발견 및 수정]** `finetune.py`의 `best_da_miou`가 `checkpoint.pth.tar`에
     저장이 안 돼서, `--resume`으로 재개할 때마다 `-1.0`으로 리셋됨 → resume 직후
     첫 epoch은 실제 da mIoU 값과 무관하게 무조건 "새 best"로 찍혀서 더 좋았던
     `best.pth`를 더 나쁜 체크포인트로 덮어썼을 위험이 있었음. `best_da_miou`/
     `best_ll_iou`를 checkpoint에 저장/복원하도록 고침 + **`best.pth`(da 기준)와는
     별도로 `best_ll.pth`(ll IOU 기준)를 추가로 저장**하도록 수정(노트북 cell-19) —
     기존엔 best.pth가 da 기준으로만 갱신돼서 ll이 나빠져도 체크포인트가 안 남는
     문제가 있었음.
- **학습 완료** — 최종 결과물은 `~/Downloads/best.pth`/`best(1).pth`(small config,
  581KB, 채널 8~128 확인함). **주의**: 파일명에 숫자가 붙어도 두 파일 다 동일 학습의
  같은 시점(용량 완전히 동일) — 다른 버전 아님.

- **구모델(실차 배포중 `best.onnx`) vs 신모델(small,470) vs YOLOPv2(사전학습,
  파인튜닝 0회) 3-way 비교 몽타주 제작** (`scripts/make_three_way_compare_montage.py`,
  → `outputs/montages/three_way_compare_montage.png`). 프레임은 원래 실패/커브 등
  어려운 프레임 8장(진짜 완전실패 3장 + 커브+차선부분미검출 5장) : 원래 잘 잡던
  프레임 4장 = 2:1 비율로 선정(`triage_result.csv`/`curve_result.csv`/
  `missing_line_result.csv` 조합), 노트북 cell-12와 동일한 `seed=42` 셔플로
  train/val 여부도 재현해서 각 썸네일에 표시. **결과(12장 평균 ROI 커버리지)**:

  | | da 평균 | ll 평균 |
  |---|---|---|
  | 구모델 | 0.487 | 0.020 |
  | 신모델(small,470) | **0.604** | 0.038 |
  | YOLOPv2 | 0.447 | **0.048** |

  da는 신모델이 셋 중 제일 좋고 구모델도 확실히 이김. ll은 아직 YOLOPv2가 앞섬(이미
  §2.9에서 예상된 결과 — 그래서 YOLOPv2를 ll pseudo label 소스로 쓰는 것).

- **[중요 발견] da pseudo label(396장, 자동생성) 자체가 letterbox 버그로 학습된
  모델의 산출물이라 왜곡을 물려받았을 가능성 확인**. 경위:
  1. 부트스트랩 모델(§2.8 `twinlitenetplus_small_finetuned_refined.onnx`)은
     letterbox 버그가 **발견되기 전**(§2.11보다 시간상 앞섬)에 학습된 것.
  2. 이 모델로 470장 전체의 da pseudo label을 생성함(§2.9
     `build_pseudo_label_dataset.py`) — 스크립트 자체의 전처리(plain resize)는
     내부적으로 일관되지만, **그 안에서 쓰는 모델 가중치 자체가 이미 어긋난 좌표로
     학습된 상태**였음(학습 시 이미지는 letterbox로 좌우 64px 패딩, 라벨은 안 그럼 →
     모델이 "내용물은 가운데 512px에 있다"고 배웠는데 추론 땐 plain resize로 꽉 찬
     640px 이미지를 받음 — train/inference 전처리 불일치).
  3. bootstrap 74장 실제 GT로 이미 검증했던 da IoU 0.852(§2.9)가 낮게 나온 이유의
     일부가 이 왜곡 때문이었을 가능성.
  - **직접 검증**: `scripts/check_da_overpaint.py` 작성 — 신모델(방금 학습 끝난
    small,470)을 **사람이 직접 그린 74장 GT**에 돌려서 과다포함(FP)/과소포함(FN)
    비율 측정. 결과: mean IoU=0.877, FP/GT=6.9%, FN/GT=6.3% — **극적인 과다포함은
    아님**(대부분 경계선이 살짝 두꺼운 정도 + 라바콘 검은 받침대를 도로로 인식 못
    하는 게 주된 오차, `outputs/montages/da_overpaint_check.png`).
  - 근데 **3-way 몽타주의 "신모델" 컬럼만 잘라서 다시 보니(`three_way_col2_new_only.png`)
    커브 프레임들(`001372`/`000851`/`001693`/`001431`/`001430`)에서 파란색(da)이
    실제 트랙 경계 훨씬 밖까지 넓게 칠해짐**을 육안으로 확인. 이 프레임들이 어느
    쪽(사람라벨 74 vs 자동생성 396)인지 대조한 결과 **전부 자동생성 pseudo label
    (396장) 쪽**이었고, 사람 라벨(74장)인 `001884`만 상대적으로 덜함 — 위 가설과
    정확히 일치. **결론: da pseudo label의 왜곡이 특히 커브 구간에서 두드러지고,
    신모델이 그 왜곡(도로 폭 과대평가)을 학습 과정에서 그대로 물려받음.**
  - 참고: ll pseudo label은 YOLOPv2(외부 사전학습, letterbox 버그 체인과 완전
    무관)에서 나온 거라 이 문제와 무관함 — da에만 해당.

### 2.13 GitHub 레포 생성 + README 정리

- 이 프로젝트가 지금까지 git 저장소가 아니었음 → `git init` + `.gitignore`(데이터셋/
  모델가중치/외부 클론 레포/venv 제외, 코드+문서+몽타주 이미지만 추적) 세팅 후
  **private** 레포로 GitHub에 올림: https://github.com/mastic-choi/TwinLiteNet-KMU-finetune
  (`gh repo create`로 생성, `mastic-choi` 계정).
- `README.md` 작성 — 배경(원조 TwinLiteNet 실패 사례), 베이스 모델(TwinLiteNetPlus)
  링크, 학습 방법론 요약(데이터 다이어트→CVAT 라벨링→완전자동 pseudo-labeling→
  letterbox 버그/편향 대물림/체크포인트 버그 발견·수정→검증 방법론→Colab/로컬
  학습 환경), 관련 레포(TwinLiteNetPlus/TwinLiteNet/YOLOPv2/UMK) 링크.
- `outputs/montages/`(24장)·`outputs/scratch/`(27장)도 추적 대상에 포함(용량
  ~107MB, 개별 파일 최대 18MB로 GitHub 제한 안 걸림) — `outputs/onnx/`(모델
  가중치)만 계속 제외. README에 "결과 (몽타주)" 섹션 추가해서 대표 몽타주 4개
  (3-way 비교/bootstrap_v2 커브 비교/da 과다포함 검증/pseudo_dataset_v2 스팟체크)
  이미지로 직접 임베드함.
- **[사용자 판단 보류]** "TwinLiteNetPlus를 포크했어야 하지 않나?"는 질문에 —
  실제 코드 패치(BDD100K.py/loss.py/utils.py)가 노트북 cell 안 문자열 치환으로
  런타임에만 적용되고 정적으로 커밋된 적이 없어서, 지금 형태(별도 레포)가 더
  적합했다고 판단. 다만 **패치된 최종 파일 스냅샷(또는 .patch)을 이 레포에
  커밋해두면 원본 대비 diff가 명확해질 것**이라는 개선안은 사용자가 "이따가"로
  보류함 — 나중에 요청 있으면 진행.

### 2.14 40epoch(WSL/ROCm) 학습 완료 + README 개편 + 팀 피드백 (2026-08-11 저녁, 6회차)

- 이 세션에서: Windows 네이티브 ROCm은 RX 9070 XT(gfx1201)용 MIOpen 커널이 사전
  컴파일 DB에 없어서 런타임 JIT 컴파일 중 `type_traits not found`로 크래시(GitHub에
  동일 사례 다수, AMD도 "Windows 지원 미성숙"이라 인정) → **WSL2 Ubuntu-22.04로
  전환**해서 해결(amdgpu-install --usecase=wsl,rocm + repo.radeon.com의 rocm7.2
  manylinux 휠). medium config 40epoch(1068장, batch_size=8) 학습 완료 — 에폭당
  안정적으로 ~275초, 총 3시간 4분. 최종 **da mIoU 0.937**(21epoch부터 수렴),
  **ll IOU 0.567**(28epoch, `best_ll.pth`). batch_size는 32→OOM, 16↔8 속도 차이
  거의 없어서(모델이 47만 파라미터급으로 워낙 작아 MIOpen fallback 커널의 compute가
  병목으로 추정) 8 유지.
- `outputs/models/best.onnx`(+`.onnx.data`)로 export 완료(PyTorch 대비 오차
  <1e-4, opset 12 요청했으나 torch 2.9.1 dynamo exporter가 18로 자동 상향).
  README.md를 결과 중심(GIF 3개 + 라벨링 파이프라인 몽타주 + 최종 성능 표)으로
  전면 개편, letterbox 버그 등 상세 트러블슈팅 서술은 README에서 빼고 이 문서에만
  남기기로 함(README는 "프로젝트 대문", 시행착오는 PROGRESS.md 전담).
- 원조 TwinLiteNet vs 우리 모델 실주행 비교 GIF 3개 제작(`dataset/` 2123장 원본에서
  `scripts/detect_curve_frames_pth.py`/`detect_turn_direction.py`로 곡률+방향
  자동 검출해서 커브/좌회전 구간 선별): 커브(frame_000800~935, S자),
  좌회전 단독(frame_000400~480), 직진 대조군(frame_000933~985) —
  `outputs/montages/{curve,left_turn,straight}_old_vs_new.gif`.

**팀 피드백(2026-08-11 21:35, 윤성·최준수, 카톡 — 원문 요약)**:
- 기존에 인식 불가능하던 부분에서 큰 진전 확인, 전반적으로 안정성이 늘었다는 인상.
- **[신규 발견] 차선과 차량이 평행하지 않고 각도가 생기는 순간 da(주행가능영역)가
  많이 흔들림.** 지금까지 raw 데이터가 정상 주행(차량이 대략 차선과 평행하게
  진행) 위주라, 각도가 있는 상황 자체가 학습 데이터에 부족했을 가능성이 큼 —
  지금 GIF들의 커브/좌회전도 "차량은 정면 유지, 차선만 휘는" 경우라 이 문제와는
  다른 케이스. 최준수 가설: 실차로 **지그재그 주행**하며 raw를 따면 이 각도
  상황이 자연히 데이터에 포함될 것.
- 로컬(RX 9070 XT/ROCm) medium 학습 시간 재확인 및 팀 공유: **~4시간, Colab으로는
  절대 불가능** — 앞으로도 로컬 학습 기본, 가능하면 Colab 시도 안 함.
- 방향성: "안 되는 부분 위주로" 데이터를 모아서 다음 학습에 반영하는 흐름을
  계속 유지하기로 함(기존 커브/글레어/콘 처럼, 이제 "비평행 각도"도 그 목록에 추가).

### 2.15 da를 ll 경계로 크롭(clip)하는 아이디어 탐색 — width-borrow로 잠정 결론 (2026-08-11 밤, 6회차)

**동기**: pipeline_montage에서 커브 구간(`frame_000851` 등)의 da가 트랙 경계 밖으로
넘치는 게 보임 → "da 예측을 YOLOPv2 ll 라인으로 잘라내면 어떨까"라는 질문에서 시작.
§2.9에서 이미 순수 geometric(ll 두 선 사이=da로 통째 재구성) 방식은 IoU 0.375로
실패했다고 기록돼 있었는데, 이번엔 **재구성이 아니라 기존 da 예측의 바깥쪽만 잘라내는
보정** 방식으로 다시 시도 — 여러 버전을 만들고 실제로 눈으로 확인하며 반복 개선함.

**시도한 방법과 실패 원인 (전부 `scripts/pseudo_label/`에 스크립트 남아있음)**:

1. **row별 ll 전체 min~max로 자르기** (`clip_da_with_ll_demo.py` 최초 버전) — 한쪽
   선만 보이는 row에서 그 선 자체의 폭(수 px)으로 da가 통째로 뭉개짐(34% 날아감).
2. **성분을 화면 좌/우 절반으로 분류해서 자르기** — 커브에서 화면 전체 평균이 아니라
   그 row에서의 실제 좌/우가 뒤바뀌는 경우 발생, 중앙 점선도 엉뚱하게 좌/우 중
   하나로 분류돼서 트랙 한가운데를 잘라먹음(`frame_001708`에서 확인).
3. **da 자체 가장자리 40px 이내의 선만 신뢰** — 안전하지만 너무 보수적, 멀리 떨어진
   노이즈 섬을 못 잡음.
4. **연결성분 중 "가장 긴 2개"를 좌/우 경계선으로 채택** — 소실점 근처에서 좌/우 선이
   실제로 화면상 붙어버리는 경우(`frame_000434`, 물리적으로 실재하는 현상, 검출
   오류 아님) 하나의 성분에 양쪽이 섞여서 트랙 대부분을 잘라먹음(43.7% 오삭제).
5. **row별 ll 픽셀 사이 최대 gap이 절대/상대(row의 da 폭 대비) 임계값을 넘을 때만
   자르기, 애매하면 그 row는 절대 안 건드림** (`make_da_clip_dataset_montage.py`) —
   `frame_000434`(오작동)와 `frame_000851`(정상 작동)의 gap 비율을 직접 재보니
   **0.41~0.57로 완전히 겹침** — row 하나만 보는 방식으로는 "같은 선의 내부 gap"과
   "진짜 좌우 분리"가 통계적으로 구분 불가능하다는 게 확인됨(수학적으로 한계 증명).
   → 이 상태로 "타협"하기로 했었으나, `frame_000876`에서 **한쪽 선이 아예 없는
   row가 45%(104/230)**라 그만큼 못 잡는다는 게 드러나 다시 탐색.

**추가로 낸 아이디어 3개 비교** (`compare_clip_ideas_multi.py`, 프레임
871/876/880/884/890):

| 방법 | 요약 | 결과 |
|---|---|---|
| **width-borrow** | 양쪽 선 다 보이는 "확신 row"의 폭을, 한쪽만 보이는 row가 빌려씀 | **채택** — 5장 다 안정적으로 8~26% 감소, 파괴적 오류 없음 |
| curve-fit | 확신 row들의 좌/우 점만 모아 2차 다항식 피팅 | 대체로 비슷하나 `frame_000871`에서 다항식이 트랙 중앙을 잘못 파먹는 오류 발생(31.7%가 다 정상 삭제는 아님) |
| da-smoothness | ll 안 쓰고 이웃 row 대비 da 폭이 튀는 곳만 제거 | 사실상 무효(0~0.5%) — 침범이 급격한 turn이 아니라 완만한 추세라 "튐 감지"로는 안 잡힘 |

**HSV 색상 기반 필터도 시도**(`compare_hsv_idea.py`, ll 아예 안 씀 — "실제 데이터셋은
색/질감으로 도로를 구분한다"는 아이디어) — 화면 하단 확실한 트랙 픽셀에서 HSV 범위를
캘리브레이션(H:20~130, S:5~200, V:30~255)했으나, **글레어/반사 때문에 회색 바닥과
트랙 색이 겹쳐서** 실제 침범 영역은 거의 못 거르고(0~6.8%) 오히려 트랙 안쪽에 작은
오탐 반점만 생김. 폐기.

**현재 결론**: **width-borrow**가 시도한 것 중 제일 안정적. 다만:
- `frame_000434`류(소실점 근처 좌우 합쳐짐)에서 width-borrow가 안전한지는 **아직
  미확인** — 다음에 이어서 할 것.
- 1068장 전체에 적용했을 때 실제로 GT 대비 정확도가 좋아지는지 **정량 검증이 전혀
  안 됨** — bootstrap 74장 사람 GT가 이 컴퓨터엔 없음(Mac에만 있음), 있어야 진짜
  검증 가능.
- 아직 라벨을 실제로 갱신하거나 재학습에 반영하지 않음 — 전부 시각적 비교 몽타주
  단계.

## 3. 다음 할 일 (미완료, 우선순위 순)

-2. **[최우선, 2026-08-12(내일) 예정 — 팀 계획]** 차선-차량 비평행(각도 있는 상황)에서
    da가 흔들리는 문제 대응:
    1. 실차로 **지그재그 주행**하며 raw 프레임 수집 (최준수, 내일 낮)
    2. 수집한 프레임 라벨링 — 기존 파이프라인 그대로 재사용 가능: da는 지금
       `best.pth`(또는 `best_ll.pth`) pseudo-label, ll은 YOLOPv2+skeleton
       정제(`scripts/build_pseudo_label_dataset_v2.py` 계열)
    3. 기존 `pseudo_dataset_v2`(1068장)에 병합 후 로컬(WSL/ROCm)에서 medium
       config 재학습 (내일 저녁, 최준수 집 PC — 예상 ~3~4시간)
    4. 학습 후 **비평행/각도 있는 프레임 위주로 old-vs-new 비교**해서 실제
       개선됐는지 검증. 이 세션에서 만든 `scripts/scan_old_vs_new_gap.py`
       (전체 프레임 ROI 커버리지 스캔) 재사용 가능하되, 지금은 곡률(차선이
       휘는 정도) 기준으로 프레임을 골랐던 걸 **"차선-차량 각도(비평행도)"
       기준**으로 바꿔서 골라야 함 — 아직 이 각도를 정량화하는 로직은 없음,
       새로 작성 필요(예: ll 마스크의 전체 기울기 vs 화면 중심축 각도 차이로
       근사 가능할 듯).
-1.5. **[다음에 이어서 할 것, §2.15]** da-클리핑(width-borrow) 마무리:
    1. `frame_000434`류(소실점 근처 좌/우 선이 화면상 붙는 경우)에서 width-borrow가
       안전한지 확인 — 아직 테스트 안 함(871/876/880/884/890 5장에만 적용해봄).
    2. bootstrap 74장 사람 GT를 Mac에서 가져와서(또는 이 컴퓨터에서 접근 가능한
       방법을 찾아서), width-borrow 적용 전/후 da가 GT 대비 IoU로 실제 개선되는지
       정량 검증 — 지금까지는 전부 시각적 판단뿐, 숫자 근거 없음.
    3. 검증되면 `pseudo_dataset_v2`의 1068장 da_masks 전체에 width-borrow 적용해서
       라벨 갱신 → 재학습 여부 결정.
-1. **[완료, 2026-08-11 §2.14]** ~~9070 XT medium 학습 끝나고 순정 TwinLiteNet 비교
    GIF 만들어서 README에 추가~~ — 완료함(`outputs/montages/{curve,left_turn,
    straight}_old_vs_new.gif`, README "실제 주행 비교" 섹션).
0. **[진행 중] bootstrap_v2 Stage 1 완료 + 1027장으로 확장한 pseudo_dataset v2 생성 중.**
   - **완료**: `bootstrap_v2.zip`(134장, 기존74+신규60) 만들어서 Colab Stage 1 학습
     완료(small config, 40 epoch) → `twinlitenetplus_small_bootstrap_v2.onnx`
     받음 → `outputs/onnx/`에 보관, onnxruntime으로 정상 동작 검증 완료(입출력
     `images`/`da`/`ll`, track_drive `dl_lane.py` 스펙과 일치).
   - **완료**: `scripts/compare_bootstrap_v2_da.py`로 순정 TwinLiteNet vs
     small,470(오염) vs bootstrap_v2 3-way 비교(74장 GT 정량 + 커브 프레임 시각,
     `outputs/montages/bootstrap_v2_vs_old_curve_montage.png`). **결과**: 순정은
     셋 중 압도적 최하(글레어 프레임 등 완전실패 다수). small,470 vs bootstrap_v2는
     **ll은 bootstrap_v2가 거의 전 프레임에서 확실히 좋음**(YOLOPv2 기반이라 예상된
     결과), **da는 애매함**(74장 GT 기준 IoU 0.877→0.857로 오히려 소폭 하락, 커브
     경계 과다포함은 줄었지만 대신 콘더미/벽 등에 노이즈성 오탐 패치가 생김 — 134장/
     40epoch로는 470장보다 데이터가 부족해서 생긴 현상으로 추정). epoch 증량 재시도
     여부는 보류하고 일단 이 모델로 다음 단계 진행하기로 함(2026-08-11 결정) — da의
     "체계적 편향"은 줄었으니 pseudo label 소스로는 여전히 이전보다 나을 것으로 판단.
   - **결정(2026-08-11)**: da/ll이 이제 완전 자동 생성이라 사람 라벨링 병목이 없음 →
     **더 이상 470장에 묶일 필요 없다고 판단**, `dataset/`(2123장 원본) 전체 기준으로
     dHash 근접중복 제거 상한까지 확장 → **1027장**으로 pseudo_dataset v2 재구성
     하기로 함(658장(stride=2)/1027장(stride=1, 선택됨) 중 사용자가 1027 선택).
     `scripts/build_diet_v2.py`(`diet_dataset.py`의 `SUCCESS_STRIDE`를 3→1로 완화한
     버전) 실행 완료 → `dataset_diet_v2/`(1027장: 실패후보153+콘141+일반성공733).
   - **완료**: `scripts/build_pseudo_label_dataset_v2.py`로 1027장 전체에 da(=
     bootstrap_v2 모델)+ll(=YOLOPv2+skeleton) pseudo label 생성 → `pseudo_dataset_v2/`
     (YOLOPv2 ll 미검출 프레임 1/1027뿐, 양호).
   - **완료**: `scripts/merge_human_labels_v2.py`로 사람 라벨(74+60=134장) 병합 —
     da/이미지는 `bootstrap_v2/` 그대로, **ll은 버그 수정**: 기존 74장은
     `bootstrap_refined/ll_masks`(skeleton 정제, 8px 고정 — `bootstrap_v2/ll_masks`에
     원본 브러시 두께로 잘못 들어가 있던 걸 바로잡음), 신규 60장은
     `bootstrap_v2/ll_masks`(YOLOPv2+skeleton, 이미 8px) 그대로. dedup 단계(1027장
     선별)에서 빠졌던 사람 라벨 프레임 41장은 별도로 추가 편입 → **최종
     1068장**(사람검증 134 + 자동생성 934).
   - **완료**: 자동생성 934장 중 12장 랜덤 스팟체크(`outputs/montages/
     pseudo_dataset_v2_spotcheck.png`) — 대체로 da가 트랙 경계에 잘 붙고 커브도
     매끄러움, 일부(`frame_000926` 등) 벽 쪽 노이즈 패치 소수 잔존(§2.12에서 확인된
     bootstrap_v2 특성, 치명적이진 않음).
   - **완료**: `pseudo_dataset_v2.zip`(471MB) 로컬 생성 완료. 노트북 cell-12/cell-21/
     cell-23/cell-25(원래 순번 기준)를 `pseudo_dataset_v2`/`finetune_out_v2` 경로로
     갱신 완료. **[사고 기록]** 이 과정에서 `NotebookEdit`에 `cell_id="cell-21"`처럼
     원본 셀에 저장된 id가 없는(구버전 노트북 셀들은 `id` 필드 자체가 없음) 위치
     기반 라벨을 줬다가, 그 사이 삽입한 새 셀들 때문에 위치가 밀려서 **엉뚱한 셀(Stage
     1 데이터 준비 셀)을 덮어쓰는 사고**가 있었음 → python으로 직접 `.ipynb` JSON을
     열어 인덱스 확인 후 복구. **교훈**: 셀 삽입/삭제로 위치가 바뀔 수 있는 노트북에서
     `cell_id`로 숫자 위치 라벨(`cell-N`)을 쓸 땐, 그 라벨이 stored id가 아니라
     Read 시점 위치 기반일 수 있음을 감안하고 직전에 다시 확인할 것.
   - **다음(이어서 할 일)**:
     1. 로컬 `pseudo_dataset_v2.zip`을 Drive(`{DRIVE_ROOT}/pseudo_dataset_v2.zip`)에
        업로드.
     2. Colab에서 노트북 0~11 → 16~19 → (필요시 Stage 1은 이미 완료했으니 생략) →
        cell-12(pseudo_dataset_v2 준비, "## 3" 섹션) → cell-21(학습, "## 7" 섹션,
        savedir `finetune_out_v2`) 순서로 실행. 1027장은 470장 대비 학습시간 대략
        2배 예상 — Colab 할당량 소진 위험 더 큼을 감안할 것.
     3. 학습 끝나면 `scripts/check_da_overpaint.py`/`compare_bootstrap_v2_da.py` 계열로
        74장 GT 기준 재검증 — 노이즈 패치 문제가 1068장 스케일에서 해소됐는지 확인.
     4. 이후 나머지(`labeling_handoff.zip` 중 아직 미완료인 40장 포함) 데이터셋도
        팀원이 순차적으로 추가 라벨링 예정(사용자 계획).
2. **medium config 학습을 집 개인 PC(Windows, AMD RX 9070 XT, 시스템램 64GB)로 진행** —
   Colab 무료 GPU로는 시간이 너무 오래 걸려서(§2.12) small로 대체 완주함.
   **완료**: `finetune_twinlitenetplus_local_windows_rocm.ipynb` 작성 — Colab판과
   데이터 파이프라인/패치(letterbox 수정·ll loss 가중치 2.0배·ll LR 2.5배·
   best_da/best_ll 분리저장)는 동일, 차이는 (1) Google Drive 마운트 대신 로컬
   `BASE_DIR` 사용, (2) ROCm PyTorch 설치 셀 추가(RX 9070 XT는 ROCm 7.2 공식
   지원 대상, Windows는 **Python 3.12 + Adrenalin 드라이버 26.2.2 이상** 필요 —
   `torch[+torchvision/torchaudio] rocm-rel-7.2.1` cp312 wheel을
   repo.radeon.com에서 직접 설치), (3) `torch.cuda.*` API가 ROCm/HIP에서도
   그대로 동작해서 `finetune.py` 코드 자체는 무수정 재사용 가능. 데이터셋은
   `pseudo_dataset_v2`(1068장) 기준으로 설정해둠.
   **다음 할 일**: 이 노트북 + `pseudo_dataset_v2.zip`을 Windows PC로 옮기고
   `BASE_DIR` 경로 수정 후 실행. ROCm 버전 번호(7.2.1)는 실행 전에 공식 문서
   (rocm.docs.amd.com)에서 최신인지 재확인할 것 — 확인 시점 이후 버전업됐을 수 있음.
   VRAM 여유 있으면(9070 XT면 충분할 가능성 높음) `BATCH_SIZE`를 8보다 올려서
   학습시간 단축 시도해볼 것.
3. `finetune_out_470_live/best.pth`(da 기준) 말고 **`best_ll.pth`(ll 기준, §2.12에서
   추가)도 Drive에서 받아서 3-way 몽타주/과다포함 체크를 한 번 더 돌려볼 것** — da
   기준 best와 ll 기준 best가 다른 epoch일 수 있어서, 최종적으로 어느 쪽을 실차에
   반영할지 비교 필요.
4. `pseudo_dataset`에 사람 라벨(74장) 병합은 완료(§2.10, `merge_human_labels.py`) —
   단 예전 `bootstrap/` 스냅샷 기준이라 그 뒤 CVAT에서 팀원이 더 완료한 프레임은 아직
   미반영. **CVAT Jobs 목록 CSV**(Task → Jobs 탭 → export) 받으면 최신 completed
   목록으로 `merge_human_labels.py` 다시 돌릴 것. (커브 프레임은 특히 사람 라벨
   비중을 늘리는 게 위 1번 문제의 근본 해결책이 될 수 있음 — pseudo label
   재생성만으론 여전히 모델 편향에 의존하는 한계가 있음.)
5. `labeling_handoff.zip`(100장, §2.10)은 **CVAT에 업로드되어 팀원이 60장(da만) 완료함**
   (§3의 0번 참고, 2026-08-11 확인) — 남은 40장도 마저 라벨링 요청할 것.
6. 팀원들이 CVAT에서 ll을 브러시 대신 **Polyline 도구**로 그리도록 전환할지 결정(§2.7) —
   전환하면 CVAT export 포맷을 COCO 대신 "CVAT for images 1.1"(XML, polyline 네이티브
   지원)로 바꿔야 함, 아직 미검증. (지금은 어차피 ll을 YOLOPv2 자동 생성으로 대체하는
   쪽으로 가고 있어서 우선순위 낮아짐 — 사람 검증이 필요한 프레임에만 해당.)
7. 2123장 유실 버그(§2.1, 20장)는 여전히 미해결 — 그 20장을 `dataset/`(2123장 원본)에서
   다시 꺼내서 수동으로 `raw/`에 추가할지 여부 결정 필요.
8. 최종 모델 나오면 노트북 8번 섹션(ONNX export)·9번 섹션(실차 반영) 그대로 따라가면 됨 —
   `track_drive/track_drive/config.py`의 `DL_INPUT_H`를 360→384로 바꿔야 하는 것 포함.
   **실차에 바로 쓰기 전에 반드시 §2.9의 SAM 관련 결론과 무관하게 실제 주행 테스트로
   검증할 것** — 지금까지는 전부 정적 이미지/ROI 커버리지 기준 검증이라 실주행 특성
   (지연시간, 프레임 간 흔들림 등)은 아직 확인 안 됨.
9. CLRerNet은 다운로드 리소스(`clrernet_onnx/`, 1.9GB)를 정리 삭제함(§2.11) — 필요하면
   `scripts/clrernet_decode.py`(작성해둔 numpy 디코더)와 함께 나중에 처음부터 재시도.
   우선순위 낮음(YOLOPv2로 이미 충분히 좋은 ll 소스 확보됨).

## 4. 환경/도구 메모

- 로컬 파이썬 가상환경 2개:
  - `/Users/mastic-choi/code/fine-tune/.venv_triage` — opencv, numpy, onnxruntime,
    pycocotools, **scikit-image**(skeleton화용, 이번 세션에 추가 설치) (마스크/CVAT 관련
    스크립트 대부분 이걸로 실행)
  - `/Users/mastic-choi/code/fine-tune/.venv_yolo` — python3.13 기반, ultralytics + torch
    (MPS 가속 확인됨) + onnxruntime + **transformers/accelerate/timm/tensorboard/addict/
    pathspec/scikit-image/gdown**(4회차 세션에 추가 설치, Grounding DINO/SegFormer/
    Mask2Former/UFLD-v2 테스트용). 콘 검출(`cone_best.pt`)이랑 SAM(`sam2.1_b.pt`,
    ultralytics가 자동 다운로드) 둘 다 이 venv로 실행. `build_pseudo_label_dataset.py`,
    `make_ensemble_sam_drafts.py`도 이 venv 필요(torch/YOLOPv2 쓰니까).
- **4회차 세션에 클론한 외부 레포**(`/Users/mastic-choi/code/fine-tune/` 바로 아래) —
  `YOLOPv2/`(✅ 채택, `data/weights/yolopv2.pt` 156MB 사전학습 가중치 포함, ll 소스로
  실전 사용 중)만 남기고, `Ultra-Fast-Lane-Detection-v2/`, `CLRerNet/`, `PINTO_CLRerNet/`,
  `CLRNet/`, `GANet/`, `CondLaneNet/`, `PETR/`, `LaneGAP/`(전부 §2.9에 정리된 이유로
  안 씀 — CUDA 전용이거나 멀티카메라 전제라 구조적으로 불가하거나 우리 도메인에서
  거의 전멸)는 아직 코드는 남아있음(디스크 차지, 필요 없으면 삭제 가능). 다운로드만
  하고 안 쓴 `clrernet_onnx/`, `ufldv2_onnx/`(합쳐서 1.6GB)는 4회차 후반에 삭제함.
- **스크립트 위치 (6회차에 재정리됨)**: "메인에 파일이 너무 많다"는 지적으로 최상위에
  흩어져 있던 스크립트 15개 + `scripts/` 바로 밑에 평평하게 있던 38개(총 53개)를
  전부 **기능별 하위 폴더로 이동**함 — `scripts/{data_prep,cvat,pseudo_label,eval,
  montage,export}/`. 이제 최상위엔 `.py` 파일이 하나도 없음(README.md/PROGRESS.md/
  .gitignore/노트북 3개만). 프로젝트 루트를 가리키던 `os.path.dirname(...)` 패턴은
  전부 깊이 변화(1단계 더 깊어짐)에 맞춰 dirname 호출을 하나씩 추가해서 일괄 수정함
  (예: 기존 `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` →
  `os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`).
  폴더 간 크로스 임포트(`make_da_from_ll_montage.py`가 `pseudo_label/da_from_ll.py`를
  쓰는 경우)는 `sys.path.insert`로 명시 처리. **이 문서(PROGRESS.md)의 위 시간순
  기록(1~5회차)에 나오는 스크립트 경로 언급은 전부 이동 전 기준**이니, 실제 파일을
  찾을 땐 파일명으로 검색하거나 위 카테고리 분류를 참고할 것.
- **5회차 세션에 추가한 로컬 스크립트/파일**:
  - `scripts/make_three_way_compare_montage.py` — 구모델(onnx)/신모델(pth)/YOLOPv2
    3-way 비교 몽타주, 재사용 가능(§2.12)
  - `scripts/check_da_overpaint.py` — 신모델 da를 사람 GT(bootstrap 74장) 대비
    과다포함(FP)/과소포함(FN) 비율로 검증(§2.12)
  - `_TwinLiteNetPlus_ref/`(프로젝트 루트) — 로컬에서 `TwinLiteNetPlus`의 `model/model.py`
    아키텍처 코드가 필요해서 얕은 클론해둔 것(state_dict만으론 모델을 못 만들어서
    필요). 3~4회차 때처럼 다 쓰고 삭제하는 대신 이번엔 재사용 스크립트가 계속
    참조하므로 **남겨둠** — 나중에 필요 없어지면 삭제해도 무방(다시 클론하면 됨).
  - `~/Downloads/best.pth`/`best(1).pth` — 5회차에 완료된 470장 small config 학습
    결과(둘 다 동일 파일, 브라우저가 중복 다운로드로 이름만 다르게 붙인 것).
    `finetune_out_470_live/best_ll.pth`(ll 기준)는 아직 로컬에 안 받음(다음 할 일 참고).
- **폴더 정리**(3~4회차 세션): 몽타주/스크래치 PNG는 `outputs/montages/`, `outputs/scratch/`로,
  초기에 시도했다 폐기한 CVAT import 방식(§2.3의 1·2번)은 `archive/cvat_import_experiments/`로
  이동. `raw 2/`(Finder 중복), `_cvat_export_extracted/`(임시), `dataset_balanced/`,
  `pseudo_dataset 2/`·`pseudo_dataset 3/`(Finder 중복 추정), `labeling_handoff/`·
  `labeling_request_hard_frames/`(zip에 이미 다 들어있음), 로컬 검증용 임시 클론
  (`TwinLiteNetPlus_ref/`, `TwinLiteNetPlus_fresh_test/`)은 삭제함. `umk_twinlite_raw.zip`/
  `images_todo_470.zip`(이미 Drive 업로드 완료, 로컬 원본과 중복)은 확인 차 일단 유지.
- Mac 사양: Apple M4, 16GB RAM, MPS(Metal) 가속 가능 — SAM2-base 정도는 로컬에서 무리 없이
  돌아감(장당 ~1초).
- CVAT 접근은 **Claude in Chrome**(`mcp__claude-in-chrome__*`)로 사용자의 실제 로그인된
  Chrome을 직접 조작해서 확인/수정함(스크린샷만으로 안 되는 것들 — 예: Drive 중복 폴더
  발견 및 정리를 이 방법으로 처리함).
- `track_drive` 실제 코드 위치: `/Users/mastic-choi/code/UMK/track_drive/`
  (config.py, perception/dl_lane.py 등 — ROI/임계값/입출력 스펙의 출처).
- TwinLiteNetPlus 레포: `https://github.com/chequanghuy/TwinLiteNetPlus`(Colab에서 매번
  클론). 로그의 `X/299` 표시는 `utils.py`에 하드코딩된 `300-1` 상수라 실제 학습과 무관
  (무시할 것). **(4회차 정정)** `BDD100K.py`/`loss.py`/`utils.py`는 GitHub 원본 자체를
  고칠 순 없지만, Colab에서 클론된 로컬 사본은 노트북 cell 17에서 문자열 치환으로
  실제 패치함(§2.7의 ll 손실 가중치, §2.9의 ll LR 상향, §2.11의 letterbox 버그 수정
  전부 이 방식) — 재클론 없이 셀을 여러 번 실행해도 안전하게 멱등 처리돼 있음.

## 5. 자잘한 교훈 (다시 안 헤매려고 적어둠)

1. Google Drive에 같은 이름 폴더가 여러 개 생기면 Colab `drive.mount()`가 어느 쪽을
   가리키는지 못 정해서 파일이 0개로 보임 — 항상 폴더 중복 여부 먼저 의심할 것.
2. CVAT `Upload annotations`는 zip의 `labelmap.txt` 색상이 아니라 **라벨 이름**만
   프로젝트 라벨과 일치하면 됨(색은 CVAT 표시용일 뿐, 안 맞아도 무방).
3. CVAT 라벨 export에서 `background`처럼 실제 안 그린 라벨도 카테고리 목록엔 나타날 수
   있음 — 실제 사용된 카테고리인지 annotation 존재 여부로 확인할 것.
4. torch 최신버전 onnx export → `onnxscript` 필요 + 가중치 외부 파일(`.onnx.data`) 분리될
   수 있음, 둘 다 같이 챙길 것.
5. 몽타주/비교 이미지를 만들 때 **ROI 밖 시각 정보에 속지 말 것** — 항상 실제 기능
   ROI(`y=250:390`) 기준 숫자로 재확인.
6. CVAT 무료 플랜은 AI Tools(SAM 등) 호출에 월별 한도가 있음 — 소진되면 그 달엔 무료로
   못 씀(우회 불가). 오픈소스 SAM을 로컬에서 직접 돌리면 한도 없이 대체 가능(`ultralytics`
   패키지가 제일 간단).
7. **(4회차에 정정)** SAM은 da(큰 뭉친 영역)에도, ll(얇고 긴 구조)에도 결국 둘 다 안
   맞았음 — 처음엔 "da는 되고 ll만 안 된다"고 생각했는데, bootstrap GT로 직접 검증해보니
   **완벽한 GT를 그대로 넣어도 da IoU가 1.0→0.75로 떨어짐**. SAM은 "시각적으로 이어진
   영역"을 통째로 채우는 방식이라, da/ll 경계가 "시각적 경계"가 아니라 "차선이 정의하는
   의미적 경계"인 우리 트랙에서는 프롬프트를 점/박스 뭘 줘도 구조적으로 못 맞춤. §2.9 참고.
8. CVAT에서 "Task 전체 Upload annotations"가 이미 completed된 job의 annotation까지
   덮어쓰는지 아직 검증 안 됐음 — 부분 업데이트가 필요한 상황(지금이 그럼)에서는 실행 전에
   반드시 이걸 먼저 확인할 것(작은 테스트 job으로 먼저 시도해보는 걸 권장).
9. **GT(정답)가 실험 중간에 바뀌면(예: 라벨 두께 재정의) 절대 학습 로그의 지표 숫자를
   전/후로 직접 비교하지 말 것** — 같은 품질이어도 GT가 더 엄격해지면 숫자가 구조적으로
   낮게 나옴. 항상 별도 스크립트로 실제 이미지에 다시 돌려서 비교할 것(§2.9의 refined
   학습 로그 오판 사례).
10. **"커버리지(coverage)"나 "면적 비율" 같은 단순 통계만으로 모델 품질을 판단하지 말 것**
    — SegFormer/Mask2Former가 da 커버리지 0.87로 "깨끗해 보였지만" 실제 GT와 IoU를
    재보니 0.75였고 30~50% 과다예측이었음. 반드시 사람이 라벨링한 GT가 있으면 IoU로
    직접 검증할 것 — 없으면 "카더라"임을 인지하고 확언하지 말 것.
11. SAM에 박스(bbox) 프롬프트를 줄 때, 대상이 사각형이 아니라 **사다리꼴/삼각형처럼 넓게
    퍼진 모양이면 axis-aligned bounding box가 화면 전체만큼 커질 수 있음**(위는 좁고
    아래는 화면 폭 전체인 도로 모양이 전형적 사례) — 그러면 SAM이 "박스 전체에서 찾아라"를
    받은 셈이 돼서 벽/천장까지 과다검출함. 비사각형 영역은 박스 대신 점 여러 개나 마스크
    프롬프트를 쓸 것(단, 우리 케이스에선 점을 여러 개 줘도 결국 SAM이 다 묶어서 더 크게
    번지는 것으로 확인됨 — 근본적으로 이 도메인엔 SAM 자체가 안 맞음).
12. **차선 검출 모델의 공개 벤치마크 성능(CULane/TuSimple mAP 등)은 우리 도메인 전이
    성능을 전혀 예측 못 함** — UFLD-v2는 CULane에서 잘 나오는 모델인데 우리 트랙에서
    거의 전멸(20장 중 17장 미검출)했고, YOLOPv2는 상대적으로 덜 알려졌는데 파인튜닝
    없이도 우리 도메인에서 제일 잘 됐음. 벤치마크 숫자로 어떤 모델이 우리한테 맞을지
    미리 예측하지 말고 항상 우리 GT로 직접 검증할 것.
13. **(5회차)** `train.py`류 스크립트에서 `best_metric` 같은 "지금까지 최고 기록" 변수를
    지역변수로만 들고 있으면 `--resume` 재개 시 초기화돼서, 재개 직후 첫 평가가 실제
    값과 무관하게 무조건 "새 기록"으로 찍혀 더 좋았던 체크포인트를 덮어쓸 수 있음 —
    이런 상태값은 반드시 checkpoint에 같이 저장/복원할 것(§2.12).
14. **(5회차) pseudo-label(자동 라벨) 파이프라인은 그걸 만드는 데 쓴 모델의 결함을
    그대로 대물림한다** — pseudo label 생성 스크립트 자체의 전처리가 멀쩡해도, 그
    안에서 추론에 쓰는 모델 가중치가 어떤 버그(이번엔 letterbox 좌표 어긋남) 있는
    상태로 학습된 거라면 그 모델의 예측 편향(이번엔 커브에서 도로 폭 과대평가)이
    고스란히 라벨에 박히고, 그걸로 다음 모델을 학습시키면 편향이 계속 유전됨.
    pseudo label을 새로 만들 때마다 "이걸 만든 모델 자체는 검증됐는가"부터 확인할
    것 — 사람이 그린 소수 GT와 pseudo label 소스 모델의 예측을 IoU로 대조하는 것만으론
    부족하고(§2.12에서 IoU 0.852라는 "괜찮아 보이는" 숫자 뒤에 커브 특화 편향이
    숨어있었음), 카테고리별(커브/직선/실패 등)로 나눠서 편향이 특정 상황에 몰려있지
    않은지 봐야 함.
