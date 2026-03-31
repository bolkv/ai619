# test_segmentation

ChatClinic 프로젝트용 3개 medical image segmentation 모델의 테스트 환경.

| 모델 | 모달리티 | 아키텍처 | 데이터셋 |
|------|---------|---------|---------|
| CXR Lung Seg | 2D CXR (PNG/JPG) | PSPNet (TorchXRayVision) | Montgomery County CXR |
| Brain Tumor Seg | 3D MRI (4채널) | SegResNet (MONAI Bundle) | MSD Task01 BrainTumour |
| Pancreas Tumor Seg | 3D CT | DiNTS (MONAI Bundle) | MSD Task07 Pancreas |

## 1. 환경 설정 (Anaconda)

```bash
# 가상환경 생성
conda create -n AI619 python=3.11 -y
conda activate AI619

# PyTorch 설치 (CUDA 12.4)
pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# 나머지 의존성
pip install -r requirements.txt
```
- monai 설치 시 PyTorch가 다른 버전으로 덮어씌워질 수 있습니다. 설치 후 PyTorch를 다시 설치하세요.

## 2. 데이터셋 및 모델 다운로드

```bash
conda activate AI619

# 전체 다운로드 (MONAI 번들 + MSD 데이터셋 + CXR 샘플)
python setup_bundles.py

# 개별 다운로드
python setup_bundles.py --bundles-only      # MONAI 번들(모델 가중치)만
python setup_bundles.py --datasets-only     # MSD 데이터셋만 (Brain Tumor, Pancreas)
python setup_bundles.py --cxr-only          # CXR 샘플 이미지만
```

다운로드 후 디렉토리 구조:

```
test_segmentation/
├── bundles/                          # MONAI 번들 (모델 가중치 + config)
│   ├── brats_mri_segmentation/
│   └── pancreas_ct_dints_segmentation/
└── data/
    ├── msd/                          # MSD 데이터셋
    │   ├── Task01_BrainTumour/
    │   └── Task07_Pancreas/
    └── cxr_samples/                  # CXR 테스트 이미지 (PNG)
```

> MSD 데이터셋은 수GB 크기이므로 다운로드에 시간이 걸릴 수 있습니다.

## 3. 모델 실행

모든 실행은 `run.py`를 통해 Hydra config으로 제어합니다.

### CXR Lung Segmentation

2D 흉부 X-ray에서 폐 영역을 세그멘테이션합니다.

```bash
# CPU 실행 (기본)
python run.py tool=cxr_lung_seg device=cpu

# GPU 실행
python run.py tool=cxr_lung_seg device=gpu
```

### Brain Tumor Segmentation

3D MRI (4채널: T1, T1ce, T2, FLAIR)에서 뇌종양을 세그멘테이션합니다.

```bash
# GPU 실행 (권장)
python run.py tool=brain_tumor_seg device=gpu

# CPU 실행 (느림)
python run.py tool=brain_tumor_seg device=cpu
```

### Pancreas Tumor Segmentation

3D CT에서 췌장 및 종양을 세그멘테이션합니다.

```bash
# GPU 실행 (권장, GPU에서도 1-2분 소요)
python run.py tool=pancreas_tumor_seg device=gpu

# CPU 실행 (매우 느림)
python run.py tool=pancreas_tumor_seg device=cpu
```

### 입력 데이터 선택

기본적으로 MSD 데이터셋의 **정렬 기준 첫 번째 파일**을 자동으로 사용합니다.
특정 샘플을 지정하려면 `paths.input_image`를 override하세요:

```bash
# Brain Tumor: 특정 MRI 볼륨 지정
python run.py tool=brain_tumor_seg paths.input_image=./data/msd/Task01_BrainTumour/imagesTr/BRATS_003.nii.gz

# Pancreas: 특정 CT 볼륨 지정
python run.py tool=pancreas_tumor_seg paths.input_image=./data/msd/Task07_Pancreas/imagesTr/pancreas_042.nii.gz

# CXR: 특정 X-ray 이미지 지정
python run.py tool=cxr_lung_seg paths.cxr_image_path=./data/cxr_samples/my_xray.png
```

### Hydra config override 예시

```bash
python run.py tool=brain_tumor_seg tool.roi_size=[128,128,128]
python run.py tool=cxr_lung_seg paths.output_dir=./my_outputs
```

## 4. 출력

결과는 `outputs/{tool_name}/{timestamp}/`에 저장됩니다:

| 모델 | 출력 파일 |
|------|----------|
| CXR Lung Seg | `lung_mask.png` + `visualization.png` |
| Brain Tumor Seg | `brain_tumor_seg.nii.gz` + `visualization.png` |
| Pancreas Tumor Seg | `*_pancreas_seg.nii.gz` + `visualization.png` |

## 프로젝트 구조

```
test_segmentation/
├── configs/
│   ├── config.yaml              # Hydra 기본 config
│   ├── tool/                    # 모델별 config
│   ├── device/                  # gpu.yaml / cpu.yaml
│   └── paths/                   # 경로 설정
├── tools/
│   ├── cxr_lung_seg/infer.py    # TorchXRayVision PSPNet
│   ├── brain_tumor_seg/infer.py # MONAI SegResNet
│   └── pancreas_tumor_seg/infer.py # MONAI DiNTS
├── run.py                       # 통합 실행 진입점
├── setup_bundles.py             # 번들/데이터 다운로드
├── visualize.py                 # 결과 시각화
└── requirements.txt
```

## 주의사항

- BraTS 라벨: 0, 1, 2, 4 (라벨 3 없음). ET=label4, TC=label1+4, WT=label1+2+4
- Pancreas DiNTS는 GPU에서도 1-2분 소요
- MONAI 번들 다운로드 시 수백MB~수GB 네트워크 전송 필요

---

## 부록: 알아두면 좋은 용어와 파일 형식

### 파일 확장자

| 확장자 | 이름 | 설명 |
|--------|------|------|
| `.nii` / `.nii.gz` | **NIfTI** (Neuroimaging Informatics Technology Initiative) | 3D 의료 영상 표준 포맷. CT/MRI 볼륨 데이터를 저장한다. `.gz`는 gzip 압축 버전으로, 용량이 크기 때문에 거의 항상 압축 형태로 사용한다. Python에서는 `nibabel` 라이브러리로 읽는다. |
| `.dcm` | **DICOM** (Digital Imaging and Communications in Medicine) | 병원 장비(CT, MRI, X-ray 등)에서 직접 출력하는 원본 포맷. 환자 정보 + 이미지가 함께 들어있다. 한 환자의 CT 촬영이 수백 개의 `.dcm` 슬라이스로 구성되기도 한다. |
| `.png` / `.jpg` | 일반 이미지 | 2D 의료 영상(CXR, 병리, 안저 사진 등)에서 사용. DICOM에서 변환하거나, 공개 데이터셋에서 바로 제공되기도 한다. |
| `.npy` | NumPy 배열 | Python NumPy 배열을 그대로 저장한 파일. 중간 결과나 예측값을 빠르게 저장/로드할 때 사용한다. |
| `.pt` | PyTorch 모델 가중치 | 학습된 모델의 파라미터(가중치)를 저장한 파일. `torch.load()`로 불러온다. |

### 의료 영상 기본 개념

| 용어 | 설명 |
|------|------|
| **볼륨 (Volume)** | 3D 이미지 데이터. CT/MRI는 2D 슬라이스를 쌓아 3D 볼륨을 구성한다. shape 예시: `(512, 512, 128)` = 가로 512 x 세로 512 x 깊이 128. |
| **복셀 (Voxel)** | 3D 이미지의 최소 단위. 2D의 픽셀(pixel)에 해당하는 3D 버전이다. |
| **Affine 행렬** | NIfTI에 포함된 4x4 변환 행렬. 복셀 좌표 (i, j, k)를 실제 물리적 위치 (mm 단위)로 변환한다. 이게 있어야 영상이 올바른 방향과 스케일로 표시된다. |
| **Spacing / pixdim** | 복셀 간 실제 거리 (mm). 예: `[1.0, 1.0, 1.0]`이면 각 복셀이 1mm 간격. CT/MRI마다 다르므로 전처리 시 통일해준다. |
| **Axial / Sagittal / Coronal** | 3D 볼륨을 자르는 3가지 방향. Axial = 위에서 아래로 (가장 흔한 뷰), Sagittal = 옆에서, Coronal = 앞에서. |
| **ROI (Region of Interest)** | 관심 영역. Sliding Window Inference에서 `roi_size`는 한 번에 모델에 입력하는 3D 패치 크기이다. |
| **Sliding Window Inference** | 3D 볼륨이 GPU 메모리에 한번에 안 들어갈 때, ROI 크기의 패치를 겹치며 이동하면서 추론하고 결과를 합치는 기법. `overlap`은 패치 간 겹침 비율이다. |

### 세그멘테이션 관련 용어

| 용어 | 설명 |
|------|------|
| **세그멘테이션 마스크** | 입력 이미지와 같은 크기의 출력. 각 픽셀/복셀에 클래스 번호(0, 1, 2, ...)가 할당된다. |
| **다채널 마스크 vs 라벨맵** | Brain Tumor는 3채널 이진 마스크 (ET/TC/WT 각각 0 or 1), Pancreas는 단일 라벨맵 (값이 0/1/2). 두 방식 모두 자주 쓰인다. |
| **Dice Score** | 세그멘테이션 성능 지표. 예측 마스크와 정답 마스크의 겹침 정도. 1.0이면 완벽 일치, 0이면 전혀 안 겹침. |
| **MSD (Medical Segmentation Decathlon)** | 10개 의료 세그멘테이션 태스크를 모은 공개 벤치마크 데이터셋. Task01=뇌종양, Task07=췌장 등. |

### BraTS 데이터 구조 (Brain Tumor)

```
Task01_BrainTumour/
├── imagesTr/          # 학습용 이미지 (4D NIfTI: T1, T1ce, T2, FLAIR 4채널)
├── labelsTr/          # 학습용 라벨 (3D NIfTI: 값 0/1/2/4)
└── dataset.json       # 데이터셋 메타정보
```

4개 MRI 모달리티:
- **T1**: 기본 구조 영상
- **T1ce**: 조영제 주입 후 촬영. 종양 경계가 밝게 보임
- **T2**: 수분(부종)이 밝게 보임
- **FLAIR**: T2와 비슷하지만 뇌척수액 신호를 억제

### Pancreas 데이터 구조

```
Task07_Pancreas/
├── imagesTr/          # 학습용 CT (3D NIfTI, 단일 채널)
├── labelsTr/          # 학습용 라벨 (0=배경, 1=췌장, 2=종양)
└── dataset.json
```
