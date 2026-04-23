# Medical Segmentation Harness

A collection of medical image segmentation tools built on MONAI Bundle, nnU-Net v2, and torchxrayvision. Each tool is an independent plugin under `tools/<task>_seg/`, sharing the same `execute(payload) -> dict` interface.

## Directory Layout

```
.
├── setup_bundles.py                    # Download MONAI bundles / MSD datasets / CXR samples
└── tools/
    ├── __init__.py
    ├── brain_tumor_seg/
    │   ├── logic.py                    # Plugin entrypoint: execute(payload) -> dict
    │   ├── infer.py                    # Inference code
    │   ├── tool.json                   # Plugin manifest
    │   ├── config.yaml                 # Tool settings
    │   ├── requirements.txt
    │   ├── weights/                    # Pretrained weights
    │   ├── dataset/                    # Input data
    │   └── results/                    # Output masks
    ├── spleen_seg/                     # Same layout
    ├── pancreas_tumor_seg/             # Same layout
    ├── lung_seg/                       # Same layout
    └── multi_organ_seg/
        ├── logic.py, infer.py, tool.json, config.yaml
        ├── weights/, dataset/, results/
        └── vendor/nnunetv2/            # Vendored nnU-Net v2 source
```

---

## Tools

### 1. Brain Tumor Segmentation (`brain_tumor_seg`)

**Model file**: `tools/brain_tumor_seg/weights/brats_mri_segmentation/models/brain_tumor_seg.pt`

**Description**: Segments brain tumor regions (ET, TC, WT) from 3D MRI using MONAI Bundle's SegResNet.

**Dependencies**: `torch`, `torchvision`, `monai==1.3.2`, `nibabel==5.4.2`, `numpy==1.26.4`, `omegaconf==2.3.0`, `scipy==1.13.1`

**Input**: 4-channel 3D MRI NIfTI file (T1, T1ce, T2, FLAIR) — `.nii` / `.nii.gz`

**Output**: 3-class NIfTI segmentation mask (ET: Enhancing Tumor, TC: Tumor Core, WT: Whole Tumor)

**Limitations**:
- Only supports 4-channel MRI (single-sequence input not accepted)
- Fixed ROI size `[224, 224, 144]` — requires ~8 GB VRAM or more
- Only validated on MSD Task01_BrainTumour format

---

### 2. Spleen Segmentation (`spleen_seg`)

**Model file**: `tools/spleen_seg/weights/spleen_ct_segmentation/models/spleen_seg.pt`

**Description**: Segments the spleen from 3D abdominal CT using MONAI Bundle's UNet.

**Dependencies**: `torch`, `torchvision`, `monai==1.3.2`, `nibabel==5.4.2`, `numpy==1.26.4`, `omegaconf==2.3.0`, `scipy==1.13.1`

**Input**: Single-channel 3D CT NIfTI file — `.nii` / `.nii.gz`

**Output**: 2-class NIfTI segmentation mask (Background, Spleen)

**Limitations**:
- Designed for abdominal CT only (accuracy not guaranteed on chest CT, etc.)
- Fixed ROI size `[96, 96, 96]`

---

### 3. Pancreas + Tumor Segmentation (`pancreas_tumor_seg`)

**Model file**: `tools/pancreas_tumor_seg/weights/pancreas_ct_dints_segmentation/models/pancreas_tumor_seg.pt`

**Description**: Segments the pancreas and tumor from 3D abdominal CT using MONAI Bundle's DiNTS (NAS-based architecture).

**Dependencies**: `torch`, `torchvision`, `monai==1.3.2`, `nibabel==5.4.2`, `numpy==1.26.4`, `omegaconf==2.3.0`, `scipy==1.13.1`

**Input**: Single-channel 3D CT NIfTI file — `.nii` / `.nii.gz`

**Output**: 3-class NIfTI segmentation mask (Background, Pancreas, Tumor)

**Limitations**:
- Designed for abdominal CT only
- Fixed ROI size `[96, 96, 96]`
- Relatively slow inference due to DiNTS architecture

---

### 4. Lung Segmentation (`lung_seg`)

**Model file**: `tools/lung_seg/weights/cxr/pspnet_chestxray_best_model_4.pth`

**Description**: Segments lung regions from 2D chest X-ray (CXR) images using torchxrayvision's PSPNet.

**Dependencies**: `torch`, `torchvision`, `torchxrayvision==1.4.0`, `scikit-image==0.23.2`, `numpy==1.26.4`, `omegaconf==2.3.0`, `scipy==1.13.1`

**Input**: 2D CXR image — `.png` / `.jpg`

**Output**: Binary lung mask PNG (Left Lung + Right Lung extracted from 14-class output, indices [4, 5])

**Limitations**:
- 2D CXR only (not compatible with CT/MRI)
- Input is resized to 512x512
- Weights are auto-downloaded on first run if missing (internet required)

---

### 5. Multi-Organ Segmentation (`multi_organ_seg`)

**Model files**: `tools/multi_organ_seg/weights/nnunet/.../fold_2/multi_organ_seg.pth` + `tools/multi_organ_seg/weights/sam/sam_vit_h_4b8939.pth`

**Description**: Simultaneously segments 16 organs from 3D abdominal CT using nnU-Net v2 + MaskSAM.

**Dependencies**: `torch`, `torchvision`, `nibabel==5.4.2`, `numpy==1.26.4`, `omegaconf==2.3.0`, `scipy==1.13.1`, `scikit-image==0.23.2`, `scikit-learn==1.5.1`, `SimpleITK==2.5.3`, `pandas==2.3.3`, `tqdm==4.67.3`, `batchgenerators==0.25.1`, `dynamic-network-architectures==0.3.1`, `acvl-utils==0.2.5`, `einops`, `fvcore`, and more — see `tools/multi_organ_seg/requirements.txt` for the full list. nnU-Net v2 is vendored under `vendor/nnunetv2/` and does not require a separate pip install.

**Input**: Single-channel 3D CT NIfTI file — `.nii` / `.nii.gz`

**Output**: 16-class NIfTI segmentation mask (Spleen, Right Kidney, Left Kidney, Gallbladder, Esophagus, Liver, Stomach, Aorta, Inferior Vena Cava, Pancreas, Right Adrenal Gland, Left Adrenal Gland, Duodenum, Bladder, Prostate/Uterus)

**Limitations**:
- Very high VRAM usage (SAM ViT-H + nnU-Net loaded simultaneously)
- TTA mirroring is disabled (`use_mirroring: false`) to save memory
- Weights cannot be auto-downloaded — manual placement required
- Only validated on AMOS 2022 CT data

---

## How to Call Each Tool

All tools share the same interface:

```python
from tools.<task>_seg.logic import execute

result = execute(payload)
```

### Payload Fields

| Key | Type | Description |
|---|---|---|
| `nifti_path` | `str` | Path to a 3D NIfTI file (for 3D tools) |
| `image_path` | `str` | Path to a 2D image (lung_seg only) |
| `device` | `str` | `"gpu"` (default) or `"cpu"` |
| `output_dir` | `str` | Output directory (default: `tools/<task>_seg/results/<timestamp>/`) |
| `overrides` | `list[str]` | OmegaConf key=value overrides (e.g. `["tool.sw_batch_size=2"]`) |
| `file_name` | `str` | Display name for results |

### Examples

```python
# Brain Tumor
from tools.brain_tumor_seg.logic import execute
result = execute({"nifti_path": "tools/brain_tumor_seg/dataset/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz"})

# Spleen
from tools.spleen_seg.logic import execute
result = execute({"nifti_path": "tools/spleen_seg/dataset/Task09_Spleen/imagesTr/spleen_2.nii.gz"})

# Pancreas + Tumor
from tools.pancreas_tumor_seg.logic import execute
result = execute({"nifti_path": "tools/pancreas_tumor_seg/dataset/Task07_Pancreas/imagesTr/pancreas_001.nii.gz"})

# Lung (2D CXR)
from tools.lung_seg.logic import execute
result = execute({"image_path": "tools/lung_seg/dataset/MCUCXR_0001_0.png"})

# Multi-Organ
from tools.multi_organ_seg.logic import execute
result = execute({"nifti_path": "tools/multi_organ_seg/dataset/nnunet_raw/Dataset052_AMOS22_OnlyCT/imagesTs/amos_0600_0000.nii.gz"})
```

### Return Shape

```json
{
  "tool": "spleen_seg",
  "summary": "Spleen CT Segmentation: segmented 1 sample(s) in 12.3s.",
  "analysis": {
    "file_name": "spleen_2.nii.gz",
    "segmentation_tool": "spleen_seg",
    "display_name": "Spleen CT Segmentation",
    "num_classes": 2,
    "num_samples": 1,
    "elapsed_sec": 12.34,
    "targets": ["Background", "Spleen"]
  },
  "artifacts": {
    "mask_path": ".../spleen_2_spleen_seg.nii.gz",
    "mask_paths": ["..."],
    "image_path": ".../spleen_2_preproc.nii.gz"
  },
  "warnings": [],
  "provenance": { "tool_version": "0.1.0", "received_keys": ["nifti_path"] }
}
```

### Config Overrides

Pass `key=value` strings in the payload's `overrides` list to override the tool's OmegaConf configuration:

```python
execute({
    "nifti_path": "...",
    "overrides": ["tool.sw_batch_size=2", "tool.overlap=0.25"],
})
```

---

## Model Weights Placement

Each tool loads weights from its own `weights/` folder. MONAI bundle tools can be auto-downloaded via `setup_bundles.py`; others require manual placement.

```bash
# Download all MONAI bundles + MSD datasets + CXR samples
python setup_bundles.py

# Selective download
python setup_bundles.py --bundles-only    # MONAI bundles only
python setup_bundles.py --datasets-only   # MSD datasets only
python setup_bundles.py --cxr-only        # CXR samples only
```

### `brain_tumor_seg` — SegResNet

```
tools/brain_tumor_seg/weights/brats_mri_segmentation/
├── configs/inference.json
└── models/brain_tumor_seg.pt
```

### `spleen_seg` — UNet

```
tools/spleen_seg/weights/spleen_ct_segmentation/
├── configs/inference.json
└── models/spleen_seg.pt
```

### `pancreas_tumor_seg` — DiNTS

```
tools/pancreas_tumor_seg/weights/pancreas_ct_dints_segmentation/
├── configs/inference.yaml
└── models/pancreas_tumor_seg.pt
```

### `lung_seg` — PSPNet

```
tools/lung_seg/weights/cxr/
└── pspnet_chestxray_best_model_4.pth
```

> Weights are auto-downloaded by torchxrayvision on first run if missing.

### `multi_organ_seg` — MaskSAM / nnU-Net v2 (manual placement required)

```
tools/multi_organ_seg/weights/
├── nnunet/Dataset052_AMOS22_OnlyCT/MaskSAM_AMOS__nnUNetPlans__3d_fullres/
│   ├── dataset.json
│   ├── plans.json
│   └── fold_2/multi_organ_seg.pth
└── sam/sam_vit_h_4b8939.pth
```

---

## Dataset Placement

### MONAI Bundle tools (MSD task format)

```
tools/brain_tumor_seg/dataset/Task01_BrainTumour/{imagesTr,labelsTr}/
tools/spleen_seg/dataset/Task09_Spleen/{imagesTr,labelsTr}/
tools/pancreas_tumor_seg/dataset/Task07_Pancreas/{imagesTr,labelsTr}/
```

### `lung_seg` — CXR images (flat directory)

```
tools/lung_seg/dataset/MCUCXR_0001_0.png
tools/lung_seg/dataset/MCUCXR_0002_0.png
...
```

### `multi_organ_seg` — nnU-Net raw format

```
tools/multi_organ_seg/dataset/
├── nnunet_raw/Dataset052_AMOS22_OnlyCT/
│   ├── dataset.json
│   ├── imagesTs/*_0000.nii.gz
│   └── labelsTs/*.nii.gz
└── nnunet_preprocessed/   # Auto-generated by nnUNet during inference
```

---

## Installation

Each tool has its own `requirements.txt`. Install only the tools you need:

```bash
python -m venv .venv && source .venv/bin/activate

# PyTorch (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies for the tools you need
pip install -r tools/brain_tumor_seg/requirements.txt
pip install -r tools/spleen_seg/requirements.txt
pip install -r tools/pancreas_tumor_seg/requirements.txt
pip install -r tools/lung_seg/requirements.txt
pip install -r tools/multi_organ_seg/requirements.txt
```

The nnU-Net v2 source for `multi_organ_seg` is vendored under `tools/multi_organ_seg/vendor/`. `infer.py` adds it to `sys.path` automatically at import time, so no separate installation is needed.
