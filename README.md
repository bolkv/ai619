# Medical Segmentation Harness

A medical image segmentation harness wrapping MONAI bundles, nnU-Net v2, and torchxrayvision as self-contained per-task plugins. Each tool lives under `tools/<task>_seg/` with its own `logic.py` entrypoint, `infer.py` inference code, `tool.json` manifest, `config.yaml` settings, and local `weights/` / `dataset/` / `results/` folders.

## Supported Tools

| Folder | Task | Framework | Input | Dataset | Backing model |
|---|---|---|---|---|---|
| `tools/brain_tumor_seg/` | BraTS brain tumor (ET / TC / WT) | MONAI Bundle | 3D MRI (4ch) | MSD Task01_BrainTumour | `brats_mri_segmentation` (SegResNet) |
| `tools/spleen_seg/` | Spleen | MONAI Bundle | 3D CT | MSD Task09_Spleen | `spleen_ct_segmentation` (UNet) |
| `tools/pancreas_tumor_seg/` | Pancreas + tumor | MONAI Bundle | 3D CT | MSD Task07_Pancreas | `pancreas_ct_dints_segmentation` (DiNTS) |
| `tools/lung_seg/` | CXR lungs | torchxrayvision | 2D CXR | Montgomery CXR | `chestx_det` (PSPNet) |
| `tools/multi_organ_seg/` | AMOS 16-organ | nnU-Net v2 | 3D CT | AMOS 2022 (CT) | `Dataset052_AMOS22_OnlyCT` / MaskSAM |

Each tool iterates over **all cases** in its local `dataset/` directory, or can be pointed at a single file via the payload.

## Directory Layout

```
.
├── setup_bundles.py        # Download MONAI bundles + MSD datasets + CXR samples
├── Dockerfile / docker-entrypoint.sh / requirements.txt
└── tools/
    ├── __init__.py                 # Namespace package marker (no central registry)
    ├── brain_tumor_seg/
    │   ├── logic.py                # Plugin entrypoint: execute(payload) -> dict
    │   ├── infer.py                # Inference code (SegResNet, MONAI bundle)
    │   ├── tool.json               # Per-tool plugin manifest
    │   ├── config.yaml             # Per-tool settings (targets, roi_size, ...)
    │   ├── weights/                # Pretrained weights
    │   ├── dataset/                # Input data
    │   └── results/                # Output masks (timestamped subdirs)
    ├── spleen_seg/                 # same layout — spleen CT
    ├── pancreas_tumor_seg/         # same layout — pancreas + tumor CT
    ├── lung_seg/                   # same layout — CXR lung
    └── multi_organ_seg/
        ├── logic.py, infer.py, tool.json, config.yaml
        ├── weights/, dataset/, results/
        └── vendor/nnunetv2/        # vendored nnU-Net v2 source
```

## Model Weights Placement

Each tool loads its weights from its own `weights/` folder. Run `setup_bundles.py --bundles-only` to fetch the MONAI bundles automatically, or place the files manually at:

### `tools/brain_tumor_seg/` — SegResNet (BraTS)
```
tools/brain_tumor_seg/weights/brats_mri_segmentation/
├── configs/inference.json
└── models/model.pt
```

### `tools/spleen_seg/` — UNet (Spleen)
```
tools/spleen_seg/weights/spleen_ct_segmentation/
├── configs/inference.json
└── models/model.pt
```

### `tools/pancreas_tumor_seg/` — DiNTS (Pancreas + Tumor)
```
tools/pancreas_tumor_seg/weights/pancreas_ct_dints_segmentation/
├── configs/inference.yaml
└── models/model.pt
```

### `tools/lung_seg/` — PSPNet (CXR Lung)
```
tools/lung_seg/weights/cxr/
└── pspnet_chestxray_best_model_4.pth
```
(torchxrayvision uses `weights/cxr/` as its cache directory; the checkpoint is downloaded automatically on first use if missing.)

### `tools/multi_organ_seg/` — MaskSAM / nnU-Net v2 (AMOS CT) — manual
```
tools/multi_organ_seg/weights/
├── nnunet/Dataset052_AMOS22_OnlyCT/MaskSAM_AMOS__nnUNetPlans__3d_fullres/
│   ├── dataset.json
│   ├── plans.json
│   └── fold_2/checkpoint_final.pth
└── sam/sam_vit_h_4b8939.pth
```

## Dataset Placement

### MONAI-bundle tools (expect MSD task subfolders)
```
tools/brain_tumor_seg/dataset/Task01_BrainTumour/{imagesTr,labelsTr}/
tools/spleen_seg/dataset/Task09_Spleen/{imagesTr,labelsTr}/
tools/pancreas_tumor_seg/dataset/Task07_Pancreas/{imagesTr,labelsTr}/
```

### `tools/lung_seg/` — flat directory of CXR images
```
tools/lung_seg/dataset/MCUCXR_0001_0.png
tools/lung_seg/dataset/MCUCXR_0002_0.png
...
```

### `tools/multi_organ_seg/` — nnU-Net raw layout
```
tools/multi_organ_seg/dataset/
├── nnunet_raw/Dataset052_AMOS22_OnlyCT/
│   ├── dataset.json
│   ├── imagesTs/*_0000.nii.gz
│   └── labelsTs/*.nii.gz
└── nnunet_preprocessed/   # populated by nnUNet during training/inference
```

## Plugin Interface

Every tool exposes the same `execute(payload: dict) -> dict` entrypoint as `tools.<task>_seg.logic.execute`. Payload fields:

| Key | Type | Notes |
|---|---|---|
| `nifti_path` | str | Path to a 3D NIfTI volume (`.nii` / `.nii.gz`). |
| `image_path` | str | Path to a 2D CXR image (`.png` / `.jpg`). |
| `dicom_path` | str | Reserved (DICOM not supported yet). |
| `file_name` | str | Optional display name. |
| `device` | str | `"gpu"` (default) or `"cpu"`. |
| `output_dir` | str | Override for segmentation output directory (defaults to `tools/<task>_seg/results/<timestamp>/`). |
| `overrides` | list[str] | Extra `key=value` overrides applied to the tool's cfg before inference (e.g. `["tool.sw_batch_size=2"]`). |

Return shape:

```json
{
  "tool": "spleen_seg",
  "summary": "Spleen CT Segmentation: segmented 1 sample(s) from 'spleen_2.nii.gz' in 12.3s.",
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
  "provenance": { "tool_version": "0.1.0", "received_keys": [...] }
}
```

## Quickstart (Docker)

### 1. Build

```bash
docker build -t ai619 .
```

### 2. Run the container

```bash
docker run \
  --gpus '"device=3"' \
  -it --ipc=host --shm-size=32g \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  --name medseg \
  ai619
```

Drop `--gpus` on CPU-only hosts and pass `"device": "cpu"` in payloads.

### 3. Prepare data and weights

```bash
# Everything — downloads into each tool's weights/ and dataset/ folders
python setup_bundles.py

# Or partial
python setup_bundles.py --bundles-only   # MONAI bundles only
python setup_bundles.py --datasets-only  # MSD Tasks 01/07/09 only
python setup_bundles.py --cxr-only       # 10 CXR samples only
```

`multi_organ_seg` (AMOS / nnU-Net v2) requires manual placement — see **Model Weights Placement** and **Dataset Placement** above.

## Usage

Each tool is an **independent plugin** — import its `execute` directly:

```python
from tools.brain_tumor_seg.logic import execute

result = execute({
    "nifti_path": "/workspace/tools/brain_tumor_seg/dataset/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz",
})
print(result["artifacts"]["mask_path"])
```

From any other host process, each tool's [`tool.json`](tools/brain_tumor_seg/tool.json) declares its entrypoint as `plugins.<task>_seg.logic:execute`, so a plugin host can register and invoke each tool independently — no need to import from `tools/__init__.py`.

## Config Overrides

Pass `key=value` strings in the payload's `overrides` list; they are applied to the tool's cfg (`OmegaConf`) before inference:

```python
execute({
    "nifti_path": "...",
    "overrides": [
        "tool.sw_batch_size=2",
        "tool.overlap=0.25",
    ],
})
```

Each tool's default paths (set by its `logic.py`) point into its own folder:

| Key | Default |
|---|---|
| `paths.tool_dir` | `tools/<task>_seg/` |
| `paths.weights_dir` | `tools/<task>_seg/weights/` |
| `paths.dataset_dir` | `tools/<task>_seg/dataset/` |
| `paths.output_dir` | `tools/<task>_seg/results/<timestamp>/` |

MONAI-bundle tools also expose `paths.bundle_dir` / `paths.msd_data_dir` (both aliased to the local `weights/` and `dataset/` folders). `multi_organ_seg` exposes `paths.nnunet_raw` / `nnunet_preprocessed` / `nnunet_results` / `sam_checkpoint`.

## Without Docker (local Python)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e tools/multi_organ_seg/vendor  # only if using multi_organ_seg
```
