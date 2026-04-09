# Medical Segmentation Harness

A Hydra-based medical image segmentation harness that wraps MONAI bundles, nnU-Net v2, and torchxrayvision under a single dict-in / dict-out plugin interface. Designed to plug into [chatclinic-multimodal](../chatclinic-multimodal) as `medseg_harness` but also usable standalone (CLI or Python).

## Supported Tools

| Tool (`tool_name`) | Framework | Input | Dataset | Model |
|---|---|---|---|---|
| `spleen_seg` | MONAI Bundle | 3D CT | MSD Task09_Spleen | `spleen_ct_segmentation` (UNet) |
| `brain_tumor_seg` | MONAI Bundle | 3D MRI (4ch) | MSD Task01_BrainTumour | `brats_mri_segmentation` (SegResNet) |
| `pancreas_tumor_seg` | MONAI Bundle | 3D CT | MSD Task07_Pancreas | `pancreas_ct_dints_segmentation` (DiNTS) |
| `cxr_lung_seg` | torchxrayvision | 2D CXR | Montgomery CXR | `chestx_det` (PSPNet) |
| `nnunet_amos` | nnU-Net v2 | 3D CT | AMOS 2022 (CT) | `Dataset052_AMOS22_OnlyCT` / MaskSAM |

Each tool iterates over **all cases** in its dataset directory, or can be pointed at a single file via the payload.

## Directory Layout

```
.
├── logic.py                # Plugin entrypoint: execute(payload) -> dict
├── run.py                  # Thin CLI wrapper (--input, --output JSON)
├── tool.json               # Plugin manifest (chatclinic format)
├── visualize.py            # 2D/3D visualization (auto-picks best slice)
├── setup_bundles.py        # Download MONAI bundles + MSD datasets + CXR samples
├── download_spleen.py      # Spleen-only dataset download
├── Dockerfile
├── requirements.txt
├── configs/
│   ├── config.yaml         # Hydra defaults (tool, device, paths, seed)
│   ├── tool/*.yaml         # Per-tool settings
│   ├── paths/default.yaml  # datasets_dir, weights_dir, output_dir, ...
│   └── device/{gpu,cpu}.yaml
├── tools/
│   ├── spleen_seg/infer.py
│   ├── brain_tumor_seg/infer.py
│   ├── pancreas_tumor_seg/infer.py
│   ├── cxr_lung_seg/infer.py
│   └── nnunet_amos/
│       ├── infer.py
│       └── vendor/nnunetv2/   # vendored nnU-Net v2
├── weights/                # bundles/, nnunet/, sam/
├── datasets/               # msd/, cxr_samples/, nnunet_raw/
└── results/<tool>/<timestamp>/
```

## Plugin Interface

`logic.execute(payload: dict) -> dict` is the single entrypoint. Payload fields:

| Key | Type | Notes |
|---|---|---|
| `nifti_path` | str | Path to a 3D NIfTI volume (`.nii` / `.nii.gz`). |
| `image_path` | str | Path to a 2D CXR image (`.png` / `.jpg`). |
| `dicom_path` | str | Reserved (DICOM not supported yet). |
| `tool_name` | str | One of the tools above. If omitted, inferred from the file extension. |
| `file_name` | str | Optional display name. |
| `device` | str | `"gpu"` (default) or `"cpu"`. |
| `output_dir` | str | Override for segmentation output directory. |
| `overrides` | list[str] | Extra Hydra-style overrides, e.g. `["tool.sw_batch_size=2"]`. |

Return shape:

```json
{
  "tool": "medseg_harness",
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
    "image_path": ".../spleen_2_preproc.nii.gz",
    "visualization_path": ".../visualization.png"
  },
  "warnings": [],
  "provenance": { "tool_version": "0.1.0", "received_keys": [...], "segmentation_tool": "spleen_seg" }
}
```

## Quickstart (Docker)

### 1. Build

```bash
docker build -t ai619 .
```

### 2. Run the container

Expose ports for chatclinic integration and mount the repo:

```bash
docker run \
--gpus '"device=3"' \
-it \
--ipc=host \
-v $(pwd):/workspace \
--shm-size=32g \
-p 8888:8888 \
--name medseg \
ai619
```

Drop `--gpus all` on CPU-only hosts and pass `device=cpu` in payloads.

### 3. Prepare data and weights

```bash
# Everything
python setup_bundles.py

# Or partial
python setup_bundles.py --bundles-only   # MONAI bundles only
python setup_bundles.py --datasets-only  # MSD Tasks 01/07/09 only
python setup_bundles.py --cxr-only       # 10 CXR samples only
python download_spleen.py                # Spleen only (~1.5 GB)
```

`nnunet_amos` requires manual setup:

- **SAM checkpoint**: `weights/sam/sam_vit_h_4b8939.pth`
- **Trained model**: `weights/nnunet/Dataset052_AMOS22_OnlyCT/MaskSAM_AMOS__nnUNetPlans__3d_fullres/fold_2/checkpoint_final.pth`
- **Input volumes**: `datasets/nnunet_raw/Dataset052_AMOS22_OnlyCT/imagesTs/*_0000.nii.gz`

## Usage

### A) Python (direct)

```python
from logic import execute

result = execute({
    "nifti_path": "/workspace/datasets/msd/Task09_Spleen/imagesTr/spleen_2.nii.gz",
    "tool_name": "spleen_seg",
})
print(result["artifacts"]["mask_path"])
print(result["artifacts"]["visualization_path"])
```

One-liner:

```bash
python -c "from logic import execute; import json; print(json.dumps(execute({'nifti_path':'/workspace/datasets/msd/Task09_Spleen/imagesTr/spleen_2.nii.gz','tool_name':'spleen_seg'}), indent=2, default=str))"
```

### B) CLI (file-based)

```bash
cat > /tmp/payload.json <<'EOF'
{
  "nifti_path": "/workspace/datasets/nnunet_raw/Dataset052_AMOS22_OnlyCT/imagesTs/amos0005_0000.nii.gz",
  "tool_name": "nnunet_amos"
}
EOF

python run.py --input /tmp/payload.json --output /tmp/result.json
cat /tmp/result.json
```

### C) Through chatclinic-multimodal

Place a shim plugin at `chatclinic-multimodal/plugins/medseg_harness/` that re-exports `execute` from this repo (already set up — see `plugins/medseg_harness/logic.py` in that repo). Then any of:

1. **Generic tool endpoint** (works out of the box):
   ```bash
   curl -X POST http://127.0.0.1:8001/api/v1/tools/medseg_harness/run \
     -H "Content-Type: application/json" \
     -d '{"payload":{"nifti_path":"/path/to.nii.gz","tool_name":"spleen_seg"}}'
   ```

2. **Chat `@command`** requires adding an executor entry to `DIRECT_TOOL_ENDPOINT_EXECUTORS` under `"nifti"` in `app/services/chat.py` — not included by default.

## Config Overrides

Anything in `configs/` can be overridden through the payload's `overrides` list (Hydra syntax):

```python
execute({
    "nifti_path": "...",
    "tool_name": "spleen_seg",
    "overrides": [
        "tool.sw_batch_size=2",
        "tool.overlap=0.25",
        "save_output=false",
    ],
})
```

Key paths (`configs/paths/default.yaml`):

| Key | Default |
|---|---|
| `paths.datasets_dir` | `./datasets` (absolute-resolved by logic.py) |
| `paths.weights_dir` | `./weights` (absolute-resolved by logic.py) |
| `paths.bundle_dir` | `${paths.weights_dir}/bundles` |
| `paths.msd_data_dir` | `${paths.datasets_dir}/msd` |
| `paths.nnunet_raw` | `${paths.datasets_dir}/nnunet_raw` |
| `paths.nnunet_results` | `${paths.weights_dir}/nnunet` |
| `paths.sam_checkpoint` | `${paths.weights_dir}/sam/sam_vit_h_4b8939.pth` |
| `paths.output_dir` | `./results/${tool.name}/${now:...}` |

`logic.py` rewrites `paths.datasets_dir` and `paths.weights_dir` to absolute paths at runtime so the plugin works regardless of the caller's cwd.

## Visualization

- 2D tools (`cxr_lung_seg`) → `visualize_2d` overlays mask on the CXR.
- 3D tools → `visualize_3d` picks the **axial slice with the most foreground voxels** automatically (so small organs like the spleen don't land on an empty central slice).
- `_label_to_rgb` uses matplotlib's `tab20` palette, so N-class label maps (e.g. 16-class AMOS) are rendered with distinct colors.
- Disable visualization by passing `"overrides": ["save_output=false"]` in the payload.

## Without Docker (local Python)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e tools/nnunet_amos/vendor  # only if using nnunet_amos
```
