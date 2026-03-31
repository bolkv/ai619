"""
Brain Tumor Segmentation inference module.

Uses MONAI's SegResNet (brats_mri_segmentation bundle) to produce
multi-class tumor segmentation masks from 4-channel BraTS MRI volumes.

BraTS label mapping (handled by ConvertToMultiChannelBasedOnBratsClassesd):
    ET (Enhancing Tumor)  : original label 4
    TC (Tumor Core)       : original labels 1 + 4
    WT (Whole Tumor)      : original labels 1 + 2 + 4
"""

import os
import glob
import time
import logging

import torch
torch.backends.cudnn.enabled = False  # Avoid CUDNN_STATUS_NOT_INITIALIZED on WSL2
import numpy as np
import nibabel as nib
from omegaconf import DictConfig

from monai.networks.nets import SegResNet
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    NormalizeIntensityd,
)

log = logging.getLogger(__name__)


def _find_sample_data(msd_data_dir: str, dataset_task: str) -> dict:
    """Locate a sample BraTS NIfTI volume and its label.

    Handles two common layouts:
    1. Single 4-D file per subject  (e.g., BRATS_001.nii.gz)
    2. Separate per-modality files  (e.g., BRATS_001_t1.nii.gz, ..._flair.nii.gz)

    Returns
    -------
    dict
        {"image": str or list[str], "label": str, "multi_file": bool}
    """
    images_dir = os.path.join(msd_data_dir, dataset_task, "imagesTr")
    labels_dir = os.path.join(msd_data_dir, dataset_task, "labelsTr")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Image directory not found: {images_dir}\n"
            "Please download the MSD BraTS dataset first.\n"
            "  python setup_bundles.py          (to download bundle + data)\n"
            "  Expected layout: {msd_data_dir}/{dataset_task}/imagesTr/"
        )

    nifti_files = sorted(
        glob.glob(os.path.join(images_dir, "*.nii.gz"))
        + glob.glob(os.path.join(images_dir, "*.nii"))
    )
    if not nifti_files:
        raise FileNotFoundError(
            f"No NIfTI files found in {images_dir}. "
            "Please verify the MSD BraTS data was downloaded correctly."
        )

    # Check whether files are per-modality (contain _t1, _t1ce, _t2, _flair suffixes)
    modality_suffixes = ["_t1.", "_t1ce.", "_t2.", "_flair."]
    first_file = nifti_files[0]
    base = os.path.basename(first_file).lower()

    is_multi_file = any(sfx in base for sfx in modality_suffixes)

    if is_multi_file:
        # Group by subject prefix (everything before _t1 / _t1ce / _t2 / _flair)
        subject_prefix = None
        for sfx in modality_suffixes:
            idx = base.find(sfx)
            if idx != -1:
                subject_prefix = os.path.basename(first_file)[:idx]
                break

        modality_order = ["_t1.", "_t1ce.", "_t2.", "_flair."]
        modality_paths = []
        for mod in modality_order:
            pattern = os.path.join(images_dir, f"{subject_prefix}{mod}*")
            matches = glob.glob(pattern)
            # Case-insensitive fallback
            if not matches:
                pattern_lower = os.path.join(images_dir, f"{subject_prefix.lower()}{mod}*")
                matches = glob.glob(pattern_lower)
            if not matches:
                raise FileNotFoundError(
                    f"Could not find modality file matching '{subject_prefix}{mod}*' "
                    f"in {images_dir}"
                )
            modality_paths.append(matches[0])

        label_path = _find_label(labels_dir, subject_prefix)
        return {"image": modality_paths, "label": label_path, "multi_file": True}

    # Single 4-D file per subject
    label_name = os.path.basename(first_file)
    label_path = _find_label(labels_dir, os.path.splitext(os.path.splitext(label_name)[0])[0])
    return {"image": first_file, "label": label_path, "multi_file": False}


def _find_label(labels_dir: str, subject_id: str) -> str:
    """Find the label file for a given subject ID."""
    if not os.path.isdir(labels_dir):
        log.warning("Label directory not found: %s — skipping label loading.", labels_dir)
        return ""

    candidates = (
        glob.glob(os.path.join(labels_dir, f"{subject_id}.nii.gz"))
        + glob.glob(os.path.join(labels_dir, f"{subject_id}.nii"))
        + glob.glob(os.path.join(labels_dir, f"{subject_id}*"))
    )
    if candidates:
        return sorted(candidates)[0]

    log.warning("No label file found for subject '%s' in %s.", subject_id, labels_dir)
    return ""


def _load_weights(model: torch.nn.Module, bundle_dir: str, bundle_name: str) -> None:
    """Load pretrained weights from the downloaded bundle."""
    weights_path = os.path.join(bundle_dir, bundle_name, "models", "model.pt")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            "Please download the MONAI bundle first:\n"
            "  python setup_bundles.py"
        )
    log.info("Loading model weights from: %s", weights_path)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)


def run_inference(cfg: DictConfig) -> dict:
    """Run Brain Tumor Segmentation inference.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration with keys described in the module-level docstring.

    Returns
    -------
    dict
        {"mask_path": str, "num_classes": int, "elapsed_sec": float}
    """
    start_time = time.time()

    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    device = torch.device(cfg.device_name)

    # ------------------------------------------------------------------
    # 1. Build model
    # ------------------------------------------------------------------
    log.info("Creating SegResNet model ...")
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=cfg.tool.input_channels,   # 4
        out_channels=cfg.tool.num_classes,      # 3
        dropout_prob=0.2,
    )

    # ------------------------------------------------------------------
    # 2. Load pretrained weights
    # ------------------------------------------------------------------
    _load_weights(model, cfg.paths.bundle_dir, cfg.tool.bundle_name)
    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 3. Find input data (uploaded file or MSD sample)
    # ------------------------------------------------------------------
    input_image = cfg.paths.get("input_image", None)
    if input_image and os.path.isfile(input_image):
        log.info("Using uploaded image: %s", input_image)
        sample = {"image": input_image, "label": "", "multi_file": False}
    else:
        sample = _find_sample_data(cfg.paths.msd_data_dir, cfg.tool.dataset_task)
        log.info("Using MSD sample image: %s", sample["image"])

    # ------------------------------------------------------------------
    # 4. Build transforms and load data
    # ------------------------------------------------------------------
    if sample["multi_file"]:
        # Stack individual modality files into a single 4-channel volume
        log.info("Loading %d modality files and stacking ...", len(sample["image"]))
        volumes = []
        affine = None
        for mod_path in sample["image"]:
            nii = nib.load(mod_path)
            if affine is None:
                affine = nii.affine
            volumes.append(nii.get_fdata(dtype=np.float32))
        # Stack along new first axis -> (4, H, W, D)
        stacked = np.stack(volumes, axis=0)
        img_tensor = torch.from_numpy(stacked).unsqueeze(0)  # (1, 4, H, W, D)

        # Normalize intensity: per-channel, nonzero only
        for ch in range(img_tensor.shape[1]):
            ch_data = img_tensor[0, ch]
            nonzero_mask = ch_data != 0
            if nonzero_mask.any():
                mean = ch_data[nonzero_mask].mean()
                std = ch_data[nonzero_mask].std()
                if std > 0:
                    img_tensor[0, ch] = torch.where(
                        nonzero_mask,
                        (ch_data - mean) / std,
                        ch_data,
                    )

        # Load label if available
        label_data = None
        if sample["label"] and os.path.isfile(sample["label"]):
            label_nii = nib.load(sample["label"])
            label_data = label_nii.get_fdata(dtype=np.float32)
    else:
        # Single 4-D file — use MONAI transform pipeline
        data_dict = {"image": sample["image"]}
        has_label = sample["label"] and os.path.isfile(sample["label"])
        if has_label:
            data_dict["label"] = sample["label"]

        if has_label:
            transforms = Compose([
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ])
        else:
            transforms = Compose([
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ])

        data_dict = transforms(data_dict)
        img_tensor = data_dict["image"].unsqueeze(0)  # (1, C, H, W, D)
        affine = data_dict["image"].meta.get("affine", np.eye(4)) if hasattr(data_dict["image"], "meta") else np.eye(4)

        if has_label:
            label_data = data_dict["label"]
        else:
            label_data = None

    log.info("Input tensor shape: %s", list(img_tensor.shape))
    img_tensor = img_tensor.to(device)

    # ------------------------------------------------------------------
    # 5. Sliding-window inference
    # ------------------------------------------------------------------
    roi_size = list(cfg.tool.roi_size)
    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=cfg.tool.sw_batch_size,
        overlap=cfg.tool.overlap,
    )

    log.info(
        "Running sliding-window inference (roi=%s, sw_batch=%d, overlap=%.2f) on %s ...",
        roi_size,
        cfg.tool.sw_batch_size,
        cfg.tool.overlap,
        device,
    )

    with torch.no_grad():
        output = inferer(img_tensor, model)  # (1, 3, H, W, D)

    # ------------------------------------------------------------------
    # 6. Post-processing: sigmoid + threshold
    # ------------------------------------------------------------------
    output_sigmoid = torch.sigmoid(output)
    seg_mask = (output_sigmoid > 0.5).float()  # (1, 3, H, W, D)

    seg_np = seg_mask[0].cpu().numpy().astype(np.uint8)  # (3, H, W, D)
    log.info("Segmentation mask shape: %s", list(seg_np.shape))

    for i, target in enumerate(cfg.tool.targets):
        voxel_count = int(seg_np[i].sum())
        log.info("  %s: %d positive voxels", target, voxel_count)

    # ------------------------------------------------------------------
    # 7. Save output as NIfTI
    # ------------------------------------------------------------------
    # Use the original affine if available, otherwise identity
    if affine is None:
        affine = np.eye(4)
    if isinstance(affine, torch.Tensor):
        affine = affine.cpu().numpy()
    affine = np.array(affine, dtype=np.float64)

    # Transpose from (C, H, W, D) to (H, W, D, C) for NIfTI convention
    seg_nifti_data = np.transpose(seg_np, (1, 2, 3, 0))

    mask_filename = "brain_tumor_seg.nii.gz"
    mask_path = os.path.join(cfg.paths.output_dir, mask_filename)
    nifti_img = nib.Nifti1Image(seg_nifti_data, affine)
    nib.save(nifti_img, mask_path)
    log.info("Saved 3-channel segmentation mask to: %s", mask_path)

    elapsed = time.time() - start_time
    log.info("Inference completed in %.2f seconds.", elapsed)

    # Return image/label paths for visualization
    img_path = sample["image"]  # str or list[str]
    lbl_path = sample.get("label", "")

    return {
        "mask_path": mask_path,
        "num_classes": int(cfg.tool.num_classes),
        "elapsed_sec": round(elapsed, 4),
        "image_path": img_path,
        "label_path": lbl_path,
    }
