"""
Pancreas + Tumor Segmentation inference module.

Uses MONAI's ``pancreas_ct_dints_segmentation`` bundle (DiNTS architecture
discovered via neural architecture search) to produce a 3-class label map
from abdominal CT volumes:

    0 = Background
    1 = Pancreas
    2 = Tumor

The bundle must already be downloaded to ``cfg.paths.bundle_dir``.
Run ``setup_bundles.py`` first if it is missing.
"""

import glob
import os
import time
import logging

import numpy as np
import torch
torch.backends.cudnn.enabled = False  # Avoid CUDNN_STATUS_NOT_INITIALIZED on WSL2
import nibabel as nib
from omegaconf import DictConfig

from monai.bundle import ConfigParser
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

log = logging.getLogger(__name__)


def _find_sample_data(msd_data_dir: str, dataset_task: str) -> dict:
    """Return the path to the first NIfTI image and its corresponding label.

    Returns
    -------
    dict
        {"image": str, "label": str}
    """
    images_dir = os.path.join(msd_data_dir, dataset_task, "imagesTr")
    labels_dir = os.path.join(msd_data_dir, dataset_task, "labelsTr")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"MSD training images directory not found: {images_dir}\n"
            "Please download the Medical Segmentation Decathlon data for "
            f"{dataset_task} into {msd_data_dir}."
        )

    nifti_files = sorted(
        glob.glob(os.path.join(images_dir, "*.nii.gz"))
        + glob.glob(os.path.join(images_dir, "*.nii"))
    )
    if not nifti_files:
        raise FileNotFoundError(
            f"No NIfTI files found in {images_dir}. "
            "Ensure the MSD dataset has been properly extracted."
        )

    image_path = nifti_files[0]

    # Find corresponding label
    label_path = ""
    if os.path.isdir(labels_dir):
        basename = os.path.basename(image_path)
        label_candidates = (
            glob.glob(os.path.join(labels_dir, basename))
            + glob.glob(os.path.join(labels_dir, os.path.splitext(os.path.splitext(basename)[0])[0] + "*"))
        )
        if label_candidates:
            label_path = sorted(label_candidates)[0]
            log.info("Found GT label: %s", label_path)

    return {"image": image_path, "label": label_path}


def run_inference(cfg: DictConfig) -> dict:
    """Run pancreas + tumor segmentation inference.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.  Expected keys:

        - ``cfg.device_name``
        - ``cfg.tool.bundle_name``
        - ``cfg.tool.dataset_task``
        - ``cfg.tool.roi_size``
        - ``cfg.tool.sw_batch_size``
        - ``cfg.tool.overlap``
        - ``cfg.tool.num_classes``
        - ``cfg.paths.bundle_dir``
        - ``cfg.paths.msd_data_dir``
        - ``cfg.paths.output_dir``

    Returns
    -------
    dict
        ``{"mask_path": str, "num_classes": int, "elapsed_sec": float}``
    """
    start_time = time.time()

    # ------------------------------------------------------------------
    # Validate bundle
    # ------------------------------------------------------------------
    bundle_root = os.path.join(cfg.paths.bundle_dir, cfg.tool.bundle_name)
    if not os.path.isdir(bundle_root):
        raise FileNotFoundError(
            f"Bundle not found at {bundle_root}. "
            "Please run setup_bundles.py to download the "
            f"'{cfg.tool.bundle_name}' bundle first."
        )

    # Try both .json and .yaml — official MONAI bundles may ship either format
    config_file = os.path.join(bundle_root, "configs", "inference.json")
    if not os.path.isfile(config_file):
        config_file = os.path.join(bundle_root, "configs", "inference.yaml")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(
            f"Bundle config not found in {bundle_root}/configs/ "
            "(tried inference.json and inference.yaml). "
            "The bundle may be corrupted — re-run setup_bundles.py."
        )

    weights_path = os.path.join(bundle_root, "models", "model.pt")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}. "
            "The bundle may be incomplete — re-run setup_bundles.py."
        )

    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    device = torch.device(cfg.device_name)

    # ------------------------------------------------------------------
    # Load model via MONAI bundle ConfigParser
    # ------------------------------------------------------------------
    log.info(
        "Loading DiNTS model from bundle: %s", cfg.tool.bundle_name
    )

    # MONAI bundle configs call torch.load internally without weights_only=False.
    # PyTorch >= 2.6 defaults weights_only=True, which breaks numpy-containing checkpoints.
    # Monkey-patch torch.load to allow loading these trusted MONAI checkpoints.
    _original_torch_load = torch.load
    torch.load = lambda *a, **kw: _original_torch_load(*a, **{**kw, "weights_only": kw.get("weights_only", False)})

    # Patch hardcoded CUDA references in the bundle config to respect cfg.device_name
    import tempfile

    with open(config_file) as f:
        config_text = f.read()
    config_text = config_text.replace("torch.device('cuda:0')", f"torch.device('{cfg.device_name}')")
    config_text = config_text.replace("torch.device('cuda')", f"torch.device('{cfg.device_name}')")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(config_text)
        patched_config_file = tmp.name

    parser = ConfigParser()
    parser.read_config(patched_config_file)
    parser["bundle_root"] = bundle_root
    parser["device"] = device
    os.unlink(patched_config_file)

    model = parser.get_parsed_content("network_def")
    model = model.to(device)

    # Load pretrained weights
    log.info("Loading weights from: %s", weights_path)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    log.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Locate input data (uploaded file or MSD sample)
    # ------------------------------------------------------------------
    input_image = cfg.paths.get("input_image", None)
    if input_image and os.path.isfile(input_image):
        sample_path = input_image
        label_path = ""
        log.info("Using uploaded image: %s", sample_path)
    else:
        sample = _find_sample_data(
            cfg.paths.msd_data_dir, cfg.tool.dataset_task
        )
        sample_path = sample["image"]
        label_path = sample["label"]
        log.info("Using MSD sample image: %s", sample_path)

    # ------------------------------------------------------------------
    # Preprocessing pipeline
    # ------------------------------------------------------------------
    has_label = label_path and os.path.isfile(label_path)
    if has_label:
        keys = ["image", "label"]
        transforms = Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=[1.0, 1.0, 1.0],
                mode=["bilinear", "nearest"],
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-87,
                a_max=199,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ])
        data = {"image": sample_path, "label": label_path}
    else:
        transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=[1.0, 1.0, 1.0],
                mode="bilinear",
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-87,
                a_max=199,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ])
        data = {"image": sample_path}
    data = transforms(data)

    # Move image tensor to device: (1, C, H, W, D)
    image_tensor = data["image"].unsqueeze(0).to(device)
    log.info("Input tensor shape: %s", list(image_tensor.shape))

    # ------------------------------------------------------------------
    # Inference (sliding window)
    # ------------------------------------------------------------------
    roi_size = list(cfg.tool.roi_size)
    sw_batch_size = int(cfg.tool.sw_batch_size)
    overlap = float(cfg.tool.overlap)

    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
    )

    log.info(
        "Running sliding-window inference (roi=%s, sw_batch=%d, overlap=%.2f) "
        "on device: %s — this may take 1-2 minutes on GPU ...",
        roi_size,
        sw_batch_size,
        overlap,
        device,
    )
    with torch.no_grad():
        output = inferer(image_tensor, model)  # (1, num_classes, H, W, D)

    # ------------------------------------------------------------------
    # Post-processing: argmax to get label map
    # ------------------------------------------------------------------
    post_transform = AsDiscrete(argmax=True)
    label_map = post_transform(output[0])  # (1, H, W, D)
    label_map_np = label_map.squeeze(0).cpu().numpy().astype(np.uint8)
    log.info(
        "Label map shape: %s, unique values: %s",
        label_map_np.shape,
        np.unique(label_map_np).tolist(),
    )

    # ------------------------------------------------------------------
    # Save output as NIfTI
    # ------------------------------------------------------------------
    # Preserve original affine from the preprocessed image if available,
    # otherwise use identity.
    if hasattr(data["image"], "affine"):
        affine = data["image"].affine
        if isinstance(affine, torch.Tensor):
            affine = affine.cpu().numpy()
    else:
        affine = np.eye(4)

    nifti_img = nib.Nifti1Image(label_map_np, affine)

    sample_basename = os.path.splitext(
        os.path.splitext(os.path.basename(sample_path))[0]
    )[0]
    mask_filename = f"{sample_basename}_pancreas_seg.nii.gz"
    mask_path = os.path.join(cfg.paths.output_dir, mask_filename)

    nib.save(nifti_img, mask_path)
    log.info("Saved segmentation mask to: %s", mask_path)

    # Save preprocessed image for visualization overlay
    # (the mask is in preprocessed space due to Spacing/Orientation transforms)
    preproc_img_np = data["image"].squeeze(0).cpu().numpy() if hasattr(data["image"], "cpu") else np.asarray(data["image"]).squeeze(0)
    preproc_img_path = os.path.join(cfg.paths.output_dir, f"{sample_basename}_preproc.nii.gz")
    nib.save(nib.Nifti1Image(preproc_img_np, affine), preproc_img_path)
    log.info("Saved preprocessed image to: %s", preproc_img_path)

    # Save preprocessed GT label (resampled to match prediction space)
    preproc_label_path = ""
    if has_label:
        preproc_label_np = data["label"].squeeze(0).cpu().numpy() if hasattr(data["label"], "cpu") else np.asarray(data["label"]).squeeze(0)
        preproc_label_path = os.path.join(cfg.paths.output_dir, f"{sample_basename}_gt.nii.gz")
        nib.save(nib.Nifti1Image(preproc_label_np.astype(np.uint8), affine), preproc_label_path)
        log.info("Saved preprocessed GT label to: %s", preproc_label_path)

    elapsed = time.time() - start_time
    log.info("Inference completed in %.2f seconds.", elapsed)

    return {
        "mask_path": mask_path,
        "num_classes": int(cfg.tool.num_classes),
        "elapsed_sec": round(elapsed, 4),
        "image_path": preproc_img_path,
        "label_path": preproc_label_path,
    }
