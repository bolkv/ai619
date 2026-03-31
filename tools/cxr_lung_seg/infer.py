"""
CXR Lung Segmentation inference module.

Uses TorchXRayVision's PSPNet (chestx_det) to produce lung segmentation masks
from chest X-ray images.
"""

import os
import time
import logging

import numpy as np
import torch
torch.backends.cudnn.enabled = False  # Avoid CUDNN_STATUS_NOT_INITIALIZED on WSL2
import torchvision
import torchxrayvision as xrv
from skimage import io as skimage_io
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def run_inference(cfg: DictConfig) -> dict:
    """Run CXR lung segmentation inference.

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

    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------
    input_path = cfg.paths.cxr_image_path
    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Input CXR image not found: {input_path}. "
            "Please verify cfg.paths.cxr_image_path points to a valid PNG/JPG file."
        )

    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    device = torch.device(cfg.device_name)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    log.info("Loading PSPNet model (chestx_det) ...")
    model = xrv.baseline_models.chestx_det.PSPNet()
    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Preprocessing pipeline
    # ------------------------------------------------------------------
    log.info("Preprocessing image: %s", input_path)

    # 1. Load image
    img = skimage_io.imread(input_path)

    # 2. Normalize [0, 255] -> [-1024, 1024]
    img = xrv.datasets.normalize(img, 255)

    # 3. Convert RGB to grayscale if needed
    if img.ndim == 3 and img.shape[2] == 3:
        img = img.mean(2)

    # 4. Add channel dimension -> (1, H, W)
    img = img[None, ...]

    # 5. Center-crop and resize
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(cfg.tool.input_size),
    ])
    img = transform(img)

    # 6. Convert to tensor -> (1, 1, input_size, input_size)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    log.info("Running inference on device: %s", device)
    with torch.no_grad():
        output = model(img_tensor)  # [1, 14, 512, 512]

    # ------------------------------------------------------------------
    # Extract lung masks
    # ------------------------------------------------------------------
    lung_indices = list(cfg.tool.lung_indices)  # e.g. [4, 5]
    lung_masks = output[0, lung_indices, :, :]  # [2, 512, 512]

    # Combine left and right lung masks (logical OR after thresholding)
    combined_mask = (lung_masks.max(dim=0).values > 0.5).cpu().numpy().astype(np.uint8)
    combined_mask *= 255  # scale for PNG visibility

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    mask_filename = "lung_mask.png"
    mask_path = os.path.join(cfg.paths.output_dir, mask_filename)
    skimage_io.imsave(mask_path, combined_mask, check_contrast=False)
    log.info("Saved combined lung mask to: %s", mask_path)

    # Optionally save full 14-class prediction
    if cfg.get("save_output", False):
        full_pred = (output[0].cpu().numpy() > 0.5).astype(np.uint8)  # [14, 512, 512]
        full_pred_path = os.path.join(cfg.paths.output_dir, "full_prediction.npy")
        np.save(full_pred_path, full_pred)
        log.info("Saved full 14-class prediction to: %s", full_pred_path)

    elapsed = time.time() - start_time
    log.info("Inference completed in %.2f seconds.", elapsed)

    return {
        "mask_path": mask_path,
        "num_classes": int(cfg.tool.num_classes),
        "elapsed_sec": round(elapsed, 4),
        "image_path": input_path,
        "label_path": "",
    }
