"""Visualization utilities for segmentation results."""

import os
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helper: BraTS label map -> multi-channel (ET, TC, WT)
# ---------------------------------------------------------------------------
def _brats_label_to_multichannel(label_data: np.ndarray) -> np.ndarray:
    """Convert BraTS integer label map (0,1,2,4) to 3-channel binary mask.

    Returns (3, H, W, D) with channels ET, TC, WT.
    """
    et = (label_data == 4).astype(np.float32)
    tc = np.isin(label_data, [1, 4]).astype(np.float32)
    wt = np.isin(label_data, [1, 2, 4]).astype(np.float32)
    return np.stack([et, tc, wt], axis=0)


# ---------------------------------------------------------------------------
# Helper: build TP/FP/FN error map
# ---------------------------------------------------------------------------
def _error_map(pred: np.ndarray, gt: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Create an RGB error overlay on a grayscale background.

    Green = TP, Red = FP, Blue = FN.
    """
    bg_norm = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
    rgb = np.stack([bg_norm] * 3, axis=-1).astype(np.float32)

    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt

    alpha = 0.6
    rgb[tp] = rgb[tp] * (1 - alpha) + np.array([0, 1, 0]) * alpha  # green
    rgb[fp] = rgb[fp] * (1 - alpha) + np.array([1, 0, 0]) * alpha  # red
    rgb[fn] = rgb[fn] * (1 - alpha) + np.array([0, 0, 1]) * alpha  # blue

    return np.clip(rgb, 0, 1)


# ---------------------------------------------------------------------------
# 2D visualization
# ---------------------------------------------------------------------------
def visualize_2d(
    image_path: str,
    mask_path: str,
    output_path: str,
    alpha: float = 0.4,
    gt_path: Optional[str] = None,
) -> None:
    """Overlay 2D segmentation mask on original CXR image, optionally with GT comparison.

    Args:
        image_path: Path to the original CXR image (PNG/JPG).
        mask_path: Path to the segmentation mask.
        output_path: Path to save the visualization PNG.
        alpha: Transparency of the mask overlay.
        gt_path: Optional path to ground-truth mask for comparison.
    """
    from skimage import io as skio

    img = skio.imread(image_path)
    if img.ndim == 3:
        img = img.mean(axis=2)

    mask = skio.imread(mask_path)

    has_gt = gt_path is not None and os.path.isfile(gt_path)

    if has_gt:
        gt = skio.imread(gt_path)
        ncols = 5
    else:
        ncols = 3

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))

    # Original
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Prediction mask
    axes[1].imshow(mask, cmap="jet")
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    # Prediction overlay
    axes[2].imshow(img, cmap="gray")
    axes[2].imshow(mask, cmap="jet", alpha=alpha)
    axes[2].set_title("Pred Overlay")
    axes[2].axis("off")

    if has_gt:
        # GT overlay
        axes[3].imshow(img, cmap="gray")
        axes[3].imshow(gt, cmap="jet", alpha=alpha)
        axes[3].set_title("GT Overlay")
        axes[3].axis("off")

        # Pred vs GT error map
        pred_bin = mask > 127
        gt_bin = gt > 127
        err = _error_map(pred_bin, gt_bin, img.astype(np.float32))
        axes[4].imshow(err)
        axes[4].set_title("Pred vs GT\n(G=TP  R=FP  B=FN)")
        axes[4].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 3D visualization
# ---------------------------------------------------------------------------
def visualize_3d(
    mask_path: str,
    output_path: str,
    class_names: Optional[List[str]] = None,
    slice_idx: Optional[int] = None,
    image_path: Optional[Union[str, List[str]]] = None,
    gt_path: Optional[str] = None,
) -> None:
    """Visualize a central axial slice of a 3D segmentation mask.

    Shows original image overlay and optional GT comparison.

    Args:
        mask_path: Path to the NIfTI segmentation mask.
        output_path: Path to save the visualization PNG.
        class_names: List of class/channel names for the legend.
        slice_idx: Axial slice index. If None, uses the central slice.
        image_path: Path to the original NIfTI image (str or list of str for multi-modality).
        gt_path: Path to ground-truth NIfTI label for comparison.
    """
    import nibabel as nib

    nii = nib.load(mask_path)
    pred_data = np.asarray(nii.dataobj)

    # ------------------------------------------------------------------
    # Load original image (if provided)
    # ------------------------------------------------------------------
    orig_data = None
    if image_path:
        if isinstance(image_path, (list, tuple)):
            # Multi-modality (e.g., BraTS): use FLAIR (index 3) or last available
            display_idx = min(3, len(image_path) - 1)
            orig_nii = nib.load(image_path[display_idx])
            orig_data = np.asarray(orig_nii.dataobj, dtype=np.float32)
        else:
            orig_nii = nib.load(image_path)
            orig_data = np.asarray(orig_nii.dataobj, dtype=np.float32)
            if orig_data.ndim == 4:
                # 4D file: pick FLAIR channel (index 3) or last
                ch = min(3, orig_data.shape[3] - 1)
                orig_data = orig_data[..., ch]

    # ------------------------------------------------------------------
    # Load GT (if provided)
    # ------------------------------------------------------------------
    gt_data_raw = None
    if gt_path and os.path.isfile(gt_path):
        gt_nii = nib.load(gt_path)
        gt_data_raw = np.asarray(gt_nii.dataobj, dtype=np.float32)

    # ------------------------------------------------------------------
    # Determine mask type and process
    # ------------------------------------------------------------------
    is_multichannel = pred_data.ndim == 4

    # Auto-pick the axial slice with the most foreground so small organs
    # (e.g. spleen) don't land on an empty central slice.
    if slice_idx is None:
        if is_multichannel:
            # pred_data: (H, W, D, C) -> sum foreground over H, W, C per slice
            fg_per_slice = (pred_data > 0).sum(axis=(0, 1, 3))
        else:
            fg_per_slice = (pred_data > 0).sum(axis=(0, 1))
        if fg_per_slice.max() > 0:
            slice_idx = int(fg_per_slice.argmax())

    if is_multichannel:
        _visualize_3d_multichannel(
            pred_data, orig_data, gt_data_raw,
            class_names, slice_idx, output_path,
        )
    else:
        _visualize_3d_labelmap(
            pred_data, orig_data, gt_data_raw,
            class_names, slice_idx, output_path,
        )


def _visualize_3d_multichannel(
    pred_data: np.ndarray,
    orig_data: Optional[np.ndarray],
    gt_data_raw: Optional[np.ndarray],
    class_names: Optional[List[str]],
    slice_idx: Optional[int],
    output_path: str,
) -> None:
    """Visualize multi-channel prediction (e.g., brain tumor ET/TC/WT)."""

    # pred_data: (H, W, D, C) -> (C, H, W, D)
    pred = np.transpose(pred_data, (3, 0, 1, 2))
    num_classes = pred.shape[0]
    depth = pred.shape[3]
    if slice_idx is None:
        slice_idx = depth // 2

    # Convert GT if available (BraTS label map -> multi-channel)
    gt = None
    if gt_data_raw is not None:
        if gt_data_raw.ndim == 3:
            gt = _brats_label_to_multichannel(gt_data_raw)
            # GT might have different depth; use matching slice
            gt_depth = gt.shape[3]
            gt_slice_idx = min(slice_idx, gt_depth - 1)
        elif gt_data_raw.ndim == 4:
            gt = np.transpose(gt_data_raw, (3, 0, 1, 2))
            gt_slice_idx = min(slice_idx, gt.shape[3] - 1)

    has_orig = orig_data is not None
    has_gt = gt is not None

    colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
    ]

    # --- Determine layout ---
    # Row 1: Original | Pred Overlay | GT Overlay | Error Map
    # Row 2: Per-class Pred | Per-class GT
    if has_gt:
        nrows = 2
        top_cols = 2 + (1 if has_orig else 0) + 1  # orig + pred_overlay + gt_overlay + error
        bot_cols = num_classes * 2  # pred + gt per class
        ncols = max(top_cols, bot_cols)
    else:
        nrows = 2
        top_cols = 1 + (1 if has_orig else 0)  # orig + pred_overlay
        bot_cols = num_classes
        ncols = max(top_cols, bot_cols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    # Hide all axes first
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c].axis("off")

    col = 0

    # --- Row 1: Original ---
    if has_orig:
        orig_slice = orig_data[:, :, min(slice_idx, orig_data.shape[2] - 1)]
        axes[0, col].imshow(orig_slice, cmap="gray")
        axes[0, col].set_title("Original")
        col += 1

    # --- Row 1: Prediction overlay ---
    pred_combined = np.zeros((*pred.shape[1:3], 3))
    for i in range(min(num_classes, len(colors))):
        mask_slice = pred[i, :, :, slice_idx]
        for c in range(3):
            pred_combined[:, :, c] += mask_slice * colors[i][c]
    pred_combined = np.clip(pred_combined, 0, 1)

    if has_orig:
        bg = orig_slice.copy()
        bg_norm = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
        bg_rgb = np.stack([bg_norm] * 3, axis=-1)
        overlay = bg_rgb * 0.6 + pred_combined * 0.4
        axes[0, col].imshow(np.clip(overlay, 0, 1))
    else:
        axes[0, col].imshow(pred_combined)
    axes[0, col].set_title("Pred Overlay")
    col += 1

    # --- Row 1: GT overlay ---
    if has_gt:
        gt_combined = np.zeros((*gt.shape[1:3], 3))
        for i in range(min(num_classes, len(colors))):
            gt_mask_slice = gt[i, :, :, gt_slice_idx]
            for c in range(3):
                gt_combined[:, :, c] += gt_mask_slice * colors[i][c]
        gt_combined = np.clip(gt_combined, 0, 1)

        if has_orig:
            overlay_gt = bg_rgb * 0.6 + gt_combined * 0.4
            axes[0, col].imshow(np.clip(overlay_gt, 0, 1))
        else:
            axes[0, col].imshow(gt_combined)
        axes[0, col].set_title("GT Overlay")
        col += 1

        # --- Row 1: Error map (combined) ---
        pred_any = pred[:, :, :, slice_idx].max(axis=0) > 0.5
        gt_any = gt[:, :, :, gt_slice_idx].max(axis=0) > 0.5
        if has_orig:
            err = _error_map(pred_any, gt_any, orig_slice)
        else:
            bg_blank = np.zeros(pred_any.shape, dtype=np.float32)
            err = _error_map(pred_any, gt_any, bg_blank)
        axes[0, col].imshow(err)
        axes[0, col].set_title("Pred vs GT\n(G=TP  R=FP  B=FN)")
        col += 1

    # --- Row 2: Per-class breakdown ---
    for i in range(num_classes):
        name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"

        # Pred per-class
        pred_slice = pred[i, :, :, slice_idx]
        if has_gt:
            ci = i * 2
        else:
            ci = i
        axes[1, ci].imshow(pred_slice, cmap="gray")
        axes[1, ci].set_title(f"Pred {name}")

        # GT per-class
        if has_gt:
            gt_slice = gt[i, :, :, gt_slice_idx]
            axes[1, ci + 1].imshow(gt_slice, cmap="gray")
            axes[1, ci + 1].set_title(f"GT {name}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _visualize_3d_labelmap(
    pred_data: np.ndarray,
    orig_data: Optional[np.ndarray],
    gt_data_raw: Optional[np.ndarray],
    class_names: Optional[List[str]],
    slice_idx: Optional[int],
    output_path: str,
) -> None:
    """Visualize single-channel label map prediction (e.g., pancreas)."""

    depth = pred_data.shape[2]
    if slice_idx is None:
        slice_idx = depth // 2

    pred_slice = pred_data[:, :, slice_idx]
    num_labels = int(pred_slice.max())

    has_orig = orig_data is not None
    has_gt = gt_data_raw is not None

    # Build a label->RGB colormap that scales to arbitrary class counts.
    # Label 0 (background) is always transparent/black; foreground labels
    # cycle through matplotlib's tab20 palette.
    _tab20 = plt.get_cmap("tab20").colors  # 20 distinct RGB tuples

    def _label_to_rgb(label_slice: np.ndarray) -> np.ndarray:
        rgb = np.zeros((*label_slice.shape, 3))
        max_lbl = int(label_slice.max())
        for lbl in range(1, max_lbl + 1):
            color = _tab20[(lbl - 1) % len(_tab20)]
            mask = label_slice == lbl
            for c in range(3):
                rgb[:, :, c] += mask * color[c]
        return np.clip(rgb, 0, 1)

    if has_gt:
        gt_slice_idx = min(slice_idx, gt_data_raw.shape[2] - 1)
        gt_slice = gt_data_raw[:, :, gt_slice_idx]
        ncols = 3 + (1 if has_orig else 0) + 1  # orig + pred_overlay + gt_overlay + pred_label + error
    else:
        ncols = 1 + (1 if has_orig else 0) + 1  # orig + pred_overlay + pred_label

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    col = 0

    # Original
    if has_orig:
        orig_slice = orig_data[:, :, min(slice_idx, orig_data.shape[2] - 1)]
        axes[col].imshow(orig_slice, cmap="gray")
        axes[col].set_title("Original")
        axes[col].axis("off")
        col += 1

    # Pred overlay on original (or standalone color label map)
    pred_rgb = _label_to_rgb(pred_slice)
    if has_orig:
        bg_norm = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min() + 1e-8)
        bg_rgb = np.stack([bg_norm] * 3, axis=-1).astype(np.float32)
        # Only overlay non-background
        fg_mask = pred_slice > 0
        overlay = bg_rgb.copy()
        overlay[fg_mask] = bg_rgb[fg_mask] * 0.5 + pred_rgb[fg_mask] * 0.5
        axes[col].imshow(np.clip(overlay, 0, 1))
    else:
        axes[col].imshow(pred_rgb)
    axes[col].set_title("Pred Overlay")
    axes[col].axis("off")
    col += 1

    # GT overlay on original
    if has_gt:
        gt_rgb = _label_to_rgb(gt_slice)
        if has_orig:
            fg_mask_gt = gt_slice > 0
            overlay_gt = bg_rgb.copy()
            overlay_gt[fg_mask_gt] = bg_rgb[fg_mask_gt] * 0.5 + gt_rgb[fg_mask_gt] * 0.5
            axes[col].imshow(np.clip(overlay_gt, 0, 1))
        else:
            axes[col].imshow(gt_rgb)
        axes[col].set_title("GT Overlay")
        axes[col].axis("off")
        col += 1

    # Label map (nipy_spectral)
    axes[col].imshow(pred_slice, cmap="nipy_spectral", interpolation="nearest")
    axes[col].set_title(f"Pred Label Map (slice {slice_idx})")
    axes[col].axis("off")
    col += 1

    # Error map
    if has_gt:
        pred_fg = pred_slice > 0
        gt_fg = gt_slice > 0
        if has_orig:
            err = _error_map(pred_fg, gt_fg, orig_slice)
        else:
            err = _error_map(pred_fg, gt_fg, np.zeros_like(pred_slice, dtype=np.float32))
        axes[col].imshow(err)
        axes[col].set_title("Pred vs GT\n(G=TP  R=FP  B=FN)")
        axes[col].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
