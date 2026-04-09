"""Plugin entrypoint for the medical-segmentation harness.

Invoked by chatclinic-multimodal's tool runner as
``plugins.<this_dir>.logic:execute`` with a payload dict. The plugin wraps
the Hydra-based segmentation harness (MONAI bundles / nnU-Net / torchxrayvision)
so a single NIfTI (or CXR image) can be segmented from a dict-in / dict-out
interface, matching the nifti/dicom/image review plugin conventions.

Expected payload fields
-----------------------
nifti_path      (str, preferred) Path to a 3D NIfTI volume (.nii / .nii.gz).
image_path      (str)            Path to a 2D CXR image (.png / .jpg).
dicom_path      (str)            Reserved — DICOM not supported yet.
file_name       (str, optional)  Display name; defaults to basename of the path.
tool_name       (str, optional)  Force a specific segmentation tool
                                 (one of tools.AVAILABLE_TOOLS). If omitted,
                                 the tool is inferred from the file type.
device          (str, optional)  "gpu" (default) or "cpu".
output_dir      (str, optional)  Override for segmentation output directory.
overrides       (list, optional) Extra Hydra-style overrides, e.g.
                                 ["tool.sw_batch_size=2", "tool.overlap=0.25"].

Return shape (matches other imaging plugins)
--------------------------------------------
{
    "tool":     "<plugin tool name>",
    "summary":  "<human-readable summary>",
    "analysis": {
        "segmentation_tool": str,
        "num_classes": int,
        "num_samples": int,
        "elapsed_sec": float,
        "targets": list[str],
    },
    "artifacts": {
        "mask_path":    str,        # last / primary mask
        "mask_paths":   list[str],  # per-case masks
        "image_path":   str,
        "label_path":   str,        # empty if no GT
    },
    "warnings":   list[str],
    "provenance": {"tool_version": ..., "received_keys": [...]},
}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from tools import AVAILABLE_TOOLS, get_tool

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.join(_THIS_DIR, "configs")
_PLUGIN_NAME = "medseg_harness"
_TOOL_VERSION = "0.1.0"

# Default segmentation tool for each input modality.
_DEFAULT_TOOL_FOR_SUFFIX = {
    ".nii": "spleen_seg",
    ".nii.gz": "spleen_seg",
    ".png": "cxr_lung_seg",
    ".jpg": "cxr_lung_seg",
    ".jpeg": "cxr_lung_seg",
}


def _resolve_input_path(payload: Dict[str, Any]) -> str:
    for key in ("nifti_path", "image_path", "dicom_path", "input_path"):
        raw = payload.get(key)
        if raw:
            return str(raw).strip()
    raise ValueError(
        "Payload must include one of: nifti_path, image_path, dicom_path."
    )


def _infer_tool_name(path: str, explicit: str | None) -> str:
    if explicit:
        if explicit not in AVAILABLE_TOOLS:
            raise ValueError(
                f"Unknown tool '{explicit}'. Available: {AVAILABLE_TOOLS}"
            )
        return explicit

    lower = path.lower()
    for suffix, tool in _DEFAULT_TOOL_FOR_SUFFIX.items():
        if lower.endswith(suffix):
            return tool
    raise ValueError(
        f"Cannot infer segmentation tool from path '{path}'. "
        "Pass 'tool_name' explicitly."
    )


def _build_cfg(
    tool_name: str,
    input_path: str,
    payload: Dict[str, Any],
):
    overrides: List[str] = [f"tool={tool_name}"]

    # Anchor the two path roots to this plugin directory so the harness
    # works regardless of the caller's cwd. All derived paths
    # (bundle_dir, msd_data_dir, nnunet_*, sam_checkpoint, ...) interpolate
    # from these two via configs/paths/default.yaml.
    overrides.extend(
        [
            f"paths.datasets_dir={os.path.join(_THIS_DIR, 'datasets')}",
            f"paths.weights_dir={os.path.join(_THIS_DIR, 'weights')}",
            # output_dir is defined as ./results/... (not derived from the roots)
            # so it still needs an explicit override unless overridden by payload.
            f"paths.output_dir={os.path.join(_THIS_DIR, 'results', tool_name, '${now:%Y-%m-%d_%H-%M-%S}')}",
        ]
    )

    device = payload.get("device")
    if device:
        overrides.append(f"device={device}")

    # Route the input path to the field each tool actually reads.
    if tool_name == "cxr_lung_seg":
        overrides.append(f"paths.cxr_image_path={input_path}")
    elif tool_name == "nnunet_amos":
        overrides.append(f"+tool.input_image={input_path}")
    else:
        # MONAI bundle tools (spleen/brain/pancreas) read paths.input_image.
        overrides.append(f"+paths.input_image={input_path}")

    output_dir = payload.get("output_dir")
    if output_dir:
        overrides.append(f"paths.output_dir={output_dir}")

    extra = payload.get("overrides") or []
    overrides.extend(str(x) for x in extra)

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def execute(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Plugin entrypoint — runs one segmentation pass on the supplied file."""

    warnings: List[str] = []

    input_path = _resolve_input_path(payload)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    file_name = str(payload.get("file_name") or Path(input_path).name)
    tool_name = _infer_tool_name(input_path, payload.get("tool_name"))

    cfg = _build_cfg(tool_name, input_path, payload)
    run_inference = get_tool(tool_name)
    result = run_inference(cfg)

    # Optional visualization — mirrors run.py's old Hydra main.
    vis_path = ""
    if bool(cfg.get("save_output", True)) and result.get("mask_path"):
        try:
            from visualize import visualize_2d, visualize_3d

            vis_path = os.path.join(cfg.paths.output_dir, "visualization.png")
            os.makedirs(cfg.paths.output_dir, exist_ok=True)
            image_path = result.get("image_path", None)
            label_path = result.get("label_path", "") or None

            if cfg.tool.input_type == "2d_image":
                visualize_2d(
                    cfg.paths.cxr_image_path,
                    result["mask_path"],
                    vis_path,
                    gt_path=label_path,
                )
            else:
                visualize_3d(
                    result["mask_path"],
                    vis_path,
                    list(cfg.tool.targets),
                    image_path=image_path,
                    gt_path=label_path,
                )
        except Exception as exc:  # visualization is best-effort
            warnings.append(f"visualization failed: {exc}")
            vis_path = ""

    num_samples = int(result.get("num_samples", 1))
    elapsed = float(result.get("elapsed_sec", 0.0))

    analysis = {
        "file_name": file_name,
        "segmentation_tool": tool_name,
        "display_name": str(cfg.tool.display_name),
        "num_classes": int(result.get("num_classes", cfg.tool.num_classes)),
        "num_samples": num_samples,
        "elapsed_sec": elapsed,
        "targets": list(cfg.tool.targets) if "targets" in cfg.tool else [],
    }

    artifacts: Dict[str, Any] = {
        "mask_path": result.get("mask_path", ""),
    }
    if "mask_paths" in result:
        artifacts["mask_paths"] = list(result["mask_paths"])
    if result.get("image_path"):
        artifacts["image_path"] = result["image_path"]
    if result.get("label_path"):
        artifacts["label_path"] = result["label_path"]
    if vis_path:
        artifacts["visualization_path"] = vis_path

    summary = (
        f"{cfg.tool.display_name}: segmented {num_samples} sample(s) from "
        f"'{file_name}' in {elapsed:.1f}s."
    )

    return {
        "tool": _PLUGIN_NAME,
        "summary": summary,
        "analysis": analysis,
        "artifacts": artifacts,
        "warnings": warnings,
        "provenance": {
            "tool_version": _TOOL_VERSION,
            "received_keys": sorted(payload.keys()),
            "segmentation_tool": tool_name,
        },
    }


# Backwards-compatible alias so ``run(payload)`` still works for CLI/tests.
run = execute
