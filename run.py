"""Hydra-based unified entry point for segmentation tool testing."""

import importlib
import logging
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED on WSL2
torch.backends.cudnn.enabled = False

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(f"Tool: {cfg.tool.display_name}")
    log.info(f"Device: {cfg.device_name}")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Device validation
    if cfg.device_name.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        cfg.device_name = "cpu"

    # Check bundle directory for MONAI-based tools
    if cfg.tool.framework == "monai_bundle":
        import os

        bundle_path = os.path.join(cfg.paths.bundle_dir, cfg.tool.bundle_name)
        if not os.path.isdir(bundle_path):
            log.error(
                f"Bundle not found at {bundle_path}. "
                f"Run 'python setup_bundles.py' first to download bundles."
            )
            return

    # Dynamic import
    module = importlib.import_module(f"tools.{cfg.tool.name}.infer")
    run_inference = module.run_inference

    # Run inference
    log.info("Starting inference...")
    start = time.time()
    result = run_inference(cfg)
    elapsed = time.time() - start

    log.info(f"Inference completed in {elapsed:.1f}s")
    log.info(f"Output: {result.get('mask_path', 'N/A')}")
    log.info(f"Classes: {result.get('num_classes', 'N/A')}")

    # Visualize
    if cfg.save_output:
        from visualize import visualize_2d, visualize_3d
        import os

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
                cfg.tool.targets,
                image_path=image_path,
                gt_path=label_path,
            )

        log.info(f"Visualization saved to {vis_path}")


if __name__ == "__main__":
    main()
