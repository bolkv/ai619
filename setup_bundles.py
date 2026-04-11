"""Download MONAI bundles, MSD test datasets, and CXR sample images.

Each asset is placed inside its tool's folder under ``tools/<task>_seg/``:

    brats_mri_segmentation           -> tools/brain_tumor_seg/weights/
    spleen_ct_segmentation           -> tools/spleen_seg/weights/
    pancreas_ct_dints_segmentation   -> tools/pancreas_tumor_seg/weights/
    Task01_BrainTumour               -> tools/brain_tumor_seg/dataset/
    Task07_Pancreas                  -> tools/pancreas_tumor_seg/dataset/
    Task09_Spleen                    -> tools/spleen_seg/dataset/
    Montgomery CXR samples           -> tools/lung_seg/dataset/
"""

import argparse
import os
import urllib.request


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")

# (bundle_name, tool_folder) pairs. Bundles are extracted into
# tools/<tool>/weights/<bundle_name>/...
_BUNDLE_TO_TOOL = [
    ("brats_mri_segmentation", "brain_tumor_seg"),
    ("spleen_ct_segmentation", "spleen_seg"),
    ("pancreas_ct_dints_segmentation", "pancreas_tumor_seg"),
]

# (msd_task, tool_folder) pairs. MSD tasks are extracted into
# tools/<tool>/dataset/<task>/...
_MSD_TASK_TO_TOOL = [
    ("Task01_BrainTumour", "brain_tumor_seg"),
    ("Task07_Pancreas", "pancreas_tumor_seg"),
    ("Task09_Spleen", "spleen_seg"),
]

CXR_BASE_URL = (
    "https://data.lhncbc.nlm.nih.gov/public/"
    "Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/"
    "MontgomerySet/CXR_png"
)

CXR_SAMPLE_FILES = [
    "MCUCXR_0001_0.png",
    "MCUCXR_0002_0.png",
    "MCUCXR_0003_0.png",
    "MCUCXR_0004_0.png",
    "MCUCXR_0005_0.png",
    "MCUCXR_0006_0.png",
    "MCUCXR_0011_0.png",
    "MCUCXR_0013_0.png",
    "MCUCXR_0015_0.png",
    "MCUCXR_0016_0.png",
]


def download_cxr_samples(count: int = 10) -> None:
    """Download CXR sample images from the NIH Montgomery County CXR Set."""
    data_dir = os.path.join(_TOOLS_DIR, "lung_seg", "dataset")
    os.makedirs(data_dir, exist_ok=True)

    for fname in CXR_SAMPLE_FILES[:count]:
        dest = os.path.join(data_dir, fname)
        if os.path.isfile(dest):
            print(f"  Already exists: {fname}")
            continue
        url = f"{CXR_BASE_URL}/{fname}"
        print(f"  Downloading {fname} ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            print(f"  WARNING: Failed to download {fname}: {exc}")

    downloaded = [f for f in os.listdir(data_dir) if f.endswith(".png")]
    print(f"CXR samples ready: {len(downloaded)} images in {data_dir}")


def download_bundles() -> None:
    from monai.bundle import download

    for bundle_name, tool in _BUNDLE_TO_TOOL:
        bundle_dir = os.path.join(_TOOLS_DIR, tool, "weights")
        os.makedirs(bundle_dir, exist_ok=True)
        print(f"Downloading {bundle_name} -> {bundle_dir}")
        download(name=bundle_name, bundle_dir=bundle_dir)
    print("All bundles downloaded.")


def download_datasets() -> None:
    from monai.apps import DecathlonDataset

    for task, tool in _MSD_TASK_TO_TOOL:
        data_dir = os.path.join(_TOOLS_DIR, tool, "dataset")
        os.makedirs(data_dir, exist_ok=True)
        print(f"Downloading {task} -> {data_dir}")
        DecathlonDataset(
            root_dir=data_dir,
            task=task,
            section="validation",
            download=True,
        )
    print("All datasets downloaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download MONAI bundles, MSD datasets, and CXR samples "
        "into each tool's folder under tools/<task>_seg/."
    )
    parser.add_argument("--bundles-only", action="store_true", help="Download bundles only")
    parser.add_argument("--datasets-only", action="store_true", help="Download datasets only")
    parser.add_argument("--cxr-only", action="store_true", help="Download CXR samples only")
    parser.add_argument("--cxr-count", type=int, default=10, help="Number of CXR samples to download")
    args = parser.parse_args()

    if args.datasets_only:
        download_datasets()
    elif args.bundles_only:
        download_bundles()
    elif args.cxr_only:
        download_cxr_samples(args.cxr_count)
    else:
        download_bundles()
        download_datasets()
        download_cxr_samples(args.cxr_count)
