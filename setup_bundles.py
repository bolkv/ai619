"""Download MONAI bundles, MSD test datasets, and CXR sample images."""

import argparse
import os
import urllib.request


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


def download_cxr_samples(data_dir: str = None, count: int = 10) -> None:
    """Download CXR sample images from NIH Montgomery County CXR Set.

    This is a public TB screening CXR dataset provided by the U.S. National
    Library of Medicine.  Only *count* images are fetched (default 10).
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data", "cxr_samples")
    os.makedirs(data_dir, exist_ok=True)

    samples = CXR_SAMPLE_FILES[:count]
    for fname in samples:
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


def download_bundles(bundle_dir: str = "./bundles") -> None:
    from monai.bundle import download

    os.makedirs(bundle_dir, exist_ok=True)

    bundles = [
        "brats_mri_segmentation",
        "pancreas_ct_dints_segmentation",
    ]
    for name in bundles:
        print(f"Downloading {name}...")
        download(name=name, bundle_dir=bundle_dir)
    print("All bundles downloaded.")


def download_datasets(data_dir: str = "./data/msd") -> None:
    from monai.apps import DecathlonDataset

    os.makedirs(data_dir, exist_ok=True)

    tasks = ["Task01_BrainTumour", "Task07_Pancreas"]
    for task in tasks:
        print(f"Downloading {task}...")
        DecathlonDataset(
            root_dir=data_dir,
            task=task,
            section="validation",
            download=True,
        )
    print("All datasets downloaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MONAI bundles and MSD datasets")
    parser.add_argument("--bundles-only", action="store_true", help="Download bundles only")
    parser.add_argument("--datasets-only", action="store_true", help="Download datasets only")
    parser.add_argument("--cxr-only", action="store_true", help="Download CXR samples only")
    parser.add_argument("--bundle-dir", default="./bundles")
    parser.add_argument("--data-dir", default="./data/msd")
    parser.add_argument("--cxr-dir", default="./data/cxr_samples")
    parser.add_argument("--cxr-count", type=int, default=10, help="Number of CXR samples to download")
    args = parser.parse_args()

    if args.datasets_only:
        download_datasets(args.data_dir)
    elif args.bundles_only:
        download_bundles(args.bundle_dir)
    elif args.cxr_only:
        download_cxr_samples(args.cxr_dir, args.cxr_count)
    else:
        download_bundles(args.bundle_dir)
        download_datasets(args.data_dir)
        download_cxr_samples(args.cxr_dir, args.cxr_count)
