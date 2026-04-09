"""Download only the MSD Task09_Spleen dataset into ./datasets/msd."""

import os

from monai.apps import DecathlonDataset


def main(data_dir: str = "./datasets/msd") -> None:
    os.makedirs(data_dir, exist_ok=True)
    print("Downloading Task09_Spleen...")
    DecathlonDataset(
        root_dir=data_dir,
        task="Task09_Spleen",
        section="validation",
        download=True,
    )
    print(f"Done. Data in {os.path.join(data_dir, 'Task09_Spleen')}")


if __name__ == "__main__":
    main()
