import os
import shutil

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from scripts.sportsmot_cropper import SportsMOTCropper


@hydra.main(version_base="1.1", config_path="config", config_name="crop_sportsMOT")
def main(cfg: DictConfig):
    cropper: SportsMOTCropper = instantiate(cfg.cropper)

    # copy splits file
    src = os.path.join(cropper.splits_dir, f"{cropper.split}.txt")
    dst = os.path.join(cfg.splits_save_dir, f"{cropper.split}.txt")
    copy_file(src, dst)

    cropper.crop_and_save()


def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
