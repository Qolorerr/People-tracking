import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from scripts.wildtrack_cropper import WildTrackCropper


@hydra.main(version_base="1.1", config_path="config", config_name="crop_wildtrack")
def main(cfg: DictConfig):
    cropper: WildTrackCropper = instantiate(cfg.cropper)

    cropper.crop_and_save()


if __name__ == "__main__":
    main()
