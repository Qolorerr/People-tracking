import hydra
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from ultralytics import YOLO

from scripts import Tester
from src.base import BaseDataLoader


@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(cfg: DictConfig):
    dataloader: BaseDataLoader = instantiate(cfg.test_loader)

    accelerator: Accelerator = instantiate(cfg.accelerator)

    detection_model: YOLO | None = instantiate(cfg.detection_model)
    feature_extractor_model: nn.Module = instantiate(cfg.feature_extractor_model, num_classes=2048)

    tester = Tester(
        dataloader=dataloader,
        accelerator=accelerator,
        detection_model=detection_model,
        feature_extractor_model=feature_extractor_model,
        config=cfg,
        resume=cfg.resume,
    )
    tester.test()


if __name__ == "__main__":
    main()
