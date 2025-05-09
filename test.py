from typing import Callable

import hydra
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from ultralytics import YOLO

from src import Tester


@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(cfg: DictConfig):
    dataloader: DataLoader = instantiate(cfg.test_loader)

    accelerator: Accelerator = instantiate(cfg.accelerator)

    detection_model: YOLO | None = instantiate(cfg.detection_model)
    feature_extractor: Callable = instantiate(cfg.feature_extractor, device=str(accelerator.device))

    tester = Tester(
        dataloader,
        accelerator,
        detection_model,
        feature_extractor,
        cfg
    )
    tester.test()


if __name__ == "__main__":
    main()
