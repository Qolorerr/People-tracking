import hydra
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from ultralytics import YOLO

from src import Trainer


@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(cfg: DictConfig):
    train_dataloader: DataLoader = instantiate(cfg.train_loader)
    val_dataloader: DataLoader = instantiate(cfg.val_loader)

    accelerator: Accelerator = instantiate(cfg.accelerator)

    detection_model: YOLO | None = instantiate(cfg.detection_model)
    feature_extractor_model: nn.Module = instantiate(cfg.feature_extractor_model, num_classes=1024)

    trainer = Trainer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        accelerator=accelerator,
        detection_model=detection_model,
        feature_extractor_model=feature_extractor_model,
        config=cfg,
        resume=cfg.resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()
