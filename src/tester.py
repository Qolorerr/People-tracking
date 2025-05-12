import os
from datetime import datetime, timedelta
from typing import Callable, Any

import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO

from src.utils import TrackManager, TrackVisualizer, CropBboxesOutOfFramesMixin


class Tester(CropBboxesOutOfFramesMixin):
    def __init__(
        self,
        dataloader: DataLoader,
        accelerator: Accelerator,
        detection_model: YOLO | None,
        feature_extractor: Callable,
        config: DictConfig,
    ):
        self.dataloader: DataLoader = dataloader
        self.accelerator: Accelerator = accelerator
        self.device = self.accelerator.device
        self.detection_model: YOLO | None = detection_model
        self.feature_extractor: Callable = feature_extractor
        self.config: DictConfig = config

        self.tracklet_master: TrackManager = instantiate(config.tracklet_master, device=self.device)
        self.visualizer = TrackVisualizer()

        self.detection_model, self.feature_extractor = self.accelerator.prepare(
            self.detection_model, self.feature_extractor
        )

        cfg_tester = self.config["tester"]

        # CHECKPOINTS & TENSORBOARD
        start_time = datetime.now().strftime("%m-%d_%H-%M")
        writer_dir = str(os.path.join(cfg_tester["log_dir"], self.config["name"], start_time))
        self.writer = tensorboard.SummaryWriter(writer_dir)
        info_to_write = [
            "crop_size",
            "test_loader",
            "detection_model",
            "feature_extractor",
            "tester",
        ]
        for info in info_to_write:
            self.writer.add_text(f"info/{info}", str(self.config[info]))
        self.wrt_mode, self.wrt_step = "test", 0
        self.log_step: int = cfg_tester.get("log_per_iter", 1)

        self.confidence_threshold = self.config["confidence_threshold"]
        self.person_reshape_h, self.person_reshape_w = (
            self.config["person_reshape_h"],
            self.config["person_reshape_w"],
        )

        self.dataloader = self.accelerator.prepare(self.dataloader)

    def process_frame(self, frame: Tensor):
        with torch.no_grad():
            detections = self.detection_model(frame)[0]

        bboxes: Tensor = detections.boxes.xyxy
        confs: Tensor = detections.boxes.conf
        class_ids: Tensor = detections.boxes.cls

        person_mask = class_ids == 0
        confidence_mask = confs >= self.confidence_threshold
        mask = person_mask & confidence_mask
        bboxes = bboxes[mask]
        confs = confs[mask]

        features = []
        if len(bboxes) > 0:
            batch = self.crop_bboxes(frame, bboxes)

            with torch.no_grad():
                features = self.feature_extractor(batch)

        return {"bboxes": bboxes, "confs": confs, "features": features}

    def test(self) -> None:
        tbar = tqdm(self.dataloader, desc="Test")
        for self.wrt_step, (_, frame, _, _, is_new_video) in enumerate(tbar):
            if is_new_video.item():
                self.tracklet_master.reset()

            start_time = datetime.now()
            detections = self.process_frame(frame)
            active_tracks = self.tracklet_master.update(
                frame_idx=self.wrt_step,
                bboxes=detections["bboxes"],
                features=detections["features"],
            )
            # pprint(active_tracks)
            processing_time = datetime.now() - start_time

            self._log_fps_metric(processing_time)
            self._visualize_frame(frame, active_tracks)

    def _visualize_frame(self, frames: Tensor, active_tracks: list[dict[str, Any]]) -> None:
        if self.wrt_step % self.log_step != 0:
            return

        if len(frames.shape) == 4:
            img = frames[0]
        else:
            img = frames

        img = self.visualizer.draw(img, active_tracks)
        self.writer.add_image(
            tag=f"{self.wrt_mode}/tracklet_predictions",
            img_tensor=img,
            global_step=self.wrt_step,
            dataformats="CHW",
        )

    def _log_fps_metric(self, processing_time: timedelta) -> None:
        fps = int(1 / processing_time.total_seconds())
        self.writer.add_scalar(f"{self.wrt_mode}/fps", fps, self.wrt_step)
