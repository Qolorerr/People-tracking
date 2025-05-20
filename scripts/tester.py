import logging
import os
from datetime import datetime, timedelta
from typing import cast

import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils import tensorboard
from tqdm import tqdm
from ultralytics import YOLO

from src.base import BaseDataLoader
from src.utils import (
    TrackManager,
    TrackVisualizer,
    CropBboxesOutOfFramesMixin,
    VisualizeAndWriteFrameMixin,
    GlobalTrackManager,
    MetricsMeter,
    LoadAndSaveParamsMixin,
)


class Tester(CropBboxesOutOfFramesMixin, VisualizeAndWriteFrameMixin, LoadAndSaveParamsMixin):
    def __init__(
        self,
        dataloader: BaseDataLoader,
        accelerator: Accelerator,
        detection_model: YOLO | None,
        feature_extractor_model: nn.Module,
        config: DictConfig,
        resume: str | None = None,
    ):
        self.dataloader: BaseDataLoader = dataloader
        self.accelerator: Accelerator = accelerator
        self.device = self.accelerator.device
        self.detection_model: YOLO | None = detection_model
        self.feature_extractor_model: nn.Module = feature_extractor_model
        self.config: DictConfig = config

        self.tracklet_manager: TrackManager = instantiate(
            config.tracklet_master, device=self.device
        )
        self.visualizer = TrackVisualizer()

        self.detection_model, self.feature_extractor_model = self.accelerator.prepare(
            self.detection_model, self.feature_extractor_model
        )
        self.detection_model = cast(YOLO, self.detection_model)
        self.feature_extractor_model.eval()

        cfg_tester = self.config["tester"]

        self._is_multicam: bool = False
        if isinstance(self.tracklet_manager, GlobalTrackManager):
            for camera_id in self.dataloader.camera_ids:
                self.tracklet_manager.add_camera(camera_id)
            self._is_multicam = True

        # CHECKPOINTS & TENSORBOARD
        start_time = datetime.now().strftime("%m-%d_%H-%M")
        writer_dir = str(os.path.join(cfg_tester["log_dir"], self.config["name"], start_time))
        self.writer = tensorboard.SummaryWriter(writer_dir)
        info_to_write = [
            "test_loader",
            "detection_model",
            "feature_extractor_model",
            "tester",
        ]
        for info in info_to_write:
            self.writer.add_text(f"info/{info}", str(self.config[info]))
        self.wrt_mode, self.wrt_step = "test", 0
        self.log_step: int = cfg_tester.get("log_per_iter", 1)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metric_store = MetricsMeter(writer=self.writer)

        self.confidence_threshold = self.config["confidence_threshold"]
        self.person_reshape_h, self.person_reshape_w = (
            self.config["person_reshape_h"],
            self.config["person_reshape_w"],
        )

        self.dataloader = self.accelerator.prepare(self.dataloader)

        if resume:
            self._resume_checkpoint(resume)

    def test(self) -> None:
        tbar = tqdm(self.dataloader, desc="Test")
        for self.wrt_step, (frame_info, frame, _, _, is_new_video) in enumerate(tbar):

            camera_id, frame_idx = frame_info[0].int()
            camera_id, frame_idx = camera_id.item(), frame_idx.item()
            if is_new_video.item():
                self.tracklet_manager.reset()

            start_time = datetime.now()

            detections = self.detect_and_extract_bboxes(frame)

            kwargs = {
                "frame_idx": frame_idx,
                "bboxes": detections["bboxes"],
                "features": detections["features"],
            }
            if self._is_multicam:
                kwargs["camera_id"] = camera_id
                self.wrt_mode = f"test/cam{camera_id}"

            active_tracks = self.tracklet_manager.update(**kwargs)

            track_manager_metrics = self.tracklet_manager.get_metrics()
            self.metric_store.update(track_manager_metrics)

            if isinstance(self.tracklet_manager, LoadAndSaveParamsMixin):
                tracker_params = self.get_params()
                self.metric_store.update(tracker_params)

            processing_time = datetime.now() - start_time

            if self.log_step and frame_idx % self.log_step == 0:
                self._log_fps_metric(processing_time)
                self.visualize_and_write_frame(frame, active_tracks)
                self.metric_store.log_metrics(self.wrt_mode + f"/cam{camera_id}", frame_idx)

    def _log_fps_metric(self, processing_time: timedelta) -> None:
        fps = int(1 / processing_time.total_seconds())
        self.writer.add_scalar(f"{self.wrt_mode}/fps", fps, self.wrt_step)

    def _resume_checkpoint(self, resume_path: str) -> None:
        self.logger.info("Loading checkpoint : %s", resume_path)
        checkpoint = torch.load(resume_path)

        if checkpoint["arch"] != str(type(self.feature_extractor_model).__name__):
            self.logger.warning(
                {"Warning! Current model is not the same as the one in the checkpoint"}
            )
        self.feature_extractor_model.load_state_dict(checkpoint["state_dict"])
