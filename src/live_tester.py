import logging
import os
import sys
from datetime import datetime
from pprint import pprint
from typing import cast

import cv2
import numpy as np
import torch
import threading

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from accelerate import Accelerator
from hydra.utils import instantiate
from numpy._typing import NDArray
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.utils import tensorboard
from ultralytics import YOLO

from src.utils import (
    TrackManager,
    TrackVisualizer,
    CropBboxesOutOfFramesMixin,
    VideoWindow,
    MetricsMeter,
    LoadAndSaveParamsMixin,
    GlobalTrackManager,
    CameraWorker,
)


class LiveTester(CropBboxesOutOfFramesMixin, LoadAndSaveParamsMixin):
    def __init__(
        self,
        accelerator: Accelerator,
        detection_model: YOLO | None,
        feature_extractor_model: nn.Module,
        config: DictConfig,
        resume: str,
    ):
        self.accelerator: Accelerator = accelerator
        self.device = self.accelerator.device
        self.detection_model: YOLO | None = detection_model
        self.feature_extractor_model: nn.Module = feature_extractor_model
        self.config: DictConfig = config

        self.tracklet_master: TrackManager = instantiate(config.tracklet_master, device=self.device)
        self.visualizer = TrackVisualizer()

        self.detection_model, self.feature_extractor_model = self.accelerator.prepare(
            self.detection_model, self.feature_extractor_model
        )
        self.detection_model = cast(YOLO, self.detection_model)
        self.feature_extractor_model.eval()

        cfg_tester = self.config["live_tester"]
        self.rtsp_urls = cfg_tester.rtsp_urls
        self.transforms = instantiate(cfg_tester.transforms)

        self.is_multicam: bool = False
        if isinstance(self.tracklet_master, GlobalTrackManager):
            for camera_id in range(len(self.rtsp_urls)):
                self.tracklet_master.add_camera(camera_id)
            self.is_multicam = True

        # CHECKPOINTS & TENSORBOARD
        start_time = datetime.now().strftime("%m-%d_%H-%M")
        writer_dir = str(os.path.join(cfg_tester["log_dir"], self.config["name"], start_time))
        self.writer = tensorboard.SummaryWriter(writer_dir)
        info_to_write = [
            "crop_size",
            "detection_model",
            "feature_extractor_model",
            "live_tester",
        ]
        for info in info_to_write:
            self.writer.add_text(f"info/{info}", str(self.config[info]))
        self.wrt_mode = "test"
        self.log_step: int = cfg_tester.get("log_per_iter", 1)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metric_store = MetricsMeter(writer=self.writer)

        self.confidence_threshold = self.config["confidence_threshold"]
        self.person_reshape_h, self.person_reshape_w = (
            self.config["person_reshape_h"],
            self.config["person_reshape_w"],
        )

        self.running = False
        self.frame_idxes = [1 for _ in range(len(self.rtsp_urls))]

        self.app = QApplication(sys.argv)
        self.window = VideoWindow(len(self.rtsp_urls))
        self.window.closed.connect(lambda: setattr(self, "running", False))

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)

        if resume:
            self._resume_checkpoint(resume)

    def _capture_frames(self, camera_id: int, worker: CameraWorker):
        cap = cv2.VideoCapture(self.rtsp_urls[camera_id])
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Connection lost, attempting reconnect...")
                cap = cv2.VideoCapture(self.rtsp_urls[camera_id])
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with worker.queue_lock:
                if worker.queue.full():
                    worker.queue.get()
                worker.queue.put(frame)

            self.frame_idxes[camera_id] += 1

    def _extract_bboxes(self, frame: Tensor) -> dict[str, Tensor]:
        with torch.no_grad():
            detections = self.detection_model.predict(frame, verbose=False)[0]

            bboxes: Tensor = detections.boxes.xyxy
            confs: Tensor = detections.boxes.conf
            class_ids: Tensor = detections.boxes.cls

            person_mask = class_ids == 0
            confidence_mask = confs >= self.confidence_threshold
            valid_bbox_mask = (bboxes[:, 0] != bboxes[:, 2]) & (bboxes[:, 1] != bboxes[:, 3])
            mask = person_mask & confidence_mask & valid_bbox_mask
            bboxes = bboxes[mask]
            confs = confs[mask]

            features = []
            if len(bboxes) > 0:
                batch = self.crop_bboxes(frame, bboxes)

                features = self.feature_extractor_model(batch)

        return {"bboxes": bboxes, "confs": confs, "features": features}

    def _process_frame(self, camera_id: int, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        frame_idx = self.frame_idxes[camera_id]

        try:
            transformed = self.transforms(image=frame)
        except Exception as e:
            print("Exception:", frame_idx, datetime.now())
            raise e
        frame_tensor = transformed["image"].float().to(self.device) / 255.0

        detections = self._extract_bboxes(frame_tensor.unsqueeze(0))

        kwargs = {
            "frame_idx": frame_idx,
            "bboxes": detections["bboxes"],
            "features": detections["features"],
        }
        if self.is_multicam:
            kwargs["camera_id"] = camera_id

        active_tracks = self.tracklet_master.update(**kwargs)

        track_manager_metrics = self.tracklet_master.get_metrics()
        self.metric_store.update(track_manager_metrics)

        if isinstance(self.tracklet_master, LoadAndSaveParamsMixin):
            tracker_params = self.get_params()
            self.metric_store.update(tracker_params)

        if self.log_step and frame_idx % self.log_step == 0:
            self.metric_store.log_metrics(self.wrt_mode + f"/cam{camera_id}", frame_idx)

        img = frame_tensor.cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        img = self.visualizer.draw_np(img, active_tracks)

        return img

    def update_frames(self):
        if not self.running:
            return
        for worker, widget in zip(self.window.workers, self.window.camera_widgets):
            with worker.queue_lock:
                if not worker.queue.empty():
                    frame = worker.queue.get()

                    processed_frame = self._process_frame(worker.camera_id, frame)

                    widget.update_frame(self.frame_idxes[worker.camera_id], processed_frame)

    def run(self):
        self.running = True
        self.window.closed.connect(lambda: setattr(self, "running", False))

        self.timer.start(30)

        for i in range(len(self.rtsp_urls)):
            worker = CameraWorker(i, self.window)
            thread = threading.Thread(
                target=self._capture_frames,
                args=(
                    i,
                    worker,
                ),
            )
            thread.start()

            self.window.workers.append(worker)
            self.window.threads.append(thread)

        self.window.show()

        self.app.exec()

    def _resume_checkpoint(self, resume_path: str) -> None:
        self.logger.info(f"Loading checkpoint : {resume_path}")
        checkpoint = torch.load(resume_path)

        if checkpoint["arch"] != str(type(self.feature_extractor_model).__name__):
            self.logger.warning(
                {"Warning! Current model is not the same as the one in the checkpoint"}
            )
        self.feature_extractor_model.load_state_dict(checkpoint["state_dict"])
