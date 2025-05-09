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
from queue import Queue

from PyQt5.QtWidgets import QApplication
from accelerate import Accelerator
from hydra.utils import instantiate
from numpy._typing import NDArray
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.utils import tensorboard
from ultralytics import YOLO

from src.utils import TrackManager, TrackVisualizer, CropBboxesOutOfFramesMixin, VideoWindow, MetricsMeter


class LiveTester(CropBboxesOutOfFramesMixin):
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
            self.detection_model,
            self.feature_extractor_model
        )
        self.detection_model = cast(YOLO, self.detection_model)
        self.feature_extractor_model.eval()

        cfg_tester = self.config["live_tester"]
        self.rtsp_url = cfg_tester.rtsp_url
        self.transforms = instantiate(cfg_tester.transforms)

        # CHECKPOINTS & TENSORBOARD
        start_time = datetime.now().strftime("%m-%d_%H-%M")
        writer_dir = str(
            os.path.join(cfg_tester["log_dir"], self.config["name"], start_time)
        )
        self.writer = tensorboard.SummaryWriter(writer_dir)
        info_to_write = [
            "crop_size",
            "detection_model",
            "feature_extractor_model",
            "live_tester",
        ]
        for info in info_to_write:
            self.writer.add_text(f"info/{info}", str(self.config[info]))
        self.wrt_mode, self.wrt_step = "test", 0
        self.log_step: int = cfg_tester.get("log_per_iter", 1)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metric_store = MetricsMeter(writer=self.writer)

        self.confidence_threshold = self.config["confidence_threshold"]
        self.person_reshape_h, self.person_reshape_w = self.config["person_reshape_h"], self.config["person_reshape_w"]

        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.frame_idx = 1

        if resume:
            self._resume_checkpoint(resume)

    def _capture_frames(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Connection lost, attempting reconnect...")
                cap = cv2.VideoCapture(self.rtsp_url)
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.frame_queue.full():
                self.frame_queue.get()
            self.frame_queue.put(frame)
            self.frame_idx += 1
            self.wrt_step += 1

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

        return {
            'bboxes': bboxes,
            'confs': confs,
            'features': features
        }

    def _process_frame(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        try:
            transformed = self.transforms(image=frame)
        except Exception as e:
            print("Exception:", self.frame_idx, datetime.now())
            raise e
        frame_tensor = transformed["image"].float().to(self.device) / 255.0

        detections = self._extract_bboxes(frame_tensor.unsqueeze(0))
        active_tracks = self.tracklet_master.update(frame_idx=self.frame_idx,
                                                    bboxes=detections['bboxes'],
                                                    features=detections['features'])

        track_manager_metrics = self.tracklet_master.get_metrics()
        self.metric_store.update(track_manager_metrics)
        if self.log_step and self.frame_idx % self.log_step == 0:
            self.metric_store.log_metrics(self.wrt_mode, self.wrt_step)

        img = frame_tensor.cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        img = self.visualizer.draw_np(img, active_tracks)

        return img

    def run(self):
        self.running = True
        capture_thread = threading.Thread(target=self._capture_frames)
        capture_thread.start()

        app = QApplication(sys.argv)
        window = VideoWindow()
        window.closed.connect(lambda: setattr(self, 'running', False))
        window.show()

        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                processed_frame = self._process_frame(frame)

                window.update_frame(processed_frame)
                window.update_fps(self.frame_idx)

                app.processEvents()

    def _resume_checkpoint(self, resume_path: str) -> None:
        self.logger.info(f"Loading checkpoint : {resume_path}")
        checkpoint = torch.load(resume_path)

        if checkpoint["arch"] != str(type(self.feature_extractor_model).__name__):
            self.logger.warning(
                {"Warning! Current model is not the same as the one in the checkpoint"}
            )
        self.feature_extractor_model.load_state_dict(checkpoint["state_dict"])
