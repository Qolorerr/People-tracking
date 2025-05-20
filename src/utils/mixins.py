from typing import Any

import numpy as np
import torch
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision import transforms
from ultralytics.engine.results import Results


class CropBboxesOutOfFramesMixin:
    def _reshape_cropped_img(self, img: Tensor) -> Tensor:
        target_h, target_w = self.person_reshape_h, self.person_reshape_w

        _, _, h, w = img.shape

        scale = min(target_h / h, target_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        resized = F.resize(img, [new_h, new_w])

        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left

        padded = F.pad(resized, [pad_left, pad_top, pad_right, pad_bottom])

        return padded

    def crop_bboxes(self, frame: Tensor, bboxes: Tensor) -> Tensor:
        crops: list[Tensor] = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.int()
            crop: Tensor = frame[..., y1:y2, x1:x2]

            crops.append(self._reshape_cropped_img(crop))

        batch = torch.cat(crops)
        return batch

    def extract_bboxes(self, frame: Tensor, detections: Results) -> dict[str, Tensor]:
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

    def detect_and_extract_bboxes(self, frame: Tensor) -> dict[str, Tensor]:
        with torch.no_grad():
            detections = self.detection_model.predict(frame, verbose=False)[0]

            norm_frame = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(frame)

            data = self.extract_bboxes(norm_frame, detections)

        return data


class LoadAndSaveParamsMixin:
    def load_params(self, params: dict[str, Any]) -> None:
        self.confidence_threshold = params["confidence_threshold"]
        self.tracklet_master.motion_weight = params["motion_weight"]
        self.tracklet_master.appearance_weight = params["appearance_weight"]
        self.tracklet_master.match_threshold = params["match_threshold"]

    def get_params(self) -> dict[str, Any]:
        return {
            "confidence_threshold": self.confidence_threshold,
            "motion_weight": self.tracklet_master.motion_weight,
            "appearance_weight": self.tracklet_master.appearance_weight,
            "match_threshold": self.tracklet_master.match_threshold,
        }


class VisualizeAndWriteFrameMixin:
    def visualize_frame(self, frames: Tensor, active_tracks: list[dict[str, Any]]) -> Tensor:
        if len(frames.shape) == 4:
            img = frames[0]
        else:
            img = frames

        img = self.visualizer.draw(img, active_tracks)

        return img

    def visualize_and_write_frame(
        self, frames: Tensor, active_tracks: list[dict[str, Any]]
    ) -> None:
        if self.wrt_step % self.log_step != 0:
            return

        img = self.visualize_frame(frames, active_tracks)

        self.writer.add_image(
            tag=f"{self.wrt_mode}/tracklet_predictions",
            img_tensor=img,
            global_step=self.wrt_step,
            dataformats="CHW",
        )


class SimpleCropBboxesOutOfFramesMixin:
    def _reshape_cropped_img(self, img: Tensor) -> Tensor:
        target_h, target_w = self.person_reshape_h, self.person_reshape_w

        _, h, w = img.shape

        scale = min(target_h / h, target_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        resized = F.resize(img, [new_h, new_w])

        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left

        padded = F.pad(resized, [pad_left, pad_top, pad_right, pad_bottom])

        return padded

    def process_frame(self, frame: Tensor, bboxes: Tensor) -> list[Tensor]:
        crops = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.int()
            crop: Tensor = frame[:, y1:y2, x1:x2]

            crop = self._reshape_cropped_img(crop)
            crops.append(crop)

        return crops


class TransformFrameMixin:
    def transform_frame(self, frame: np.ndarray, boxes: list[list[int]], labels: list[int], error_message: str) -> (Tensor, Tensor, list[int]):
        boxes = (
            np.array(boxes, dtype=np.int64) if boxes else np.zeros((0, 4), dtype=np.int64)
        )
        labels = (
            np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        )

        try:
            transformed = self.transforms(image=frame, bboxes=boxes, class_labels=labels)
        except Exception as e:
            print("Exception: " + error_message)
            raise e
        frame, boxes, labels = (
            transformed["image"],
            transformed["bboxes"],
            transformed["class_labels"],
        )
        frame = frame.float() / 255.0
        boxes = torch.from_numpy(boxes).to(torch.int64)

        return frame, boxes, labels
