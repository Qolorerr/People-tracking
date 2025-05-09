from typing import Any

import torch
from torch import Tensor
import torchvision.transforms.functional as F


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
