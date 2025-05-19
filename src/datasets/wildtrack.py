import json
import os
from pathlib import Path
from typing import Any

import cv2
import torch
from torch import Tensor, LongTensor

from src.base import BaseDataset
from src.utils.mixins import TransformFrameMixin


class WildTrackDataset(BaseDataset, TransformFrameMixin):
    def __init__(
        self, load_limit: int | None = None, **kwargs
    ):
        self.annotations: list[dict[str, Any]] = []
        super().__init__(**kwargs)
        if load_limit is not None:
            self.samples = self.samples[:load_limit]

    def _set_samples(self, test: bool = False):
        frame_ids = [int(Path(png_file).stem) for png_file in os.listdir(os.path.join(self.root, self.split, "Image_subsets", "C1"))]
        frame_ids.sort()

        for frame_id in frame_ids:
            for camera_id in range(7):
                self.samples.append((camera_id, frame_id))

        self.annotations = read_annotations(self.root, self.split)

    def __getitem__(self, idx) -> (Tensor, Tensor, Tensor, LongTensor, Tensor):
        camera_id, frame_id = self.samples[idx]
        cam_anns = self.annotations[camera_id]

        frame_path = os.path.join(self.root, "Image_subsets", f"C{camera_id + 1}", f"{frame_id:08d}.png")
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_id_tensor = torch.tensor([camera_id, frame_id // 5], dtype=torch.int64)
        is_new_video = torch.tensor([False], dtype=torch.bool)

        anns: list[dict[str, Any]] = cam_anns.get(frame_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann["person_id"]))

        frame, boxes, labels = self.transform_frame(frame, boxes, labels, error_message=f"({camera_id}, {frame_id})")
        labels = torch.tensor(labels, dtype=torch.long)

        return frame_id_tensor, frame, boxes, labels, is_new_video


def read_annotations(root: str, split: str) -> list[dict[int, list[dict[str, Any]]]]:
    annotations = [{} for _ in range(7)]
    bbox_idx = 0

    annotations_dir = os.path.join(root, split, 'annotations_positions')

    for json_file in os.listdir(annotations_dir):
        if not json_file.endswith('.json'):
            continue

        frame_id = int(Path(json_file).stem)
        json_path = os.path.join(annotations_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)

        for person in data:
            person_id = person['personID']
            for view in person['views']:
                camera_id = view['viewNum']
                xmin, xmax, ymin, ymax = view['xmin'], view['xmax'], view['ymin'], view['ymax']
                if xmin < 0 or ymin < 0 or xmax >= 1920 or ymax >= 1080:
                    continue
                x, y, w, h = xmin, ymin, (xmax - xmin), (ymax - ymin)
                if w <= 0 or h <= 0:
                    continue

                if frame_id not in annotations[camera_id]:
                    annotations[camera_id][frame_id] = []
                annotations[camera_id][frame_id].append(
                    {
                        "person_id": person_id,
                        "bbox": [x, y, w, h],
                        "idx": bbox_idx,
                    }
                )
                bbox_idx += 1
    return annotations
