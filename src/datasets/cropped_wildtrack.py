import os
from pathlib import Path
from typing import Any

import cv2
import torch
from torch import Tensor, LongTensor

from src.base import BaseDataset


class CroppedWildTrackDataset(BaseDataset):
    def __init__(self, load_limit: int | None = None, **kwargs):
        self.annotations: dict[int, Any] = {}
        super().__init__(**kwargs)
        if load_limit is not None:
            self.samples = self.samples[:load_limit]

    def _set_samples(self, test: bool = False):
        split_path = os.path.join(self.root, self.split)

        if not test:
            gt_path = os.path.join(split_path, "annotations_positions", "annotation.txt")
            with open(gt_path, "r") as f:
                for line in f:
                    parts = line.strip().split(", ")
                    frame_id = int(parts[0])
                    person_id = int(parts[1])

                    self.annotations[frame_id] = person_id

        frame_ids = [int(Path(png_file).stem) for png_file in os.listdir(os.path.join(split_path, "Image_subsets"))]
        frame_ids.sort()
        for frame_id in frame_ids:
            self.samples.append((0, frame_id))

    def __getitem__(self, idx) -> (Tensor, Tensor, Tensor, LongTensor, Tensor):
        _, frame_id = self.samples[idx]

        frame_path = os.path.join(self.root, self.split, "Image_subsets", f"{frame_id:08d}.png")
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_id_tensor = torch.tensor([0, idx], dtype=torch.int64)

        try:
            transformed = self.transforms(image=frame)
        except Exception as e:
            print("Exception:", self.samples[idx])
            raise e
        frame = transformed["image"]
        frame = frame.float() / 255.0

        if self.split == "test":
            return frame, torch.tensor([]), torch.tensor([]), torch.tensor([])

        label = self.annotations.get(frame_id, 0)

        return (
            frame_id_tensor,
            frame,
            torch.tensor([]),
            torch.tensor(label, dtype=torch.long),
            torch.tensor([]),
        )
