import os
from typing import Any

import cv2
import torch
from torch import Tensor, LongTensor

from src.base import BaseDataset


class CroppedSportsMOTDataset(BaseDataset):
    def __init__(
            self,
            splits_dir: str,
            load_limit: int | None = None,
            **kwargs
    ):
        self.splits_dir = splits_dir

        split_file = os.path.join(splits_dir, f"{kwargs['split']}.txt")
        with open(split_file, 'r') as f:
            self.video_names = [line.strip() for line in f.readlines()]

        self.video_info: list[dict[str, Any]] = []
        super().__init__(**kwargs)
        if load_limit is not None:
            self.samples = self.samples[:load_limit]

    def _set_samples(self, test: bool = False):
        self._max_num_pids = 0
        for video_idx, video_name in enumerate(self.video_names):
            video_path = os.path.join(self.root, self.split, video_name)

            annotations = {}
            if not test:
                pids = set()
                gt_path = os.path.join(video_path, 'gt', 'gt.txt')
                with open(gt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(', ')
                        frame_id = int(parts[0])
                        person_id = int(parts[1])
                        pids.add(person_id)

                        annotations[frame_id] = person_id
                self._max_num_pids = max(len(pids), self._max_num_pids)

            self.video_info.append({
                'im_dir': os.path.join(video_path, "img1"),
                'im_ext': ".jpg",
                'annotations': annotations,
            })

            for frame_id in annotations.keys():
                self.samples.append((video_idx, frame_id))

    def __getitem__(self, idx) -> (Tensor, Tensor, Tensor, LongTensor, Tensor):
        video_idx, frame_id = self.samples[idx]
        video = self.video_info[video_idx]

        frame_path = os.path.join(video['im_dir'], f"{frame_id:06d}{video['im_ext']}")
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

        if self.split == 'test':
            return frame, torch.tensor([]), torch.tensor([]), torch.tensor([])

        label = video['annotations'].get(frame_id, 0)

        return frame_id_tensor, frame, torch.tensor([]), torch.tensor(label, dtype=torch.long), torch.tensor([])
