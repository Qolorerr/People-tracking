import os
import configparser
import random
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor, LongTensor

from src.base import BaseDataset


class SportsMOTDataset(BaseDataset):
    def __init__(
            self,
            splits_dir: str,
            load_limit: int | None = None,
            shuffle: bool = False,
            **kwargs
    ):
        self.splits_dir = splits_dir

        split_file = os.path.join(splits_dir, f"{kwargs['split']}.txt")
        with open(split_file, 'r') as f:
            self.video_names = [line.strip() for line in f.readlines()]
        if shuffle:
            random.shuffle(self.video_names)

        self.video_info: list[dict[str, Any]] = []
        super().__init__(**kwargs)
        if load_limit is not None:
            self.samples = self.samples[:load_limit]

    def _set_samples(self, test: bool = False):
        self._max_num_pids = 0
        for video_idx, video_name in enumerate(self.video_names):
            video_path = os.path.join(self.root, self.split, video_name)
            seqinfo_path = os.path.join(video_path, 'seqinfo.ini')

            config = configparser.ConfigParser()
            config.read(seqinfo_path)
            seq_info = config['Sequence']
            im_dir = os.path.join(video_path, seq_info['imDir'])
            im_ext = seq_info['imExt']
            frame_count = int(seq_info['seqLength'])
            im_width = int(seq_info['imWidth'])
            im_height = int(seq_info['imHeight'])
            frame_rate = int(seq_info['frameRate'])

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
                        x, y, w, h = map(int, parts[2:6])

                        if frame_id not in annotations:
                            annotations[frame_id] = []
                        annotations[frame_id].append({
                            'person_id': person_id,
                            'bbox': [x, y, w, h],
                        })
                self._max_num_pids = max(len(pids), self._max_num_pids)

            self.video_info.append({
                'im_dir': im_dir,
                'im_ext': im_ext,
                'annotations': annotations,
                'frame_count': frame_count,
                'metadata': {
                    'im_width': im_width,
                    'im_height': im_height,
                    'frame_rate': frame_rate
                }
            })

            for frame_id in range(1, frame_count + 1):
                self.samples.append((video_idx, frame_id))

    def __getitem__(self, idx) -> (Tensor, Tensor, Tensor, LongTensor, Tensor):
        video_idx, frame_id = self.samples[idx]
        video = self.video_info[video_idx]

        frame_path = os.path.join(video['im_dir'], f"{frame_id:06d}{video['im_ext']}")
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_id_tensor = torch.tensor([0, idx], dtype=torch.int64)
        is_new_video = torch.tensor([frame_id == 1], dtype=torch.bool)

        if self.split == 'test':
            try:
                transformed = self.transforms(image=frame)
            except Exception as e:
                print("Exception:", self.samples[idx])
                raise e
            frame = transformed["image"]
            frame = frame.float() / 255.0

            return frame_id_tensor, frame, torch.tensor([]), torch.tensor([]), is_new_video

        anns: list[dict[str, Any]] = video['annotations'].get(frame_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self._max_num_pids * video_idx + int(ann['person_id']))

        boxes = np.array(boxes, dtype=np.int64) if boxes else \
            np.zeros((0, 4), dtype=np.int64)
        labels = np.array(labels, dtype=np.int64) if labels else \
            np.zeros((0,), dtype=np.int64)

        try:
            transformed = self.transforms(image=frame, bboxes=boxes, class_labels=labels)
        except Exception as e:
            print("Exception:", self.samples[idx])
            raise e
        frame, boxes, labels = transformed["image"], transformed["bboxes"], transformed["class_labels"]
        frame = frame.float() / 255.0
        boxes = torch.from_numpy(boxes).to(torch.int64)
        labels = torch.tensor(labels, dtype=torch.long)

        return frame_id_tensor, frame, boxes, labels, is_new_video
