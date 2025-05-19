import os
from typing import Any

import albumentations
import cv2
from tqdm import tqdm

from src.datasets.sportsmot import read_annotations
from src.utils.helpers import save_img, save_gt
from src.utils.mixins import SimpleCropBboxesOutOfFramesMixin, TransformFrameMixin


class SportsMOTCropper(SimpleCropBboxesOutOfFramesMixin, TransformFrameMixin):
    def __init__(
        self,
        root: str,
        split: str,
        transforms: albumentations.Compose,
        splits_dir: str,
        person_reshape_h: int,
        person_reshape_w: int,
        dataset_save_dir: str,
    ):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.splits_dir = splits_dir
        self.person_reshape_h, self.person_reshape_w = person_reshape_h, person_reshape_w
        self.dataset_save_dir = dataset_save_dir

        split_file = os.path.join(splits_dir, f"{split}.txt")
        with open(split_file, "r") as f:
            self.video_names = [line.strip() for line in f.readlines()]

        self.annotations: list[dict[int, list[dict[str, Any]]]] = []

        self._set_samples()

    def _set_samples(self):
        self._max_num_pids = 0
        for video_name in self.video_names:
            video_path = os.path.join(self.root, self.split, video_name)

            annotations, num_pids = read_annotations(video_path)
            self._max_num_pids = max(num_pids, self._max_num_pids)

            self.annotations.append(annotations)

    def crop_and_save(self):
        tbar = tqdm(self.annotations, desc="Crop and save bboxes")
        for video_idx, video_annotations in enumerate(tbar):
            video_name = self.video_names[video_idx]
            video_path = os.path.join(self.root, self.split, video_name)

            save_video_path = os.path.join(self.dataset_save_dir, self.split, video_name)
            os.makedirs(os.path.join(save_video_path, "img1"), exist_ok=True)
            os.makedirs(os.path.join(save_video_path, "gt"), exist_ok=True)

            video_labels, video_save_ids = [], []
            for frame_id, frame_annotations in video_annotations.items():
                frame_path = os.path.join(video_path, "img1", f"{frame_id:06d}.jpg")
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                boxes, labels, save_ids = [], [], []

                for ann in frame_annotations:
                    x, y, w, h = ann["bbox"]
                    boxes.append([x, y, x + w, y + h])
                    labels.append(self._max_num_pids * video_idx + int(ann["person_id"]))
                    save_ids.append(ann["idx"])

                frame, boxes, labels = self.transform_frame(frame, boxes, labels, error_message=f"({video_idx}, {frame_id})")
                crops = self.process_frame(frame=frame, bboxes=boxes)

                video_labels.extend(labels)
                video_save_ids.extend(save_ids)

                # save crops
                for save_id, crop in zip(save_ids, crops):
                    frame_path = os.path.join(save_video_path, "img1", f"{save_id:06d}.jpg")
                    save_img(frame_path, crop)

            # save gt
            gt_path = os.path.join(save_video_path, "gt", "gt.txt")
            save_gt(gt_path, video_save_ids, video_labels)
