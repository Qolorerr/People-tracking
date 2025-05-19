import os
from pathlib import Path
from typing import Any

import albumentations
import cv2
from tqdm import tqdm

from src.datasets.wildtrack import read_annotations
from src.utils.helpers import save_img, save_gt
from src.utils.mixins import SimpleCropBboxesOutOfFramesMixin, TransformFrameMixin


class WildTrackCropper(SimpleCropBboxesOutOfFramesMixin, TransformFrameMixin):
    def __init__(
        self,
        root: str,
        split: str,
        transforms: albumentations.Compose,
        person_reshape_h: int,
        person_reshape_w: int,
        dataset_save_dir: str,
    ):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.person_reshape_h, self.person_reshape_w = person_reshape_h, person_reshape_w
        self.dataset_save_dir = dataset_save_dir

        self.annotations: list[dict[int, list[dict[str, Any]]]] = []

        self._set_samples()

    def _set_samples(self):
        self.annotations = read_annotations(self.root, self.split)

    def crop_and_save(self):
        frame_ids = [int(Path(png_file).stem) for png_file in
                     os.listdir(os.path.join(self.root, self.split, "Image_subsets", "C1"))]
        frame_ids.sort()

        save_annotation_path = os.path.join(self.dataset_save_dir, self.split, "annotations_positions")
        os.makedirs(save_annotation_path, exist_ok=True)

        video_labels, video_save_ids = [], []

        tbar = tqdm(frame_ids, desc="Crop and save bboxes")
        for frame_id in tbar:
            for camera_id in range(7):
                save_image_path = os.path.join(self.dataset_save_dir, self.split, "Image_subsets")
                os.makedirs(save_image_path, exist_ok=True)

                frame_annotations = self.annotations[camera_id].get(frame_id, [])
                if not frame_annotations:
                    continue

                frame_path = os.path.join(self.root, self.split, "Image_subsets", f"C{camera_id + 1}", f"{frame_id:08d}.png")
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                boxes, labels, save_ids = [], [], []

                for ann in frame_annotations:
                    x, y, w, h = ann["bbox"]
                    boxes.append([x, y, x + w, y + h])
                    labels.append(int(ann["person_id"]))
                    save_ids.append(ann["idx"])

                frame, boxes, labels = self.transform_frame(frame, boxes, labels, error_message=f"({camera_id}, {frame_id})")
                crops = self.process_frame(frame=frame, bboxes=boxes)

                video_labels.extend(labels)
                video_save_ids.extend(save_ids)

                # save crops
                for save_id, crop in zip(save_ids, crops):
                    frame_path = os.path.join(save_image_path, f"{save_id:08d}.png")
                    save_img(frame_path, crop)

        # save gt
        gt_path = os.path.join(save_annotation_path, "annotation.txt")
        save_gt(gt_path, video_save_ids, video_labels)
