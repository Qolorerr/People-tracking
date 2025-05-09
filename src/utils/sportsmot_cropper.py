import os
from typing import Any

import albumentations
import cv2
import numpy as np
import torch
from torch import Tensor
import torchvision.transforms.functional as F
from tqdm import tqdm


class SportsMOTCropper:
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
        with open(split_file, 'r') as f:
            self.video_names = [line.strip() for line in f.readlines()]

        self.annotations: list[dict[int, list[dict[str, Any]]]] = []

        self._set_samples()

    def _set_samples(self):
        self._max_num_pids = 0
        for video_idx, video_name in enumerate(self.video_names):
            video_path = os.path.join(self.root, self.split, video_name)

            bbox_idx = 1

            annotations = {}
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
                        'idx': bbox_idx,
                    })
                    bbox_idx += 1
            self._max_num_pids = max(len(pids), self._max_num_pids)

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
                    x, y, w, h = ann['bbox']
                    boxes.append([x, y, x + w, y + h])
                    labels.append(self._max_num_pids * video_idx + int(ann['person_id']))
                    save_ids.append(ann['idx'])

                boxes = np.array(boxes, dtype=np.int64) if boxes else \
                    np.zeros((0, 4), dtype=np.int64)
                labels = np.array(labels, dtype=np.int64) if labels else \
                    np.zeros((0,), dtype=np.int64)

                try:
                    transformed = self.transforms(image=frame, bboxes=boxes, class_labels=labels)
                except Exception as e:
                    print("Exception:", (video_idx, frame_id))
                    raise e
                frame, boxes, labels = transformed["image"], transformed["bboxes"], transformed["class_labels"]
                frame = frame.float() / 255.0
                boxes = torch.from_numpy(boxes).to(torch.int64)
                crops = self.process_frame(frame=frame, bboxes=boxes)

                video_labels.extend(labels)
                video_save_ids.extend(save_ids)

                # save crops
                for save_id, crop in zip(save_ids, crops):
                    frame_path = os.path.join(save_video_path, "img1", f"{save_id:06d}.jpg")
                    self.save_img(frame_path, crop)

            # save gt
            gt_path = os.path.join(save_video_path, "gt", "gt.txt")
            self.save_gt(gt_path, video_save_ids, video_labels)

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

    @staticmethod
    def save_img(dst: str, img: Tensor):
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dst, img_np, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    @staticmethod
    def save_gt(dst: str, save_ids: list[int], labels: list[int]):
        with open(dst, 'w') as f:
            for save_id, label in zip(save_ids, labels):
                f.write(f"{save_id}, {label}\n")
