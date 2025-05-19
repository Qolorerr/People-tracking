import cv2
import numpy as np
from torch import Tensor


def save_img(dst: str, img: Tensor):
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dst, img_np, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def save_gt(dst: str, save_ids: list[int], labels: list[int]):
    with open(dst, "w") as f:
        for save_id, label in zip(save_ids, labels):
            f.write(f"{save_id}, {label}\n")
