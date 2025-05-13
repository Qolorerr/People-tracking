from typing import Any

import torch
import cv2
import numpy as np
import matplotlib.colors as mcolors
from numpy._typing import NDArray
from torch import Tensor


class TrackVisualizer:
    def __init__(self):
        self.color_map: dict[int, tuple[int, int, int]] = self._create_color_map()
        self.dash_pattern: list[int] = [5, 5]  # [line length, gap length]

    @staticmethod
    def _create_color_map() -> dict[int, tuple[int, int, int]]:
        base_colors = list(mcolors.TABLEAU_COLORS.values())
        return {
            i: tuple(int(255 * c) for c in mcolors.to_rgb(color))
            for i, color in enumerate(base_colors)
        }

    @staticmethod
    def _tensor_to_numpy(img_tensor: Tensor) -> NDArray[np.uint8]:
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        return (img * 255).astype(np.uint8)

    @staticmethod
    def _numpy_to_tensor(img_np: NDArray[np.uint8]) -> Tensor:
        return torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0

    def _draw_dashed_line(
        self, img: NDArray[np.uint8], pt1, pt2, color: tuple[float, float, float], thickness=2
    ):
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
        dash_count = int(dist / sum(self.dash_pattern))

        for i in range(dash_count):
            start = i * sum(self.dash_pattern)
            end = start + self.dash_pattern[0]
            alpha = start / dist
            beta = end / dist
            p1 = (int(pt1[0] + (pt2[0] - pt1[0]) * alpha), int(pt1[1] + (pt2[1] - pt1[1]) * alpha))
            p2 = (int(pt1[0] + (pt2[0] - pt1[0]) * beta), int(pt1[1] + (pt2[1] - pt1[1]) * beta))
            cv2.line(img, p1, p2, color, thickness)

    def draw(self, image_tensor: Tensor, tracklets: list[dict[str, Any]]) -> Tensor:
        """
        Основная функция отрисовки
        Args:
            image_tensor: Тензор изображения [C, H, W]
            tracklets: Список треклетов {'id': person_id, 'uuid': track_id, 'frames': list[Frame]}
        Returns:
            Тензор изображения с визуализацией [C, H, W]
        """
        img_np = self._tensor_to_numpy(image_tensor)
        img_np = self.draw_np(img_np, tracklets)
        img_tensor = self._numpy_to_tensor(img_np)
        return img_tensor

    def draw_np(
        self, img_np: NDArray[np.uint8], tracklets: list[dict[str, Any]]
    ) -> NDArray[np.uint8]:
        """
        Основная функция отрисовки
        Args:
            img_np: NDArray изображения [C, H, W]
            tracklets: Список треклетов {'id': person_id, 'uuid': track_id, 'frames': list[Frame]}
        Returns:
            NDArray изображения с визуализацией [C, H, W]
        """
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        for tracklet in tracklets:
            person_id: int = tracklet["track_id"]
            history: list[tuple[int, int, int]] = tracklet["history"]
            bbox: Tensor = tracklet["bbox"]
            color: tuple[int, int, int] = self.color_map[person_id % len(self.color_map)]

            for i, record in enumerate(history):
                pt2: tuple[int, int] = record[1:]

                # Draw points
                if i == len(history) - 1:
                    cv2.circle(img_np, pt2, 5, color, -1)
                else:
                    cv2.circle(img_np, pt2, 3, color, -1)

                # Draw lines
                if i == 0:
                    continue

                prev_record: tuple[int, int, int] = history[i - 1]
                pt1: tuple[int, int] = prev_record[1:]

                is_dashed = abs(record[0] - prev_record[1]) > 1
                if is_dashed:
                    self._draw_dashed_line(img_np, pt1, pt2, color, 1)
                else:
                    cv2.line(img_np, pt1, pt2, color, 2)

            # Draw bbox
            x1, y1, x2, y2 = map(int, bbox.cpu().tolist())

            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)

            # Draw sign
            label = f"ID: {person_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_np, (x1, y1 - (h + 10)), (x1 + w, y1), color, -1)
            cv2.putText(
                img_np, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        return img_np
