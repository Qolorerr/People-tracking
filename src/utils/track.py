import torch
from torch import Tensor

from src.base import BaseTrack
from .kalman_filter import KalmanFilter


class Track(BaseTrack):
    def __init__(
        self,
        frame_idx: int,
        track_id: int,
        bbox: Tensor,
        feature: Tensor,
        track_length_vis: int = 25,
    ):
        super().__init__(track_id=track_id, bbox=bbox, track_length_vis=track_length_vis)

        self.start_frame_idx = frame_idx

        # convert bbox to [cx, cy, area, aspect_ratio] format
        x1, y1, x2, y2 = bbox
        w: Tensor = x2 - x1
        h: Tensor = y2 - y1
        cx: Tensor = (x1 + x2) / 2
        cy: Tensor = (y1 + y2) / 2
        s: Tensor = w * h
        r: Tensor = w / h

        self.kf = KalmanFilter(self.device)
        self._mean, self._covariance = self.kf.initiate(
            torch.tensor([cx, cy, s, r], device=self.device)
        )

        self.feature = feature
        self.hits = 1
        self.age = 0
        self.history: list[tuple[int, int, int]] = [(frame_idx, int(cx.item()), int(cy.item()))]

    def predict(self) -> None:
        self._mean, self._covariance = self.kf.predict(self._mean, self._covariance)
        self.age += 1
        self.time_since_update += 1

    def update(
        self, bbox: Tensor, feature: Tensor, frame_idx: int | None = None, *args, **kwargs
    ) -> None:
        x1, y1, x2, y2 = bbox
        w: Tensor = x2 - x1
        h: Tensor = y2 - y1
        cx: Tensor = (x1 + x2) / 2
        cy: Tensor = (y1 + y2) / 2
        s: Tensor = w * h
        r: Tensor = w / h

        measurement = torch.tensor([cx, cy, s, r], device=self.device)
        self._mean, self._covariance = self.kf.update(self._mean, self._covariance, measurement)

        # update appearance feature with EMA
        self.feature = 0.9 * self.feature + 0.1 * feature
        self.time_since_update = 0
        self.hits += 1
        if frame_idx is None:
            frame_idx = self.start_frame_idx + self.age
        self.history.append((frame_idx, int(cx.item()), int(cy.item())))

    def get_state(self) -> Tensor:
        cx, cy, s, r = self._mean[:4]
        w = torch.sqrt(torch.abs(s * r))
        h = torch.sqrt(torch.abs(s / r))
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2])

    def get_frames_to_vis(self, curr_frame_idx: int) -> list[tuple[int, int, int]]:
        idx_from = curr_frame_idx - (self.track_length_vis - 1)
        return list(filter(lambda record: idx_from <= record[0] <= curr_frame_idx, self.history))
