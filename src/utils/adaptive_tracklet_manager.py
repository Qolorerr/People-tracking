from typing import Any

import numpy as np
import torch
from torch import Tensor

from .tracklet import Track
from .metrics import AdaptiveMetrics
from .tracklet_manager import TrackManager


class AdaptiveTrackManager(TrackManager):
    def __init__(self, adapt_params: dict[str, Any], adapt_per_step: int = 30, **kwargs):
        super().__init__(**kwargs)

        self.metrics = AdaptiveMetrics()
        self.adapt_params = adapt_params
        self.adapt_per_step = adapt_per_step

    def update(self, frame_idx: int, bboxes: Tensor, features: Tensor) -> list[dict[str, Any]]:
        if not isinstance(bboxes, Tensor):
            bboxes = torch.tensor(bboxes)
        if not isinstance(features, Tensor):
            features = torch.tensor(features)

        for track in self.tracks:
            track.predict()

        matches, unmatched_detections = self._find_matches(bboxes, features)
        pred_bboxes = (
            torch.stack([t.get_state() for t in self.tracks])
            if len(self.tracks) > 0
            else torch.empty(0)
        )

        features_matches = {}

        # update matched tracks
        for r, c in matches:
            self.tracks[r].update(bboxes[c], features[c], frame_idx)
            features_matches[self.tracks[r].track_id] = features[c]

        # create new tracks for unmatched detections
        for d in unmatched_detections:
            new_track = Track(frame_idx, self.next_id, bboxes[d], features[d])
            self.tracks.append(new_track)
            self.next_id += 1

        deleted_tracks_count = self._clean()

        self.metrics.update(
            matches=matches,
            frame_detections=bboxes,
            track_predictions=pred_bboxes,
            frame_features=features_matches,
            deleted_tracks_count=deleted_tracks_count,
        )

        if self.adapt_per_step and frame_idx % self.adapt_per_step == 0:
            self._adapt_self_params()

        active_tracks = self.get_active_tracks_info(frame_idx)

        return active_tracks

    def _adapt_self_params(self) -> None:
        if "adapt_weights" in self.adapt_params:
            self._adapt_weights()

    def _adapt_weights(self) -> None:
        active_tracks_count = self._get_active_tracks_count()
        if active_tracks_count == 0:
            # no detections, freeze weights
            return

        params: dict[str, float] = self.adapt_params["adapt_weights"]

        delta: float = params.get("delta", 0.01)
        prediction_error_th: float = params.get("prediction_error_th", 0.5)
        feature_consistency_th: float = params.get("feature_consistency_th", 0.7)
        min_motion_w: float = params.get("min_motion_w", 0.1)
        max_motion_w: float = params.get("max_motion_w", 0.9)

        avg_error = np.mean([m["prediction_error"] for m in self.metrics.window])
        avg_consistency = np.mean([m["feature_consistency"] for m in self.metrics.window])

        motion_w = self.motion_weight
        if avg_error > prediction_error_th:
            motion_w = max(min_motion_w, motion_w - delta)
        elif avg_consistency < feature_consistency_th:
            motion_w = min(max_motion_w, motion_w + delta)

        self.motion_weight = motion_w
        self.appearance_weight = 1 - motion_w

    def _get_active_tracks_count(self) -> int:
        return len([t for t in self.tracks if t.time_since_update == 0])

    def _clean(self) -> int:
        before = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.tracklet_expiration]
        after = len(self.tracks)

        return before - after

    def reset(self):
        super().reset()
        self.metrics.reset()

    def get_metrics(self) -> dict[str, float]:
        return self.metrics.current
