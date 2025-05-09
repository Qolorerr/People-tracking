from typing import Any

import torch
from torch import Tensor

from .tracklet import Track
from .metrics import AdaptiveMetrics
from .tracklet_manager import TrackManager


class AdaptiveTrackManager(TrackManager):
    def __init__(self, adapt_per_step: int = 30, **kwargs):
        super().__init__(**kwargs)

        self.metrics = AdaptiveMetrics()
        self.adapt_per_step = adapt_per_step

    def update(self, frame_idx: int, bboxes: Tensor, features: Tensor) -> list[dict[str, Any]]:
        if not isinstance(bboxes, Tensor):
            bboxes = torch.tensor(bboxes)
        if not isinstance(features, Tensor):
            features = torch.tensor(features)

        for track in self.tracks:
            track.predict()

        matches, unmatched_detections = self._find_matches(bboxes, features)
        pred_bboxes = torch.stack([t.get_state() for t in self.tracks]) if len(self.tracks) > 0 else torch.empty(0)

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

        self.metrics.update(matches=matches,
                            frame_detections=bboxes,
                            track_predictions=pred_bboxes,
                            frame_features=features_matches,
                            deleted_tracks_count=deleted_tracks_count)

        active_tracks = []
        for track in self.tracks:
            if track.time_since_update == 0:
                active_tracks.append({
                    'track_id': track.track_id,
                    'bbox': track.get_state(),
                    'feature': track.feature,
                    'hits': track.hits,
                    'age': track.age,
                    'history': track.get_frames_to_vis(frame_idx)
                })

        return active_tracks

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
