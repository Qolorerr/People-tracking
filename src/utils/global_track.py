from collections import defaultdict

import torch
from torch import Tensor

from src.base import BaseTrack
from .track import Track


class GlobalTrack(BaseTrack):
    def __init__(
        self,
        camera_id: int,
        frame_idx: int,
        global_track_id: int,
        local_track: Track,
        track_length_vis: int = 25,
    ):
        super().__init__(
            track_id=global_track_id,
            bbox=local_track.get_state(),
            track_length_vis=track_length_vis,
        )

        self.checked_cameras: set[int] = set()

        self.local_track_map: dict[int, Track] = {}

        self.last_frame_idx = frame_idx
        self.global_history: defaultdict[int, list[tuple[int, int, int]]] = defaultdict(list)

        self.update(camera_id, frame_idx, local_track)

    def iterate(self, frame_idx: int) -> None:
        self.time_since_update = max(frame_idx - self.last_frame_idx, 0)

    def update(self, camera_id: int, frame_idx: int, local_track: Track, *args, **kwargs) -> None:
        self.checked_cameras.add(camera_id)

        self.local_track_map[camera_id] = local_track

        self.last_frame_idx = frame_idx
        self.time_since_update = 0
        self.hits += 1
        self.global_history[camera_id].append(local_track.history[-1])

    def is_contains_track(self, camera_id: int, track_id: int) -> bool:
        return (
            camera_id in self.checked_cameras
            and self.local_track_map[camera_id].track_id == track_id
        )

    def get_features(self, camera_id: int) -> Tensor:
        if camera_id in self.checked_cameras:
            return self.local_track_map[camera_id].feature
        return torch.mean(torch.stack([local_track.feature for local_track in self.local_track_map.values()]), dim=0)

    def get_frames_to_vis(self, camera_id: int, curr_frame_idx: int) -> list[tuple[int, int, int]]:
        idx_from = curr_frame_idx - (self.track_length_vis - 1)
        return list(
            filter(
                lambda record: idx_from <= record[0] <= curr_frame_idx,
                self.global_history[camera_id],
            )
        )

    def clean(self):
        for camera_id, local_track in self.local_track_map.items():
            if local_track.deleted:
                self.checked_cameras.remove(camera_id)
                self.local_track_map.pop(camera_id)
        if len(self.checked_cameras) == 0:
            self.deleted = True
