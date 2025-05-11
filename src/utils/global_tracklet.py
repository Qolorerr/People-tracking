import torch
from torch import Tensor

from src.base import BaseTrack
from src.utils import Track


class GlobalTrack(BaseTrack):
    def __init__(self, camera_id: int, frame_idx: int, global_track_id: int, local_track: Track, track_length_vis: int = 25):
        super().__init__(track_id=global_track_id, bbox=local_track.get_state(), track_length_vis=track_length_vis)

        self.checked_cameras: set[int] = set()

        self.local_track_map: dict[int, Track] = {}

        self.last_frame_idx = frame_idx

        self.update(camera_id, frame_idx, local_track)

    def iterate(self, frame_idx: int) -> None:
        self.time_since_update = max(frame_idx - self.last_frame_idx, 0)

    def update(self, camera_id: int, frame_idx: int, local_track: Track) -> None:
        self.local_track_map[camera_id] = local_track

        self.last_frame_idx = frame_idx
        self.time_since_update = 0
        self.hits += 1

    def is_contains_track(self, camera_id: int, track_id: int) -> bool:
        return camera_id in self.checked_cameras and self.local_track_map[camera_id].track_id == track_id

    def get_features(self) -> Tensor:
        return torch.stack([local_track.feature for local_track in self.local_track_map.values()])
