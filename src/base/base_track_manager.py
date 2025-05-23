from typing import Any

from .base_track import BaseTrack


class BaseTrackManager:
    def __init__(self, tracklet_expiration: int = 25, device="cuda"):
        self.tracks: list[BaseTrack] = []
        self.next_id = 0
        self.tracklet_expiration = tracklet_expiration
        self.device = device

    def update(self, *args, **kwargs) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _clean(self) -> None:
        for track in self.tracks:
            if track.time_since_update > self.tracklet_expiration:
                track.deleted = True
        self.tracks = [t for t in self.tracks if not t.deleted]

    def reset(self):
        self.tracks = []
        self.next_id = 0

    def get_metrics(self) -> dict[str, float]:
        return {}

    def get_active_tracks(self) -> list[BaseTrack]:
        return list(filter(lambda t: t.time_since_update == 0, self.tracks))

    def get_active_tracks_info(self, *args, **kwargs) -> list[dict[str, Any]]:
        raise NotImplementedError
