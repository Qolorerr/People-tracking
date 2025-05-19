from torch import Tensor


class BaseTrack:
    def __init__(self, track_id: int, bbox: Tensor, track_length_vis: int = 25):
        self.track_id = track_id
        self.device = bbox.device

        self.time_since_update = 0
        self.hits = 0
        self.track_length_vis = track_length_vis

        self.deleted = False

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError
