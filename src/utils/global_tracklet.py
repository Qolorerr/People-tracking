import torch
from torch import Tensor


class GlobalTrack:
    def __init__(self, camera_id: int, frame_idx: int, global_track_id: int, local_track_id: int, bbox: Tensor, feature: Tensor, track_length_vis: int = 25):
        self.checked_cameras: set[int] = set()

        self.track_id = global_track_id
        self.track_id_map: dict[int, int] = {}
        self.device = bbox.device

        self.feature_map: dict[int, Tensor] = {}
        self.last_frame_idx = frame_idx
        self.time_since_update = 0
        self.hits = 0
        self.global_history: dict[int, list[tuple[int, int, int]]] = {}
        self.track_length_vis = track_length_vis

        self.update(camera_id, frame_idx, local_track_id, bbox, feature)

    def iterate(self, frame_idx: int) -> None:
        self.time_since_update = max(frame_idx - self.last_frame_idx, 0)

    def update(self, camera_id: int, frame_idx: int, track_id: int, bbox: Tensor, feature: Tensor) -> None:
        x1, y1, x2, y2 = bbox
        cx: Tensor = (x1 + x2) / 2
        cy: Tensor = (y1 + y2) / 2

        self.track_id_map[camera_id] = track_id

        if camera_id not in self.checked_cameras:
            # update appearance feature with EMA
            self.feature_map[camera_id] = 0.9 * self.feature_map[camera_id] + 0.1 * feature
            self.global_history[camera_id].append((frame_idx, int(cx.item()), int(cy.item())))
        else:
            self.checked_cameras.add(camera_id)

            self.feature_map[camera_id] = feature
            self.global_history[camera_id] = [(frame_idx, int(cx.item()), int(cy.item()))]

        self.last_frame_idx = frame_idx
        self.time_since_update = 0
        self.hits += 1

    def is_contains_track(self, camera_id: int, track_id: int) -> bool:
        return camera_id in self.checked_cameras and self.track_id_map[camera_id] == track_id

    def get_features(self) -> Tensor:
        return torch.stack([feature for feature in self.feature_map.values()])

    # def get_state(self) -> Tensor:
    #     cx, cy, s, r = self._mean[:4]
    #     w = torch.sqrt(torch.abs(s * r))
    #     h = torch.sqrt(torch.abs(s / r))
    #     x1 = cx - w / 2
    #     y1 = cy - h / 2
    #     x2 = cx + w / 2
    #     y2 = cy + h / 2
    #     return torch.stack([x1, y1, x2, y2])

    # def get_frames_to_vis(self, curr_frame_idx: int) -> list[tuple[int, int, int]]:
    #     idx_from = curr_frame_idx - (self.track_length_vis - 1)
    #     return list(filter(lambda record: idx_from <= record[0] <= curr_frame_idx, self.global_history))
