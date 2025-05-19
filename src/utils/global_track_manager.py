import copy
from typing import Any, cast

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
import torch.nn.functional as F

from src.base import BaseTrackManager
from .track import Track
from .global_track import GlobalTrack


class GlobalTrackManager(BaseTrackManager):
    def __init__(
        self,
        camera_manager_instance: BaseTrackManager,
        tracklet_expiration: int = 25,
        match_threshold: float = 0.7,
        device="cuda",
    ):
        super().__init__(tracklet_expiration=tracklet_expiration, device=device)

        self.tracks = cast(list[GlobalTrack], self.tracks)
        self.match_threshold = match_threshold
        self.camera_manager_instance = camera_manager_instance
        self.camera_managers: dict[int, BaseTrackManager] = {}

    def add_camera(self, camera_id: int) -> None:
        new_manager = copy.deepcopy(self.camera_manager_instance)
        new_manager.reset()

        self.camera_managers[camera_id] = new_manager

    def update(
        self, camera_id: int, frame_idx: int, bboxes: Tensor, features: Tensor, *args, **kwargs
    ) -> list[dict[str, Any]]:
        if camera_id not in self.camera_managers:
            return []

        for track in self.tracks:
            track.iterate(frame_idx)

        self.camera_managers[camera_id].update(frame_idx, bboxes, features)

        local_active_tracks = self.camera_managers[camera_id].get_active_tracks()

        # type check
        for local_active_track in local_active_tracks:
            if not isinstance(local_active_track, Track):
                raise TypeError("Unknown track type")
        local_active_tracks = cast(list[Track], local_active_tracks)

        unmatched_global_tracks = set(range(len(self.tracks)))
        unmatched_local_tracks = set(range(len(local_active_tracks)))

        # search if already stored
        local_stored_idx_map = self._search_already_stored_tracks(camera_id, local_active_tracks)
        for local_track_idx, global_track_idx in local_stored_idx_map.items():
            self.tracks[global_track_idx].update(
                camera_id, frame_idx, local_active_tracks[local_track_idx]
            )
            unmatched_global_tracks.remove(global_track_idx)
            unmatched_local_tracks.remove(local_track_idx)

        unmatched_global_tracks = set([global_track_idx for global_track_idx in unmatched_global_tracks if camera_id not in self.tracks[global_track_idx].checked_cameras])

        # find matches based on appearance
        matches: list[tuple[int, int]] = self._find_matches(
            camera_id, local_active_tracks, list(unmatched_global_tracks), list(unmatched_local_tracks)
        )

        # update matched tracks
        for global_track_idx, local_track_idx in matches:
            local_track = local_active_tracks[local_track_idx]

            self.tracks[global_track_idx].update(camera_id, frame_idx, local_track)

            if global_track_idx in unmatched_global_tracks:
                unmatched_global_tracks.remove(global_track_idx)
            unmatched_local_tracks.remove(local_track_idx)

        # create new tracks for unmatched detections
        for local_track_idx in unmatched_local_tracks:
            local_track = local_active_tracks[local_track_idx]

            new_track = GlobalTrack(camera_id, frame_idx, self.next_id, local_track)
            self.tracks.append(new_track)
            self.next_id += 1

        self._clean()

        active_tracks = self.get_active_tracks_info(camera_id, frame_idx)

        return active_tracks

    def _search_already_stored_tracks(
        self, camera_id: int, local_active_tracks: list[Track]
    ) -> dict[int, int]:
        local_to_global_idx_map = {}

        for local_track_idx, local_track in enumerate(local_active_tracks):
            found_global_track_idx = -1
            for global_track_idx, global_track in enumerate(self.tracks):
                if global_track.is_contains_track(camera_id, local_track.track_id):
                    found_global_track_idx = global_track_idx
                    break

            if found_global_track_idx != -1:
                local_to_global_idx_map[local_track_idx] = found_global_track_idx
                continue

        return local_to_global_idx_map

    def _find_matches(
        self,
        camera_id: int,
        local_active_tracks: list[Track],
        unmatched_global_tracks: list[int],
        unmatched_local_tracks: list[int],
    ) -> list[tuple[int, int]]:
        if len(unmatched_global_tracks) == 0 or len(unmatched_local_tracks) == 0:
            matches: list[tuple[int, int]] = []
        else:
            # appearance similarity
            global_features: Tensor = torch.stack(
                [self.tracks[idx].get_features(camera_id) for idx in unmatched_global_tracks]
            )
            global_features_norm: Tensor = F.normalize(global_features, p=2, dim=1)
            local_features: Tensor = torch.stack(
                [local_active_tracks[idx].feature for idx in unmatched_local_tracks]
            )
            local_features_norm: Tensor = F.normalize(local_features, p=2, dim=1)
            appearance_sim: Tensor = torch.mm(global_features_norm, local_features_norm.t())
            cost_matrix: Tensor = 1 - appearance_sim

            # Hungarian algorithm
            cost_matrix_np = cost_matrix.cpu().detach().numpy()
            try:
                row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
                row_ind = torch.from_numpy(row_ind).to(self.device)
                col_ind = torch.from_numpy(col_ind).to(self.device)

                matches = []
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] <= self.match_threshold:
                        matches.append(
                            (
                                unmatched_global_tracks[r.item()],
                                unmatched_local_tracks[c.item()],
                            )
                        )

            except Exception as e:
                print("Got exception:", e)
                matches: list[tuple[int, int]] = []
        return matches

    def reset(self):
        super().reset()
        for manager in self.camera_managers.values():
            manager.reset()

    def get_active_tracks_info(
        self, camera_id: int, frame_idx: int, *args, **kwargs
    ) -> list[dict[str, Any]]:
        active_tracks = []
        for track in self.tracks:
            if (
                camera_id in track.checked_cameras
                and track.local_track_map[camera_id].time_since_update == 0
            ):
                local_track = track.local_track_map[camera_id]
                active_tracks.append(
                    {
                        "track_id": track.track_id,
                        "bbox": local_track.get_state(),
                        "feature": local_track.feature,
                        "hits": track.hits,
                        "age": local_track.age,
                        "history": track.get_frames_to_vis(camera_id, frame_idx),
                    }
                )

        return active_tracks

    def _clean(self) -> None:
        super()._clean()
        for track in self.tracks:
            track.clean()
