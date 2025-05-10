from typing import Any

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
import torch.nn.functional as F

from .tracklet_manager import TrackManager
from .global_tracklet import GlobalTrack


class GlobalTrackManager:
    def __init__(self,
                 tracklet_expiration: int = 25,
                 match_threshold: float = 0.7,
                 device='cuda'):
        self.tracks: list[GlobalTrack] = []
        self.next_global_id = 0
        self.tracklet_expiration = tracklet_expiration
        self.match_threshold = match_threshold
        self.camera_managers: dict[int, TrackManager] = {}
        self.device = device

    def update(self, camera_id: int, frame_idx: int, bboxes: Tensor, features: Tensor) -> list[dict[str, Any]]:
        if camera_id not in self.camera_managers:
            return []

        for track in self.tracks:
            track.iterate(frame_idx)

        local_active_tracks = self.camera_managers[camera_id].update(frame_idx, bboxes, features)

        unmatched_global_tracks = list(range(len(self.tracks)))
        unmatched_local_tracks = list(range(len(local_active_tracks)))

        # search if already stored
        for local_track_idx, local_track_info in enumerate(local_active_tracks):
            local_track_id = local_track_info['track_id']
            bbox = local_track_info['bbox']
            feature = local_track_info['feature']

            found_global_track_idx = -1
            for global_track_idx, global_track in enumerate(self.tracks):
                if global_track.is_contains_track(camera_id, local_track_id):
                    found_global_track_idx = global_track_idx
                    break

            if found_global_track_idx != -1:
                self.tracks[found_global_track_idx].update(camera_id, frame_idx, local_track_id, bbox, feature)
                unmatched_global_tracks.remove(found_global_track_idx)
                unmatched_local_tracks.remove(local_track_idx)
                continue

        # appearance similarity
        global_features: Tensor = torch.cat([self.tracks[idx].get_features() for idx in unmatched_global_tracks])
        global_features_norm: Tensor = F.normalize(global_features, p=2, dim=1)
        local_features: Tensor = torch.stack([local_active_tracks[idx]['feature'] for idx in unmatched_local_tracks])
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
                    matches.append((unmatched_global_tracks[r.item()], unmatched_local_tracks[c.item()]))

        except Exception as e:
            print("Got exception:", e)
            matches: list[tuple[int, int]] = []

        # update matched tracks
        for global_track_idx, local_track_idx in matches:
            local_track_id = local_active_tracks[local_track_idx]['track_id']
            bbox = local_active_tracks[local_track_idx]['bbox']
            feature = local_active_tracks[local_track_idx]['feature']

            self.tracks[global_track_idx].update(camera_id, frame_idx, local_track_id, bbox, feature)

            unmatched_global_tracks.remove(global_track_idx)
            unmatched_local_tracks.remove(local_track_idx)

        # create new tracks for unmatched detections
        for local_track_idx in unmatched_local_tracks:
            local_track_id = local_active_tracks[local_track_idx]['track_id']
            bbox = local_active_tracks[local_track_idx]['bbox']
            feature = local_active_tracks[local_track_idx]['feature']

            new_track = GlobalTrack(camera_id, frame_idx, self.next_global_id, local_track_id, bbox, feature)
            self.tracks.append(new_track)
            self.next_global_id += 1
