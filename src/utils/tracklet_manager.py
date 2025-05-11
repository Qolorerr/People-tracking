from typing import Any, cast

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
import torch.nn.functional as F

from .tracklet import Track
from .metrics import compute_iou_batch
from src.base import BaseTrackManager


class TrackManager(BaseTrackManager):
    def __init__(self,
                 tracklet_expiration: int = 25,
                 motion_weight: float = 0.5,
                 appearance_weight: float = 0.5,
                 match_threshold: float = 0.7,
                 device='cuda'):
        super().__init__(tracklet_expiration=tracklet_expiration, device=device)

        self.tracks = cast(list[Track], self.tracks)
        self.motion_weight = motion_weight
        self.appearance_weight = appearance_weight
        self.match_threshold = match_threshold

    def update(self, frame_idx: int, bboxes: Tensor, features: Tensor) -> list[dict[str, Any]]:
        if not isinstance(bboxes, Tensor):
            bboxes = torch.tensor(bboxes)
        if not isinstance(features, Tensor):
            features = torch.tensor(features)

        for track in self.tracks:
            track.predict()

        matches, unmatched_detections = self._find_matches(bboxes, features)

        # update matched tracks
        for r, c in matches:
            self.tracks[r].update(bboxes[c], features[c], frame_idx)

        # create new tracks for unmatched detections
        for d in unmatched_detections:
            new_track = Track(frame_idx, self.next_id, bboxes[d], features[d])
            self.tracks.append(new_track)
            self.next_id += 1

        self._clean()

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

    # returns list of matches and list of unmatched detections
    def _find_matches(self, bboxes: Tensor, features: Tensor) -> tuple[list[tuple[int, int]], list[int]]:
        if len(self.tracks) == 0 or len(bboxes) == 0:
            matches: list[tuple[int, int]] = []
            unmatched_detections: list[int] = list(range(len(bboxes)))
        else:
            # Kalman filter based similarity
            pred_bboxes: Tensor = torch.stack([t.get_state() for t in self.tracks])
            iou_matrix: Tensor = compute_iou_batch(pred_bboxes, bboxes)
            motion_cost: Tensor = 1 - iou_matrix

            # appearance similarity
            track_features: Tensor = torch.stack([t.feature for t in self.tracks])
            track_features_norm: Tensor = F.normalize(track_features, p=2, dim=1)
            det_features_norm: Tensor = F.normalize(features, p=2, dim=1)
            appearance_sim: Tensor = torch.mm(track_features_norm, det_features_norm.t())
            appearance_cost: Tensor = 1 - appearance_sim

            cost_matrix: Tensor = self.motion_weight * motion_cost + self.appearance_weight * appearance_cost

            # Hungarian algorithm
            cost_matrix_np = cost_matrix.cpu().detach().numpy()
            try:
                row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
                row_ind = torch.from_numpy(row_ind).to(self.device)
                col_ind = torch.from_numpy(col_ind).to(self.device)

                matches = []
                unmatched_detections = list(range(len(bboxes)))

                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] <= self.match_threshold:
                        matches.append((r.item(), c.item()))
                        unmatched_detections.remove(c.item())
            except Exception as e:
                print("Got exception:", e)
                matches: list[tuple[int, int]] = []
                unmatched_detections: list[int] = list(range(len(bboxes)))
        return matches, unmatched_detections

    @staticmethod
    def _compute_association(pred_tracks: list[dict[str, Any]],
                             true_bboxes: Tensor,
                             true_labels: Tensor) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if len(pred_tracks) == 0 or len(true_bboxes) == 0:
            return [], [], []

        pred_bboxes = torch.stack([t['bbox'] for t in pred_tracks])

        iou_matrix = compute_iou_batch(pred_bboxes, true_bboxes)

        cost_matrix = 1 - iou_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())

        matches = []
        unmatched_pred = []
        unmatched_true = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] > 0.5:  # standard VOC threshold
                matches.append((pred_tracks[r]['track_id'], true_labels[c].item()))
            else:
                unmatched_pred.append(pred_tracks[r]['track_id'])
                unmatched_true.append(true_labels[c].item())

        return matches, unmatched_pred, unmatched_true
