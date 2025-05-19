from typing import Any

import numpy as np
import torch
import motmetrics as mm
from torch import Tensor


class TrackValidator:
    def __init__(self):
        self.mot_accum = mm.MOTAccumulator(auto_id=True)

    def reset(self) -> None:
        self.mot_accum = mm.MOTAccumulator(auto_id=True)

    def validate_frame(
        self, pred_tracks: list[dict[str, Any]], true_bboxes: Tensor, true_labels: Tensor
    ) -> None:
        pred_ids = [t["track_id"] for t in pred_tracks]
        true_ids = true_labels.cpu().numpy().astype(int)

        if len(pred_tracks) > 0 and len(true_bboxes) > 0:
            pred_boxes = torch.stack([t["bbox"] for t in pred_tracks]).cpu().numpy()
            true_boxes = true_bboxes.cpu().numpy()
            dist_matrix = mm.distances.iou_matrix(pred_boxes, true_boxes)
        else:
            dist_matrix = np.empty((0, 0))

        self.mot_accum.update(
            [int(id) for id in pred_ids], [int(id) for id in true_ids], dist_matrix
        )

    def get_metrics(self) -> dict[str, Any]:
        mh = mm.metrics.create()
        metrics = {
            "MOTA": "mota",
            "MOTP": "motp",
            "IDF1": "idf1",
            "FN": "num_misses",
            "FP": "num_false_positives",
            "id_switches": "num_switches",
        }
        metrics_values = mh.compute(self.mot_accum, metrics=metrics.values(), name="acc")

        return {key: metrics_values[value].iloc[0] for key, value in metrics.items()}
