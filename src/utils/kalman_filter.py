import torch
from torch import Tensor


class KalmanFilter:
    def __init__(self, device: str):
        self.device = device
        ndim, dt = 4, 1.0

        self._motion_mat = torch.eye(2 * ndim, 2 * ndim, device=device)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = torch.eye(ndim, 2 * ndim, device=device)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: Tensor) -> tuple[Tensor, Tensor]:
        mean_pos = measurement
        mean_vel = torch.zeros_like(mean_pos)
        mean = torch.cat([mean_pos, mean_vel])

        std = torch.tensor(
            [
                2 * self._std_weight_position * measurement[3],
                2 * self._std_weight_position * measurement[3],
                1e-2,
                2 * self._std_weight_position * measurement[3],
                10 * self._std_weight_velocity * measurement[3],
                10 * self._std_weight_velocity * measurement[3],
                1e-5,
                10 * self._std_weight_velocity * measurement[3],
            ],
            device=self.device,
        )

        covariance = torch.diag(std**2)
        return mean, covariance

    def predict(self, mean: Tensor, covariance: Tensor) -> tuple[Tensor, Tensor]:
        std_pos = torch.tensor(
            [
                self._std_weight_position * mean[3],
                self._std_weight_position * mean[3],
                1e-2,
                self._std_weight_position * mean[3],
            ],
            device=self.device,
        )

        std_vel = torch.tensor(
            [
                self._std_weight_velocity * mean[3],
                self._std_weight_velocity * mean[3],
                1e-5,
                self._std_weight_velocity * mean[3],
            ],
            device=self.device,
        )

        motion_cov = torch.diag(torch.cat([std_pos, std_vel]) ** 2)
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: Tensor, covariance: Tensor) -> tuple[Tensor, Tensor]:
        std = torch.tensor(
            [
                self._std_weight_position * mean[3],
                self._std_weight_position * mean[3],
                1e-1,
                self._std_weight_position * mean[3],
            ],
            device=self.device,
        )

        innovation_cov = torch.diag(std**2)
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return mean, covariance

    def update(
        self, mean: Tensor, covariance: Tensor, measurement: Tensor
    ) -> tuple[Tensor, Tensor]:
        projected_mean, projected_cov = self.project(mean, covariance)
        chol = torch.linalg.cholesky(projected_cov)

        # solve Kalman gain using Cholesky decomposition
        kalman_gain_transposed = torch.cholesky_solve((covariance @ self._update_mat.T).t(), chol)
        kalman_gain = kalman_gain_transposed.t()

        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.t()
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.t()
        return new_mean, new_covariance
