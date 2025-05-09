import itertools
import logging
import math
import os
import time
from datetime import datetime
from typing import Any, cast

import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn, Tensor, LongTensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO

from src.utils import TrackVisualizer, MetricsMeter, metrics, TrackManager, TrackValidator, CropBboxesOutOfFramesMixin


class Trainer(CropBboxesOutOfFramesMixin):
    def __init__(self,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 accelerator: Accelerator,
                 detection_model: YOLO | None,
                 feature_extractor_model: nn.Module,
                 config: DictConfig,
                 resume: str,
                 ):
        self.train_dataloader: DataLoader = train_dataloader
        self.val_dataloader: DataLoader = val_dataloader
        self.accelerator: Accelerator = accelerator
        self.device = self.accelerator.device
        self.detection_model: YOLO | None = detection_model
        self.feature_extractor_model: nn.Module = feature_extractor_model
        self.config: DictConfig = config

        self.tracklet_master: TrackManager = instantiate(config.tracklet_master, device=self.device)
        self.visualizer = TrackVisualizer()
        self.tracklet_validator = TrackValidator()

        self.detection_model, self.feature_extractor_model = self.accelerator.prepare(
            self.detection_model,
            self.feature_extractor_model
        )
        self.detection_model = cast(YOLO, self.detection_model)

        cfg_trainer = self.config["trainer"]
        self.start_epoch = 1
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.do_validation = cfg_trainer["val"]
        self.val_per_epochs = cfg_trainer["val_per_epochs"]
        self.tune_per_epochs = cfg_trainer["tune_per_epochs"]
        self.tuner_params = cfg_trainer.get("tuner_params", {})

        # LOSS
        self.loss: nn.Module = instantiate(config.loss, num_classes=1024, use_gpu=self.device == "cuda")

        # OPTIMIZER
        self.feature_extractor_model.train()
        trainable_params = self.feature_extractor_model.parameters()
        self.optimizer: Optimizer = instantiate(self.config.optimizer, trainable_params)
        self.lr_scheduler: LRScheduler = instantiate(self.config.lr_scheduler, self.optimizer)

        self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.optimizer,
            self.lr_scheduler,
        )

        # MONITORING
        self.monitor = cfg_trainer.get("monitor", "off")
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]
            self.mnt_best = -math.inf if self.mnt_mode == "max" else math.inf
            self.early_stoping = cfg_trainer.get("early_stop", math.inf)
            self.not_improved_count = 0

        # CHECKPOINTS & TENSORBOARD
        start_time = datetime.now().strftime("%m-%d_%H-%M")
        writer_dir = str(
            os.path.join(cfg_trainer["log_dir"], self.config["name"], start_time)
        )
        self.writer = tensorboard.SummaryWriter(writer_dir)
        info_to_write = [
            "crop_size",
            "test_loader",
            "detection_model",
            "feature_extractor",
            "tester",
        ]
        for info in info_to_write:
            self.writer.add_text(f"info/{info}", str(self.config[info]))
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metric_store = MetricsMeter(writer=self.writer)
        self.checkpoint_dir = os.path.join(
            cfg_trainer["save_dir"], self.config["name"], start_time
        )
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.wrt_mode, self.wrt_step = "train", 0
        self.log_per_iter: int = cfg_trainer.get("log_per_iter", 1)
        self.log_step = int(self.log_per_iter / self.train_dataloader.batch_size) + 1

        self.confidence_threshold = self.config["confidence_threshold"]
        self.person_reshape_h, self.person_reshape_w = self.config["person_reshape_h"], self.config["person_reshape_w"]

        if resume:
            self._resume_checkpoint(resume)

        self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.train_dataloader,
            self.val_dataloader
        )

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.epochs + 1):
            results = self._train_epoch(epoch)
            log = {"epoch": epoch, **results}

            if self.do_validation and epoch % self.tune_per_epochs == 0:
                tunable_params = self._get_params()

                with torch.no_grad():
                    results = self._tune_val_epoch(epoch)

                self.logger.info(f"\n## Fine tuning on epoch {epoch} ## ")
                for param_name, before_value in tunable_params.items():
                    after_value = before_value
                    if param_name in results:
                        after_value = results[param_name]
                    self.logger.info(f"{str(param_name):25s}: {before_value:.3f} -> {after_value:.3f}")

            if self.do_validation and epoch % self.val_per_epochs == 0:
                with torch.no_grad():
                    results = self._val_epoch(epoch)

                lr_metric = (results["MOTA"] + results["IDF1"]) / 2
                self.update_lr(epoch=epoch, metric=lr_metric)

                # LOGGING INFO
                self.logger.info(f"\n## Info for epoch {epoch} ## ")
                for k, v in results.items():
                    self.logger.info(f"{str(k):15s}: {v}")

                ok = self._monitor_ok(log)
                if not ok:
                    break

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False)

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.set_model_mode('train')

        end_time = time.time()
        self.metric_store.reset()

        self._calc_wrt_step(epoch, len(self.train_dataloader), 0)

        tbar = tqdm(self.train_dataloader, desc="Train")
        for batch_idx, data in enumerate(tbar):
            self.metric_store.update({"load time": time.time() - end_time})

            loss_summary = self.forward_backward(data)
            self.metric_store.update({"proc time": time.time() - end_time})
            self.metric_store.update(loss_summary)

            if batch_idx % self.log_step == 0:
                self.metric_store.update({"lr": ("{:.6f}", self.get_current_lr())})
                self.metric_store.print_metrics(tbar, epoch, "TRAIN")

            self.metric_store.log_metrics(self.wrt_mode, self.wrt_step)

            end_time = time.time()
            self._calc_wrt_step(epoch, len(self.train_dataloader), batch_idx)

        return self.metric_store.get_metrics_as_dict()

    def _load_params(self, params: dict[str, Any]) -> None:
        self.confidence_threshold = params["confidence_threshold"]
        self.tracklet_master.motion_weight = params["motion_weight"]
        self.tracklet_master.appearance_weight = params["appearance_weight"]
        self.tracklet_master.match_threshold = params["match_threshold"]

    def _get_params(self) -> dict[str, Any]:
        return {
            "confidence_threshold": self.confidence_threshold,
            "motion_weight": self.tracklet_master.motion_weight,
            "appearance_weight": self.tracklet_master.appearance_weight,
            "match_threshold": self.tracklet_master.match_threshold,
        }

    def _tune_val_epoch(self, epoch: int) -> dict[str, float]:
        self.set_model_mode('eval')
        self.metric_store.reset()
        self._calc_wrt_step(0, len(self.val_dataloader), 0)

        best_score = -1
        best_params = self._get_params()
        updated = False
        backup = best_params.copy()

        # First tune
        if epoch == self.tune_per_epochs:
            self.metric_store.update(best_params)
            self.metric_store.log_metrics(self.wrt_mode, self.wrt_step)

        confidence_thresholds = torch.clamp(torch.linspace(
            self.confidence_threshold - self.tuner_params.params.confidence_threshold.radius,
            self.confidence_threshold + self.tuner_params.params.confidence_threshold.radius,
            self.tuner_params.params.confidence_threshold.steps),
            min=self.tuner_params.min_l,
            max=self.tuner_params.max_r,
        )
        motion_weights = torch.clamp(torch.linspace(
            self.tracklet_master.motion_weight - self.tuner_params.params.motion_weight.radius,
            self.tracklet_master.motion_weight + self.tuner_params.params.motion_weight.radius,
            self.tuner_params.params.motion_weight.steps),
            min=self.tuner_params.min_l,
            max=self.tuner_params.max_r,
        )
        match_thresholds = torch.clamp(torch.linspace(
            self.tracklet_master.match_threshold - self.tuner_params.params.match_threshold.radius,
            self.tracklet_master.match_threshold + self.tuner_params.params.match_threshold.radius,
            self.tuner_params.params.match_threshold.steps),
            min=self.tuner_params.min_l,
            max=self.tuner_params.max_r,
        )

        param_ranges = [
            confidence_thresholds.tolist(),
            motion_weights.tolist(),
            match_thresholds.tolist()
        ]

        checked: set[tuple] = set()

        total_combinations = torch.prod(torch.tensor([len(r) for r in param_ranges])).item()
        try:
            with tqdm(total=total_combinations, desc="Hyperparameter Search") as tbar:
                for params in itertools.product(*param_ranges):
                    if params in checked:
                        tbar.update(1)
                        continue
                    checked.add(params)

                    self.confidence_threshold, self.tracklet_master.motion_weight, self.tracklet_master.match_threshold = params
                    self.tracklet_master.appearance_weight = 1 - self.tracklet_master.motion_weight

                    self.tracklet_validator.reset()

                    for frame_idx, data in enumerate(self.val_dataloader):
                        self.evaluate(frame_idx, data)

                    self.tracklet_master.reset()

                    val_metrics = self.tracklet_validator.get_metrics()
                    score = 0.5 * val_metrics['MOTA'] + 0.5 * val_metrics['IDF1']

                    if score > best_score:
                        best_score = score
                        best_params = self._get_params()
                        updated = True

                    tbar.update(1)

        except Exception as e:
            print("Exception during tuning:", e)
            print("Loading backup")
            self._load_params(backup)

        if not updated:
            best_params = backup
        self._load_params(best_params)

        self.tuner_params.params.confidence_threshold.radius *= self.tuner_params.alpha
        self.tuner_params.params.motion_weight.radius *= self.tuner_params.alpha
        self.tuner_params.params.match_threshold.radius *= self.tuner_params.alpha

        self._calc_wrt_step(epoch // self.tune_per_epochs, len(self.val_dataloader), 0)
        self.metric_store.update(best_params)
        self.metric_store.log_metrics(self.wrt_mode, self.wrt_step)

        return best_params

    def _val_epoch(self, epoch: int) -> dict[str, float]:
        self.set_model_mode('eval')

        end_time = time.time()
        self.metric_store.reset()
        self.tracklet_validator.reset()

        self._calc_wrt_step(epoch // self.val_per_epochs, len(self.val_dataloader), 0)

        tbar = tqdm(self.val_dataloader, desc="Val")
        for frame_idx, data in enumerate(tbar):
            self.metric_store.update({"load time": time.time() - end_time})

            active_tracks = self.evaluate(frame_idx, data)
            self.metric_store.update({"proc time": time.time() - end_time})

            if frame_idx % self.log_per_iter == 0:
                self.metric_store.print_metrics(tbar, epoch, "VAL")
                self._visualize_frame(data[0], active_tracks)

            self.metric_store.log_metrics(self.wrt_mode, self.wrt_step)

            end_time = time.time()
            self._calc_wrt_step(epoch // self.val_per_epochs, len(self.val_dataloader), frame_idx)

        self.metric_store.reset()
        self.metric_store.update(self.tracklet_validator.get_metrics())
        self.metric_store.log_metrics(self.wrt_mode, self.wrt_step)

        return self.metric_store.get_metrics_as_dict()

    def set_model_mode(self, mode: str = "train") -> None:
        self.wrt_mode = mode
        if mode == "train":
            self.feature_extractor_model.train()
        else:
            self.detection_model.training = False
            self.feature_extractor_model.eval()

    def process_frame(self, frame: Tensor) -> dict[str, Tensor]:
        with torch.no_grad():
            detections = self.detection_model.predict(frame, verbose=False)[0]

        bboxes: Tensor = detections.boxes.xyxy
        confs: Tensor = detections.boxes.conf
        class_ids: Tensor = detections.boxes.cls

        person_mask = class_ids == 0
        confidence_mask = confs >= self.confidence_threshold
        valid_bbox_mask = (bboxes[:, 0] != bboxes[:, 2]) & (bboxes[:, 1] != bboxes[:, 3])
        mask = person_mask & confidence_mask & valid_bbox_mask
        bboxes = bboxes[mask]
        confs = confs[mask]

        features = []
        if len(bboxes) > 0:
            batch = self.crop_bboxes(frame, bboxes)

            features = self.feature_extractor_model(batch)

        return {
            'bboxes': bboxes,
            'confs': confs,
            'features': features
        }

    @staticmethod
    def _filter_degenerate_bboxes(bboxes: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        valid_mask = (bboxes[:, 0] != bboxes[:, 2]) & (bboxes[:, 1] != bboxes[:, 3])
        filtered_bboxes = bboxes[valid_mask]
        filtered_labels = labels[valid_mask]
        return filtered_bboxes, filtered_labels

    def forward_backward(self, data: tuple[Tensor, Tensor, LongTensor, Tensor]) -> dict[str, float]:
        frames, _, labels, _ = data

        self.optimizer.zero_grad()

        outputs = self.feature_extractor_model(frames)
        loss = self.loss(outputs, labels)

        self.accelerator.backward(loss)
        self.optimizer.step()

        loss_summary = {
            'loss': loss.item(),
            'acc': metrics.accuracy(outputs, labels)[0].item()
        }

        return loss_summary

    def evaluate(self, frame_idx: int,
                 data: tuple[Tensor, Tensor, LongTensor, Tensor]) -> list[dict[str, Any]]:
        frame, true_bboxes, true_labels, is_new_video = data

        true_bboxes, true_labels = self._filter_degenerate_bboxes(bboxes=true_bboxes.squeeze(0),
                                                                  labels=true_labels.squeeze(0))

        if is_new_video.item():
            self.tracklet_master.reset()

        detections = self.process_frame(frame)
        active_tracks = self.tracklet_master.update(frame_idx=frame_idx,
                                                    bboxes=detections['bboxes'],
                                                    features=detections['features'])

        self.tracklet_validator.validate_frame(pred_tracks=active_tracks,
                                               true_bboxes=true_bboxes,
                                               true_labels=true_labels)

        return active_tracks

    def _calc_wrt_step(self, epoch: int, batch_size: int, batch_idx: int) -> None:
        if batch_idx % self.log_step == 0:
            self.wrt_step = (epoch - 1) * batch_size + batch_idx

    def _monitor_ok(self, log: dict[str, Any]) -> bool:
        if self.mnt_mode == "off":
            return True
        try:
            if self.mnt_mode == "min":
                improved = log[self.mnt_metric] < self.mnt_best
            else:
                improved = log[self.mnt_metric] > self.mnt_best
        except KeyError:
            self.logger.warning(
                f"The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops."
            )
            return False

        if improved:
            self.mnt_best = log[self.mnt_metric]
            self.not_improved_count = 0
        else:
            self.not_improved_count += 1

        if self.not_improved_count > self.early_stoping:
            self.logger.info(
                f"\nPerformance didn't improve for {self.early_stoping} epochs"
            )
            self.logger.warning("Training Stopped")
            return False
        return True

    def _visualize_frame(self, frames: Tensor, active_tracks: list[dict[str, Any]]) -> None:
        if self.wrt_step % self.log_step != 0:
            return

        if len(frames.shape) == 4:
            img = frames[0]
        else:
            img = frames

        img = self.visualizer.draw(img, active_tracks)
        self.writer.add_image(
            tag=f"{self.wrt_mode}/tracklet_predictions",
            img_tensor=img,
            global_step=self.wrt_step,
            dataformats="CHW",
        )

    def update_lr(self, metric: float = 0.0, epoch: int | None = None):
        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
            self.lr_scheduler.step(metrics=metric, epoch=epoch)
        else:
            self.lr_scheduler.step()

    def get_current_lr(self) -> float:
        return self.optimizer.param_groups[-1]['lr']

    def _save_checkpoint(self, epoch: int, save_best=False) -> None:
        state = {
            "arch": type(self.feature_extractor_model).__name__,
            "epoch": epoch,
            "state_dict": self.feature_extractor_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = os.path.join(self.checkpoint_dir, f"checkpoint-epoch{epoch}.pth")
        self.logger.info(f"\nSaving a checkpoint: {filename} ...")
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, f"best_model.pth")
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth")

    def _resume_checkpoint(self, resume_path: str) -> None:
        self.logger.info(f"Loading checkpoint : {resume_path}")
        checkpoint = torch.load(resume_path)

        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]
        self.not_improved_count = 0

        if checkpoint["arch"] != type(self.feature_extractor_model).__name__:
            self.logger.warning(
                {"Warning! Current model is not the same as the one in the checkpoint"}
            )
        self.feature_extractor_model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
