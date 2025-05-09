import torch
from torch import Tensor
from torch.utils import tensorboard
from tqdm import tqdm


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def average_str(self) -> str:
        fmtstr = "{name} {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class MetricsMeter:
    def __init__(self, writer: tensorboard.SummaryWriter):
        self.writer = writer
        self.reset()

    def log_metrics(self, wrt_mode: str, wrt_step: int) -> None:
        for metric, value in self.total_metrics.items():
            self.writer.add_scalar(f"{wrt_mode}/{metric}", value[0].avg, wrt_step)

    def print_metrics(
        self,
        tbar: tqdm,
        epoch: int,
        mode: str = "TRAIN",
        **kwargs,
    ) -> None:
        message = "{} ({}) | "
        message = message.format(mode, epoch)

        metrics: list[str] = []
        for metric, value in self.total_metrics.items():
            metrics.append(("{}: " + value[1]).format(metric, value[0].avg))
        message += ", ".join(metrics)

        for key, value in kwargs.items():
            if value is not None:
                new_arg = ", {}: {:.3f}"
                new_arg = new_arg.format(key, value)
                message += new_arg
        message += " |"
        tbar.set_description(message)

    def reset(self) -> None:
        self.total_metrics: dict[str, tuple[AverageMeter, str]] = {}

    def update(self, metrics: dict[str, float | tuple[str, float]]) -> None:
        for metric, value in metrics.items():
            value_format = "{:.3f}"
            if isinstance(value, tuple):
                value_format, value = value

            if metric not in self.total_metrics:
                self.total_metrics[metric] = (AverageMeter(metric), value_format)
            self.total_metrics[metric][0].update(value)

    def get_metric(self, name: str) -> float | None:
        if name in self.total_metrics:
            return self.total_metrics[name][0].avg
        return None

    def get_metrics_as_dict(self) -> dict[str, float]:
        return {metric: value[0].avg for metric, value in self.total_metrics.items()}


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


def compute_iou_batch(bboxes1: Tensor, bboxes2: Tensor):
    x1_min = torch.maximum(bboxes1[:, 0].unsqueeze(1), bboxes2[:, 0].unsqueeze(0))
    y1_min = torch.maximum(bboxes1[:, 1].unsqueeze(1), bboxes2[:, 1].unsqueeze(0))
    x2_max = torch.minimum(bboxes1[:, 2].unsqueeze(1), bboxes2[:, 2].unsqueeze(0))
    y2_max = torch.minimum(bboxes1[:, 3].unsqueeze(1), bboxes2[:, 3].unsqueeze(0))

    intersection_area = torch.clamp(x2_max - x1_min, min=0) * torch.clamp(y2_max - y1_min, min=0)
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection_area

    iou = intersection_area / union_area
    iou[union_area == 0] = 0.0
    return iou
