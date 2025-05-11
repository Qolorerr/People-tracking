import albumentations
from torch import Tensor, LongTensor
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transforms: albumentations.Compose,
    ):
        self.root = root
        self.split = split
        self.samples: list[tuple[int, int]] = []

        is_test = self.split == "test"
        self._set_samples(is_test)
        self.transforms = transforms

    def _set_samples(self, test: bool = False):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> (Tensor, Tensor, Tensor, LongTensor, Tensor):
        raise NotImplementedError

    def __repr__(self) -> str:
        fmt_str = f"Dataset: {self.__class__.__name__}\n"
        fmt_str += f"\t# data: {self.__len__()}\n"
        fmt_str += f"\tSplit: {self.split}\n"
        fmt_str += f"\tRoot: {self.root}"
        return fmt_str
