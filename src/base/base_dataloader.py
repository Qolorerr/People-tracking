from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):
    def __init__(self, camera_ids: list[int] | None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.camera_ids: list[int] = []
        if camera_ids is None:
            self.camera_ids.append(0)
        else:
            self.camera_ids = camera_ids
