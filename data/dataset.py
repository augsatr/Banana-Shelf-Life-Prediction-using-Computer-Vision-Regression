import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

from config import cfg


class BananaDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        targets: Optional[List[float]] = None,
        stage_labels: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        augment: Optional[Callable] = None,
        preprocessor: Optional[Callable] = None,
    ):
        self.image_paths = image_paths
        self.targets = targets
        self.stage_labels = stage_labels
        self.transform = transform
        self.augment = augment
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        with open(str(img_path), "rb") as f:
            raw = bytearray(f.read())
        arr = np.asarray(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.preprocessor:
            image = self.preprocessor(image)

        if self.augment:
            image = self.augment(image)

        if self.transform and hasattr(self.transform, "image"):
            augmented = self.transform(image=image)
            image = augmented["image"]
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image)
        elif self.transform:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
            image = torch.tensor(image)
        else:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
            image = torch.tensor(image)

        result = {"image": image, "path": str(img_path)}

        if self.targets is not None:
            result["shelf_life"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        if self.stage_labels is not None:
            result["stage"] = torch.tensor(self.stage_labels[idx], dtype=torch.long)

        return result


def create_dataloaders(
    image_paths: List[Path],
    targets: List[float],
    stage_labels: List[int],
    batch_size: int = None,
    val_split: float = None,
    test_split: float = None,
    preprocessor: Optional[Callable] = None,
    augment_fn: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if batch_size is None:
        batch_size = cfg.get("data", "batch_size", default=32)
    if val_split is None:
        val_split = cfg.get("data", "val_split", default=0.12)
    if test_split is None:
        test_split = cfg.get("data", "test_split", default=0.12)

    indices = list(range(len(image_paths)))
    train_idx, test_idx = train_test_split(indices, test_size=test_split, random_state=42)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_split / (1 - test_split), random_state=42
    )

    def _subset(idx_list, items):
        return [items[i] for i in idx_list]

    ds_kwargs = dict(preprocessor=preprocessor, transform=None)

    train_ds = BananaDataset(
        _subset(train_idx, image_paths),
        _subset(train_idx, targets),
        _subset(train_idx, stage_labels),
        augment=augment_fn,
        **ds_kwargs,
    )
    val_ds = BananaDataset(
        _subset(val_idx, image_paths),
        _subset(val_idx, targets),
        _subset(val_idx, stage_labels),
        **ds_kwargs,
    )
    test_ds = BananaDataset(
        _subset(test_idx, image_paths),
        _subset(test_idx, targets),
        _subset(test_idx, stage_labels),
        **ds_kwargs,
    )

    nw = cfg.get("data", "num_workers", default=0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    return train_loader, val_loader, test_loader
