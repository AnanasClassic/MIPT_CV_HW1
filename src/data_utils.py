from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataConfig:
    data_dir: str
    image_size: int = 224
    mean: Tuple[float, float, float] = IMAGENET_MEAN
    std: Tuple[float, float, float] = IMAGENET_STD


def get_transforms(
    image_size: int,
    mean: Iterable[float],
    std: Iterable[float],
) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, val_transform


def build_datasets(config: DataConfig):
    train_transform, val_transform = get_transforms(
        config.image_size, config.mean, config.std
    )
    train_dir = f"{config.data_dir}/train"
    val_dir = f"{config.data_dir}/val"
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    return train_dataset, val_dataset


def parse_normalization(
    mean: Optional[str],
    std: Optional[str],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    def parse_triplet(value: str) -> Tuple[float, float, float]:
        parts = [float(part.strip()) for part in value.split(",")]
        if len(parts) != 3:
            raise ValueError("Expected three comma-separated values.")
        return parts[0], parts[1], parts[2]

    mean_tuple = parse_triplet(mean) if mean is not None else IMAGENET_MEAN
    std_tuple = parse_triplet(std) if std is not None else IMAGENET_STD
    return mean_tuple, std_tuple
