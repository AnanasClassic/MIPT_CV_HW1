import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from src.data_utils import DataConfig, build_datasets, parse_normalization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset sanity checks.")
    parser.add_argument("--data-dir", default="data", help="Dataset root path.")
    parser.add_argument("--output-dir", default="artifacts/data_sanity")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--mean", default=None, help="Comma-separated mean values.")
    parser.add_argument("--std", default=None, help="Comma-separated std values.")
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--compute-stats",
        action="store_true",
        help="Compute dataset mean and std from training data.",
    )
    return parser.parse_args()


def class_counts(targets: list[int], classes: list[str]) -> Dict[str, int]:
    counts = {name: 0 for name in classes}
    for target in targets:
        counts[classes[target]] += 1
    return counts


def save_counts(counts: Dict[str, int], output_path: str) -> None:
    df = pd.DataFrame(
        {"class": list(counts.keys()), "count": list(counts.values())}
    )
    df.to_csv(output_path, index=False)


def save_grid(dataset, output_path: str, max_images: int) -> None:
    count = min(max_images, len(dataset))
    images = [dataset[i][0] for i in range(count)]
    grid = make_grid(images, nrow=int(np.sqrt(count)), normalize=True)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compute_mean_std(loader: DataLoader) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    total_pixels = 0

    for images, _ in loader:
        batch_pixels = images.size(0) * images.size(2) * images.size(3)
        total_pixels += batch_pixels
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_sq += (images**2).sum(dim=[0, 2, 3])

    mean = channel_sum / total_pixels
    std = torch.sqrt(channel_sum_sq / total_pixels - mean**2)
    return tuple(mean.tolist()), tuple(std.tolist())


def main() -> int:
    args = parse_args()
    mean, std = parse_normalization(args.mean, args.std)
    os.makedirs(args.output_dir, exist_ok=True)

    config = DataConfig(
        data_dir=args.data_dir,
        image_size=args.image_size,
        mean=mean,
        std=std,
    )
    train_dataset, val_dataset = build_datasets(config)

    train_counts = class_counts(train_dataset.targets, train_dataset.classes)
    val_counts = class_counts(val_dataset.targets, val_dataset.classes)

    save_counts(train_counts, f"{args.output_dir}/train_counts.csv")
    save_counts(val_counts, f"{args.output_dir}/val_counts.csv")

    save_grid(train_dataset, f"{args.output_dir}/train_grid.png", args.max_images)
    save_grid(val_dataset, f"{args.output_dir}/val_grid.png", args.max_images)

    if args.compute_stats:
        stats_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
            ]
        )
        stats_dataset = datasets.ImageFolder(
            f"{args.data_dir}/train", transform=stats_transform
        )
        stats_loader = DataLoader(
            stats_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        mean, std = compute_mean_std(stats_loader)
        stats_df = pd.DataFrame(
            {"channel": ["r", "g", "b"], "mean": mean, "std": std}
        )
        stats_df.to_csv(f"{args.output_dir}/train_stats.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
