import argparse
import csv
import os
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.data_utils import DataConfig, build_datasets, parse_normalization
from src.metrics import accuracy_from_confusion, confusion_matrix, macro_f1_from_confusion
from src.models import SmallCnn, build_vit_tiny
from src.train_utils import seed_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--data-dir", default="data", help="Path to dataset root.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--output-dir", default="artifacts", help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["cnn", "vit"], required=True)
    parser.add_argument("--mean", default=None, help="Comma-separated mean values.")
    parser.add_argument("--std", default=None, help="Comma-separated std values.")
    return parser.parse_args()


def load_model(model_name: str, num_classes: int) -> torch.nn.Module:
    if model_name == "cnn":
        return SmallCnn(num_classes=num_classes)
    if model_name == "vit":
        return build_vit_tiny(num_classes=num_classes, pretrained=False)
    raise ValueError("Unsupported model name.")


def save_confusion_matrix(
    cm: torch.Tensor, classes: list[str], output_dir: str, model_name: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"confusion_{model_name}.csv")
    png_path = os.path.join(output_dir, f"confusion_{model_name}.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + classes)
        for name, row in zip(classes, cm.tolist(), strict=False):
            writer.writerow([name] + row)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def append_metrics_row(path: str, row: dict) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    mean, std = parse_normalization(args.mean, args.std)
    config = DataConfig(
        data_dir=args.data_dir, image_size=args.image_size, mean=mean, std=std
    )
    _, val_dataset = build_datasets(config)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, num_classes=len(val_dataset.classes))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    cm = torch.zeros(
        (len(val_dataset.classes), len(val_dataset.classes)), dtype=torch.int64
    )
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            cm += confusion_matrix(
                targets.detach().cpu(), preds.detach().cpu(), len(val_dataset.classes)
            )

    acc = accuracy_from_confusion(cm)
    f1 = macro_f1_from_confusion(cm)

    save_confusion_matrix(cm, val_dataset.classes, args.output_dir, args.model)

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    row = {
        "model": args.model,
        "split": "val",
        "accuracy": f"{acc:.6f}",
        "macro_f1": f"{f1:.6f}",
        "checkpoint": args.checkpoint,
    }
    append_metrics_row(metrics_path, row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
