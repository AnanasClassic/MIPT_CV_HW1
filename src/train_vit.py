import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Tuple

import torch
from torch import nn
from torch import profiler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data_utils import DataConfig, build_datasets, parse_normalization
from src.models import build_vit_tiny
from src.train_utils import fit, seed_worker, set_deterministic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ViT-Tiny linear probe.")
    parser.add_argument("--data-dir", default="data", help="Path to dataset root.")
    parser.add_argument("--output-dir", default="artifacts/vit", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--mean", default=None, help="Comma-separated mean values.")
    parser.add_argument("--std", default=None, help="Comma-separated std values.")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=80)
    parser.add_argument("--profile-warmup", type=int, default=10)
    parser.add_argument("--profile-only", action="store_true")
    return parser.parse_args()


def make_loaders(
    data_dir: str,
    image_size: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    batch_size: int,
    num_workers: int,
    seed: int,
):
    config = DataConfig(
        data_dir=data_dir,
        image_size=image_size,
        mean=mean,
        std=std,
    )
    train_dataset, val_dataset = build_datasets(config)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_profile(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: str,
    num_steps: int,
    warmup_steps: int,
) -> None:
    activities = [profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(profiler.ProfilerActivity.CUDA)

    schedule = profiler.schedule(
        wait=0, warmup=warmup_steps, active=num_steps, repeat=1
    )
    trace_handler = profiler.tensorboard_trace_handler(output_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    model.train()
    with profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        step = 0
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            prof.step()
            step += 1
            if step >= num_steps + warmup_steps:
                break


def main() -> int:
    args = parse_args()
    set_deterministic(args.seed)

    mean, std = parse_normalization(args.mean, args.std)
    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset, val_dataset, train_loader, val_loader = make_loaders(
        args.data_dir,
        args.image_size,
        mean,
        std,
        args.batch_size,
        args.num_workers,
        args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vit_tiny(
        num_classes=len(train_dataset.classes), pretrained=True, freeze_backbone=True
    )
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    run_name = args.run_name or f"vit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    if args.profile:
        profiler_dir = os.path.join("artifacts", "profiler", "vit")
        run_profile(
            model,
            train_loader,
            device,
            profiler_dir,
            args.profile_steps,
            args.profile_warmup,
        )
        if args.profile_only:
            return 0

    history = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        len(train_dataset.classes),
        args.epochs,
        args.output_dir,
        writer=writer,
    )

    writer.close()

    config_payload = {
        "data_dir": args.data_dir,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "mean": mean,
        "std": std,
        "num_classes": len(train_dataset.classes),
        "classes": train_dataset.classes,
        "model": "vit_tiny_patch16_224",
    }
    metrics_payload = {
        "train": asdict(history["train"][-1]),
        "val": asdict(history["val"][-1]),
    }

    save_json(os.path.join(args.output_dir, "config.json"), config_payload)
    save_json(os.path.join(args.output_dir, "metrics.json"), metrics_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
