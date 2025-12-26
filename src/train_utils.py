from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.metrics import (
    accuracy_from_confusion,
    batch_metrics,
    confusion_matrix,
    macro_f1_from_confusion,
)


@dataclass
class EpochStats:
    loss: float
    accuracy: float
    macro_f1: float


def set_deterministic(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def save_checkpoint(state: Dict, output_dir: str, name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    torch.save(state, path)
    return path


def log_param_histograms(writer: SummaryWriter, model: torch.nn.Module, epoch: int) -> None:
    for name, param in model.named_parameters():
        writer.add_histogram(f"weights/{name}", param.detach().cpu(), epoch)
        if param.grad is not None:
            writer.add_histogram(f"grads/{name}", param.grad.detach().cpu(), epoch)


def train_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
    log_interval: int = 50,
) -> EpochStats:
    model.train()
    total_loss = 0.0
    total_samples = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        acc, f1 = batch_metrics(logits.detach(), targets, num_classes)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        preds = logits.detach().argmax(dim=1)
        cm += confusion_matrix(
            targets.detach().cpu(), preds.detach().cpu(), num_classes
        )

        if writer is not None and step % log_interval == 0:
            global_step = epoch * len(loader) + step
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/accuracy", acc, global_step)
            writer.add_scalar("train/macro_f1", f1, global_step)
            writer.add_scalar("train/lr", get_lr(optimizer), global_step)

    return EpochStats(
        loss=total_loss / max(1, total_samples),
        accuracy=accuracy_from_confusion(cm),
        macro_f1=macro_f1_from_confusion(cm),
    )


def eval_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> EpochStats:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            preds = logits.argmax(dim=1)
            cm += confusion_matrix(
                targets.detach().cpu(), preds.detach().cpu(), num_classes
            )

    return EpochStats(
        loss=total_loss / max(1, total_samples),
        accuracy=accuracy_from_confusion(cm),
        macro_f1=macro_f1_from_confusion(cm),
    )


def fit(
    model: torch.nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    epochs: int,
    output_dir: str,
    writer: Optional[SummaryWriter] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict[str, Iterable[EpochStats]]:
    best_metric = -float("inf")
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        train_stats = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes,
            writer=writer,
            epoch=epoch,
        )
        val_stats = eval_epoch(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
        )

        history["train"].append(train_stats)
        history["val"].append(val_stats)

        if writer is not None:
            writer.add_scalar("epoch/train_loss", train_stats.loss, epoch)
            writer.add_scalar("epoch/train_accuracy", train_stats.accuracy, epoch)
            writer.add_scalar("epoch/train_macro_f1", train_stats.macro_f1, epoch)
            writer.add_scalar("epoch/val_loss", val_stats.loss, epoch)
            writer.add_scalar("epoch/val_accuracy", val_stats.accuracy, epoch)
            writer.add_scalar("epoch/val_macro_f1", val_stats.macro_f1, epoch)
            log_param_histograms(writer, model, epoch)

        metric = val_stats.macro_f1
        is_best = metric > best_metric
        if is_best:
            best_metric = metric

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_macro_f1": val_stats.macro_f1,
            },
            output_dir,
            f"checkpoint_epoch_{epoch}.pt",
        )
        if is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_macro_f1": val_stats.macro_f1,
                },
                output_dir,
                "checkpoint_best.pt",
            )

        if scheduler is not None:
            scheduler.step()

    return history
