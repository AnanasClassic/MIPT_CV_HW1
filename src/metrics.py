from __future__ import annotations

from typing import Tuple

import torch


def confusion_matrix(
    targets: torch.Tensor, preds: torch.Tensor, num_classes: int
) -> torch.Tensor:
    targets = targets.view(-1).to(torch.int64)
    preds = preds.view(-1).to(torch.int64)
    indices = targets * num_classes + preds
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    return cm.view(num_classes, num_classes)


def accuracy_from_confusion(cm: torch.Tensor) -> float:
    correct = cm.diag().sum().item()
    total = cm.sum().item()
    return correct / max(1, total)


def macro_f1_from_confusion(cm: torch.Tensor) -> float:
    cm = cm.to(torch.float32)
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    denom = 2 * tp + fp + fn
    f1 = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(denom))
    return f1.mean().item()


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).float().sum().item()
    return correct / max(1, targets.numel())


def macro_f1(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=1)
    cm = confusion_matrix(targets, preds, num_classes)
    return macro_f1_from_confusion(cm)


def batch_metrics(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> Tuple[float, float]:
    acc = accuracy(logits, targets)
    f1 = macro_f1(logits, targets, num_classes)
    return acc, f1
