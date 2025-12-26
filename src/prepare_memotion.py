import argparse
import csv
import json
import os
import random
import time
import urllib.request
from collections import Counter
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Memotion dataset.")
    parser.add_argument("--output-dir", default="data/memotion")
    parser.add_argument("--repo", default="AshuReddy/memetion_dataset_7k")
    parser.add_argument("--label-field", default="overall_sentiment")
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-per-class", type=int, default=None)
    return parser.parse_args()


def download_file(url: str, dest_path: str, timeout: int) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(
        dest_path, "wb"
    ) as out:
        out.write(resp.read())


def class_counts(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = Counter()
    for value in values:
        if value:
            counts[value] += 1
    return counts


def select_classes(
    counts: Dict[str, int], min_count: int, num_classes: int
) -> List[str]:
    candidates = [name for name, count in counts.items() if count >= min_count]
    candidates = sorted(candidates, key=lambda name: (-counts[name], name))
    return candidates[:num_classes]


def split_items(
    items: List[str], train_ratio: float, rng: random.Random
) -> Tuple[List[str], List[str]]:
    values = list(items)
    rng.shuffle(values)
    split_idx = int(len(values) * train_ratio)
    return values[:split_idx], values[split_idx:]


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    labels_path = os.path.join("data", "raw", "memotion_labels.csv")
    if not os.path.exists(labels_path):
        labels_url = (
            f"https://huggingface.co/datasets/{args.repo}/resolve/main/"
            "memotion_dataset_7k/labels.csv"
        )
        download_file(labels_url, labels_path, args.timeout)

    image_names: List[str] = []
    labels: List[str] = []
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_names.append(row["image_name"])
            labels.append(row[args.label_field])

    counts = class_counts(labels)
    classes = select_classes(counts, args.min_count, args.num_classes)
    if len(classes) < args.num_classes:
        raise RuntimeError("Not enough classes match min_count.")

    by_class: Dict[str, List[str]] = {name: [] for name in classes}
    for name, label in zip(image_names, labels, strict=False):
        if label in by_class:
            by_class[label].append(name)

    for class_name, items in by_class.items():
        if args.max_per_class is not None:
            items = items[: args.max_per_class]
        train_items, val_items = split_items(items, args.train_ratio, rng)
        for split_name, split_items_list in (("train", train_items), ("val", val_items)):
            for image_name in split_items_list:
                url = (
                    f"https://huggingface.co/datasets/{args.repo}/resolve/main/"
                    f"memotion_dataset_7k/images/{image_name}"
                )
                dest_path = os.path.join(
                    args.output_dir, split_name, class_name, image_name
                )
                download_file(url, dest_path, args.timeout)
                time.sleep(0.01)

    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"label_field": args.label_field, "classes": classes}, f, indent=2
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
