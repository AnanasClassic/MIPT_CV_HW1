import argparse
import io
import json
import os
import time
import urllib.request

import pyarrow.parquet as pq
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Oxford-IIIT Pet dataset.")
    parser.add_argument("--output-dir", default="data/oxford_pets")
    parser.add_argument("--repo", default="timm/oxford-iiit-pet")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-per-class", type=int, default=None)
    return parser.parse_args()


def download_file(url: str, dest_path: str, timeout: int) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(
        dest_path, "wb"
    ) as out:
        out.write(resp.read())


def extract_image_bytes(value) -> bytes:
    if isinstance(value, dict):
        data = value.get("bytes")
        if data:
            return data
    raise RuntimeError("Missing image bytes in dataset.")


def get_label_names(schema) -> list[str]:
    field = schema.field("label")
    label_type = field.type
    names = getattr(label_type, "names", None)
    if names:
        return list(names)
    metadata = schema.metadata or {}
    raw = metadata.get(b"huggingface")
    if raw:
        info = json.loads(raw.decode("utf-8", errors="ignore"))
        features = info.get("info", {}).get("features")
        if isinstance(features, dict):
            label = features.get("label", {})
            class_label = label.get("class_label", {})
            names = class_label.get("names")
            if names:
                return list(names.values()) if isinstance(names, dict) else list(names)
        if isinstance(features, list):
            for feature in features:
                if feature.get("name") == "label":
                    label = feature.get("dtype", {}).get("class_label", {})
                    names = label.get("names")
                    if names:
                        return list(names.values()) if isinstance(names, dict) else list(names)
    return []


def save_split(
    parquet_path: str,
    output_dir: str,
    split_name: str,
    max_per_class: int | None,
) -> None:
    parquet = pq.ParquetFile(parquet_path)
    label_names = get_label_names(parquet.schema_arrow)
    per_class: dict[str, int] = {}
    index = 0

    for batch in parquet.iter_batches(columns=["image", "label"], batch_size=64):
        images = batch.column(0).to_pylist()
        labels = batch.column(1).to_pylist()
        for image_value, label_idx in zip(images, labels, strict=False):
            class_name = (
                label_names[label_idx] if label_names else str(label_idx)
            )
            current = per_class.get(class_name, 0)
            if max_per_class is not None and current >= max_per_class:
                continue
            image_bytes = extract_image_bytes(image_value)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            filename = f"{index}.jpg"
            dest_path = os.path.join(output_dir, split_name, class_name, filename)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            image.save(dest_path, format="JPEG", quality=95)
            per_class[class_name] = current + 1
            index += 1


def main() -> int:
    args = parse_args()

    train_url = (
        f"https://huggingface.co/datasets/{args.repo}/resolve/main/data/"
        "train-00000-of-00001.parquet"
    )
    test_url = (
        f"https://huggingface.co/datasets/{args.repo}/resolve/main/data/"
        "test-00000-of-00001.parquet"
    )
    train_path = os.path.join("data", "raw", "oxford_pets_train.parquet")
    test_path = os.path.join("data", "raw", "oxford_pets_test.parquet")

    if not os.path.exists(train_path):
        download_file(train_url, train_path, args.timeout)
    if not os.path.exists(test_path):
        download_file(test_url, test_path, args.timeout)

    save_split(train_path, args.output_dir, "train", args.max_per_class)
    time.sleep(0.1)
    save_split(test_path, args.output_dir, "val", args.max_per_class)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
