"""Exercise the image-only pipeline on macOS without third-party packages."""
import argparse
import csv
import math
import random
import struct
import subprocess
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

CANCER = {"BCC", "SCC", "MEL"}
SIZE = 16


def read_pixels(source, thumbnail):
    subprocess.run(
        ["sips", "-s", "format", "bmp", "-z", str(SIZE), str(SIZE),
         str(source), "--out", str(thumbnail)],
        check=True, capture_output=True,
    )
    bitmap = thumbnail.read_bytes()
    offset = struct.unpack_from("<I", bitmap, 10)[0]
    width, height, planes, bits, compression = struct.unpack_from("<iiHHI", bitmap, 18)
    if bitmap[:2] != b"BM" or (width, abs(height), planes) != (SIZE, SIZE, 1):
        raise ValueError(f"Unexpected BMP dimensions for {source.name}")
    if (bits, compression) not in {(24, 0), (32, 0), (32, 3)}:
        raise ValueError(f"Unsupported BMP pixel format for {source.name}")
    if compression == 3 and struct.unpack_from("<III", bitmap, 54) != (
        0xFF0000, 0xFF00, 0xFF
    ):
        raise ValueError(f"Unexpected BMP channel masks for {source.name}")
    stride = ((width * bits + 31) // 32) * 4
    line_order = range(SIZE) if height < 0 else reversed(range(SIZE))
    pixels = []
    for line in line_order:
        for column in range(SIZE):
            start = offset + line * stride + column * (bits // 8)
            blue, green, red = bitmap[start:start + 3]
            pixels.extend((channel / 255 - 0.5 for channel in (red, green, blue)))
    return pixels


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def probability(weights, pixels):
    score = sum(w * x for w, x in zip(weights[:-1], pixels)) + weights[-1]
    return 1 / (1 + math.exp(-max(-35, min(35, score))))


def lesion_examples(keys, lesions, image_paths, thumbnail):
    examples = []
    for key in keys:
        rows = lesions[key]
        label = int(rows[0]["diagnostic"] in CANCER)
        images = [read_pixels(image_paths[row["img_id"]], thumbnail) for row in rows]
        examples.append((images, label))
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, nargs="+")
    parser.add_argument("--folds", type=Path, required=True)
    args = parser.parse_args()

    folds = {row["patient_id"]: int(row["fold"]) for row in read_csv(args.folds)}
    images = {}
    for directory in args.images:
        for path in directory.rglob("*.png"):
            if path.name in images:
                raise ValueError(f"Duplicate image name: {path.name}")
            images[path.name] = path
    lesions = defaultdict(list)
    for row in read_csv(args.metadata):
        lesions[row["patient_id"], row["lesion_id"]].append(row)

    rng = random.Random(42)
    partitions = {"train": [], "validation": []}
    for key in sorted(lesions):
        fold = folds[key[0]]
        if fold != 1:
            partitions["validation" if fold == 2 else "train"].append(key)
    rng.shuffle(partitions["train"])
    rng.shuffle(partitions["validation"])
    partitions["validation"].sort(key=lambda key: len(lesions[key]) == 1)
    train_keys = partitions["train"][:48]
    validation_keys = partitions["validation"][:24]
    assert not ({key[0] for key in train_keys} & {key[0] for key in validation_keys})
    assert any(len(lesions[key]) > 1 for key in validation_keys)

    with TemporaryDirectory() as directory:
        thumbnail = Path(directory) / "image.bmp"
        train = lesion_examples(train_keys, lesions, images, thumbnail)
        validation = lesion_examples(validation_keys, lesions, images, thumbnail)

    weights = [0.0] * (SIZE * SIZE * 3 + 1)
    for _ in range(6):
        rng.shuffle(train)
        for image_list, label in train:
            for pixels in image_list:
                step = 0.03 * (label - probability(weights, pixels)) / len(image_list)
                for index, pixel in enumerate(pixels):
                    weights[index] += step * pixel
                weights[-1] += step

    predictions = [
        (sum(probability(weights, pixels) for pixels in image_list) / len(image_list), label)
        for image_list, label in validation
    ]
    assert len(predictions) == len(validation_keys)
    assert all(math.isfinite(score) and 0 < score < 1 for score, _ in predictions)
    print(f"Train: {len(train)} lesions, {sum(len(items) for items, _ in train)} images")
    print(f"Validation: {len(predictions)} lesion-level predictions from "
          f"{sum(len(items) for items, _ in validation)} images")
    print("Patient overlap: 0. Fold 1 unused. All predictions finite.")


if __name__ == "__main__":
    main()
