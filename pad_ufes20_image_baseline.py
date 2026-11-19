"""Train and evaluate an image-only baseline on one fixed PAD-UFES-20 fold."""
import argparse
import csv
import math
from collections import Counter
from itertools import groupby
from pathlib import Path
from statistics import mean

CANCER = {"BCC", "SCC", "MEL"}
DIAGNOSES = CANCER | {"ACK", "NEV", "SEK"}


def load_records(metadata, folds, directories, held_out):
    with folds.open(newline="", encoding="utf-8-sig") as stream:
        assignments = {row["patient_id"]: int(row["fold"]) for row in csv.DictReader(stream)}

    images = {}
    for directory in directories:
        for path in directory.rglob("*.png"):
            if path.name in images:
                raise ValueError(f"Duplicate image name: {path.name}")
            images[path.name] = path

    train, test = [], []
    labels = {}
    seen_images = set()
    with metadata.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            patient, lesion, image = row["patient_id"], row["lesion_id"], row["img_id"]
            diagnosis = row["diagnostic"]
            if diagnosis not in DIAGNOSES or image in seen_images:
                raise ValueError(f"Unexpected diagnosis or duplicate metadata for {image}")
            seen_images.add(image)
            key = (patient, lesion)
            label = int(diagnosis in CANCER)
            if labels.setdefault(key, label) != label:
                raise ValueError(f"Inconsistent binary labels for lesion {key}")
            record = (images[image], key, label)
            (test if assignments[patient] == held_out else train).append(record)
    if not train or not test:
        raise ValueError("Both partitions must contain records")
    return train, test


def score_lesions(grouped):
    scores = sorted((mean(probabilities), label)
                    for probabilities, label in grouped.values())
    positives = sum(label for _, label in scores)
    negatives = len(scores) - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes in the held-out fold")
    lower_negatives = 0
    favorable_pairs = 0.0
    log_loss = 0.0
    for probability, tied in groupby(scores, key=lambda item: item[0]):
        labels = [label for _, label in tied]
        tied_positives = sum(labels)
        tied_negatives = len(labels) - tied_positives
        favorable_pairs += tied_positives * (lower_negatives + 0.5 * tied_negatives)
        lower_negatives += tied_negatives
        p = min(max(probability, 1e-7), 1 - 1e-7)
        log_loss -= tied_positives * math.log(p) + tied_negatives * math.log1p(-p)
    return favorable_pairs / (positives * negatives), log_loss / len(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--held-out", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import ResNet18_Weights, resnet18

    train, test = load_records(args.metadata, args.folds, args.images, args.held_out)
    torch.manual_seed(42)
    device_name = args.device
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable. Use --device cpu.")
    device = torch.device(device_name)
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    class Photos(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            path, _, label = self.records[index]
            with Image.open(path) as source:
                return transform(source.convert("RGB")), label

    train_loader = DataLoader(Photos(train), batch_size=32, shuffle=False)
    test_loader = DataLoader(Photos(test), batch_size=32, shuffle=False)
    encoder = resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    encoder = encoder.to(device).eval()

    @torch.no_grad()
    def embed(loader):
        features, targets = [], []
        for pixels, labels in loader:
            features.append(encoder(pixels.to(device)))
            targets.append(labels.to(device, dtype=torch.float32))
        return torch.cat(features), torch.cat(targets)

    print(f"Extracting frozen ResNet-18 features on {device}...", flush=True)
    train_features, train_labels = embed(train_loader)
    test_features, _ = embed(test_loader)

    # Give each lesion equal total weight even if it has multiple photographs.
    counts = Counter(key for _, key, _ in train)
    sample_weights = torch.tensor([1 / counts[key] for _, key, _ in train], device=device)
    head = torch.nn.Linear(train_features.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.01)
    for _ in range(100):
        optimizer.zero_grad()
        logits = head(train_features).squeeze(1)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, train_labels, reduction="none"
        )
        loss = (losses * sample_weights).sum() / sample_weights.sum()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probabilities = head(test_features).squeeze(1).sigmoid().cpu().tolist()
    grouped = {}
    for (_, key, label), probability in zip(test, probabilities, strict=True):
        grouped.setdefault(key, ([], label))[0].append(probability)
    auc, log_loss = score_lesions(grouped)
    print(f"Fold {args.held_out} | train: {len(train)} photos / {len(counts)} lesions"
          f" | held out: {len(test)} photos / {len(grouped)} lesions")
    print(f"Lesion-level AUROC: {auc:.4f} | log-loss: {log_loss:.4f}")


if __name__ == "__main__":
    main()
