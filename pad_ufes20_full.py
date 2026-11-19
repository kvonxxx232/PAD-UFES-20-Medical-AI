"""Compare image and clinical feature sets on fixed patient-held-out folds."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

SYMPTOMS = ("grew", "changed", "bleed")
MODELS = ("image", "image_age_site", "image_age_site_unknown", "image_age_site_symptoms")


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


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


class Photos(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.records[index][0]) as source:
            return self.transform(source.convert("RGB"))


def clinical_features(records, metadata, train_mask):
    rows = [metadata[key] for _, key, _ in records]
    ages = np.array([float(row["age"]) for row in rows])
    age = (ages - ages[train_mask].mean()) / ages[train_mask].std()
    sites = sorted({row["region"] for row, train in zip(rows, train_mask) if train})
    site = [[row["region"] == category for category in sites] for row in rows]

    # Two indicators per symptom encode all three states with False as reference.
    unknown = [[row[name] == "UNK" for name in SYMPTOMS] for row in rows]
    yes = [[row[name] == "True" for name in SYMPTOMS] for row in rows]
    return np.column_stack([age, site]), np.array(unknown), np.array(yes)


def fit_head(features, labels, weights, train_mask, device, image_initial):
    train_features = features[train_mask]
    train_labels = labels[train_mask]
    sample_weights = weights[train_mask]
    head = torch.nn.Linear(features.shape[1], 1).to(device)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[:, :image_initial.weight.shape[1]].copy_(image_initial.weight)
        head.bias.copy_(image_initial.bias)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.01)
    for _ in range(100):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            head(train_features).squeeze(1), train_labels, reduction="none")
        (loss * sample_weights).sum().div(sample_weights.sum()).backward()
        optimizer.step()
    with torch.no_grad():
        return head(features[~train_mask]).squeeze(1).sigmoid().cpu().tolist()


def metrics(rows, model):
    grouped = {i: ([row[model]], row["label"]) for i, row in enumerate(rows)}
    return np.array(score_lesions(grouped))


def summarize(rows, seed):
    patient_rows = defaultdict(list)
    for row in rows:
        patient_rows[row["patient"]].append(row)
    groups = list(patient_rows.values())
    scores = {model: metrics(rows, model) for model in MODELS}
    main, base = "image_age_site_symptoms", "image_age_site"
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(2000):
        draw = [row for i in rng.integers(len(groups), size=len(groups)) for row in groups[i]]
        deltas.append(metrics(draw, main) - metrics(draw, base))
    intervals = np.percentile(deltas, [2.5, 97.5], axis=0)
    return {
        "lesions": len(rows),
        "patients": len(groups),
        "metrics": {model: dict(zip(("auroc", "log_loss"), value.tolist()))
                    for model, value in scores.items()},
        "symptom_delta": dict(zip(("auroc", "log_loss"), (scores[main] - scores[base]).tolist())),
        "patient_bootstrap_95_ci": dict(zip(("auroc", "log_loss"), intervals.T.tolist())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path,
                        help="CSV path (default: <output stem>_predictions.csv)")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()

    csv_path = args.predictions_output or args.output.with_name(
        f"{args.output.stem}_predictions.csv")
    if csv_path.resolve() == args.output.resolve():
        parser.error("JSON and CSV output paths must be different.")
    for path in (args.output, csv_path):
        if path.exists() or path.is_symlink():
            parser.error(f"Output already exists: {path}")

    train, test = load_records(args.metadata, args.folds, args.images, 1)
    records = train + test
    assignments = {row["patient_id"]: int(row["fold"]) for row in read_rows(args.folds)}
    photo_folds = np.array([assignments[key[0]] for _, key, _ in records])
    metadata = {}
    for row in read_rows(args.metadata):
        key = row["patient_id"], row["lesion_id"]
        fields = {field: row[field] for field in ("age", "region", "biopsed", *SYMPTOMS)}
        if metadata.setdefault(key, fields) != fields:
            raise ValueError(f"Inconsistent lesion metadata: {key}")
        if any(row[field] not in {"True", "False", "UNK"} for field in SYMPTOMS):
            raise ValueError(f"Unexpected symptom value for lesion {key}")

    torch.manual_seed(42)
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable. Use --device cpu.")
    device = torch.device(args.device)
    weights = ResNet18_Weights.DEFAULT
    encoder = resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    encoder = encoder.to(device).eval()
    loader = DataLoader(Photos(records, weights.transforms()), batch_size=32)
    print(f"Extracting features for {len(records)} images on {device}", flush=True)
    with torch.no_grad():
        embeddings = torch.cat([encoder(batch.to(device)) for batch in loader])
    initial = torch.nn.Linear(embeddings.shape[1], 1).to(device)

    counts = Counter(key for _, key, _ in records)
    image_weights = torch.tensor([1 / counts[key] for _, key, _ in records], device=device)
    labels = torch.tensor([label for _, _, label in records], dtype=torch.float32, device=device)
    predictions = {}
    fold_results = {}
    for fold in range(1, 6):
        train_mask = photo_folds != fold
        base, unknown, yes = clinical_features(records, metadata, train_mask)
        extras = (None, base, np.column_stack([base, unknown]),
                  np.column_stack([base, unknown, yes]))
        held_out = [record for record, keep in zip(records, ~train_mask) if keep]
        mask = torch.as_tensor(train_mask, device=device)
        for model, extra in zip(MODELS, extras):
            features = embeddings
            if extra is not None:
                extra = torch.tensor(extra, dtype=embeddings.dtype, device=device)
                features = torch.cat([embeddings, extra], dim=1)
            probs = fit_head(features, labels, image_weights, mask, device, initial)
            by_lesion = defaultdict(list)
            for (_, key, label), prob in zip(held_out, probs, strict=True):
                predictions.setdefault(key, {
                    "patient": key[0],
                    "lesion_id": key[1],
                    "label": label,
                    "fold": fold,
                    "biopsied": metadata[key]["biopsed"] == "True",
                })
                by_lesion[key].append(prob)
            for key, values in by_lesion.items():
                predictions[key][model] = mean(values)
        rows = [predictions[key] for key in by_lesion]
        fold_results[fold] = {model: metrics(rows, model).tolist() for model in MODELS}
        print(f"Fold {fold}: {len(rows)} lesions | {fold_results[fold]}", flush=True)

    if predictions.keys() != metadata.keys():
        raise ValueError("Prediction lesion keys do not match metadata lesion keys")
    rows = list(predictions.values())
    result = {
        "device": str(device),
        "torch": torch.__version__,
        "fold_metrics_order": ["auroc", "log_loss"],
        "folds": fold_results,
        "all_lesions": summarize(rows, 42),
        "biopsy_confirmed_lesions": summarize([row for row in rows if row["biopsied"]], 43),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["patient", "lesion_id", "label", "fold", "biopsied", *MODELS])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
