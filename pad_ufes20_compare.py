"""Compare image and clinical feature sets on fixed patient-held-out folds."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from pad_ufes20_image_baseline import load_records, score_lesions

SYMPTOMS = ("grew", "changed", "bleed")
MODELS = ("image", "image_age_site", "image_age_site_unknown", "image_age_site_symptoms")


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
    args = parser.parse_args()

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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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
                predictions.setdefault(key, {"patient": key[0], "label": label, "fold": fold,
                                             "biopsied": metadata[key]["biopsed"] == "True"})
                by_lesion[key].append(prob)
            for key, values in by_lesion.items():
                predictions[key][model] = mean(values)
        rows = [predictions[key] for key in by_lesion]
        fold_results[fold] = {model: metrics(rows, model).tolist() for model in MODELS}
        print(f"Fold {fold}: {len(rows)} lesions | {fold_results[fold]}", flush=True)

    rows = list(predictions.values())
    assert len(rows) == len(metadata) and len(records) == sum(counts.values())
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


if __name__ == "__main__":
    main()
