"""Compare explicit symptom missingness indicators on the fixed PAD-UFES-20 folds."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from pad_ufes20_full import (
    SYMPTOMS,
    Photos,
    clinical_features,
    fit_head,
    load_records,
    metrics,
    read_rows,
)

MODELS = ("model_a", "model_b")
METRICS = ("auroc", "log_loss")
FOLDS = range(1, 6)
BOOTSTRAP_DRAWS = 2000
CSV_FIELDS = ("patient", "lesion_id", "fold", "label", "biopsied", *MODELS)


def load_inputs(metadata_path, folds_path, directories):
    assignments = {}
    for row in read_rows(folds_path):
        patient, fold = row["patient_id"], int(row["fold"])
        if not patient or patient in assignments or fold not in FOLDS:
            raise ValueError(f"Invalid or repeated patient assignment: {patient!r}, {fold}")
        assignments[patient] = fold

    metadata = {}
    for row in read_rows(metadata_path):
        key = row["patient_id"], row["lesion_id"]
        fields = {name: row[name] for name in ("age", "region", "biopsed", *SYMPTOMS)}
        if not all(key) or not row["img_id"] or not fields["region"]:
            raise ValueError(f"Missing identifier or site: {key}")
        if not np.isfinite(float(fields["age"])) or fields["biopsed"] not in {"True", "False"}:
            raise ValueError(f"Invalid age or biopsy status: {key}")
        if any(fields[name] not in {"True", "False", "UNK"} for name in SYMPTOMS):
            raise ValueError(f"Invalid symptom coding: {key}")
        if metadata.setdefault(key, fields) != fields:
            raise ValueError(f"Inconsistent lesion metadata: {key}")

    if set(assignments) != {patient for patient, _ in metadata}:
        raise ValueError("Fold table and metadata must contain the same patients")
    train, test = load_records(metadata_path, folds_path, directories, 1)
    records = train + test
    photo_folds = np.array([assignments[key[0]] for _, key, _ in records])
    labels = np.array([label for _, _, label in records])
    ages = np.array([float(metadata[key]["age"]) for _, key, _ in records])
    for fold in FOLDS:
        train_mask = photo_folds != fold
        if set(labels[train_mask]) != {0, 1} or set(labels[~train_mask]) != {0, 1}:
            raise ValueError(f"Fold {fold} must have both classes in each partition")
        if ages[train_mask].std() == 0:
            raise ValueError(f"Fold {fold} has constant training age")
    return records, metadata, photo_folds, labels


def predict(records, metadata, photo_folds, labels, checkpoint, device):
    torch.manual_seed(42)
    weights = ResNet18_Weights.DEFAULT
    encoder = resnet18(weights=None)
    encoder.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    encoder.fc = torch.nn.Identity()
    encoder = encoder.to(device).eval()
    loader = DataLoader(Photos(records, weights.transforms()), batch_size=32, shuffle=False)
    print(f"Extracting shared features for {len(records)} images on {device}", flush=True)
    with torch.no_grad():
        embeddings = torch.cat([encoder(batch.to(device)) for batch in loader])
    if len(embeddings) != len(records) or not torch.isfinite(embeddings).all().item():
        raise ValueError("Invalid image embeddings")

    # fit_head copies the shared image weights and bias and zeros every clinical coefficient.
    initial = torch.nn.Linear(embeddings.shape[1], 1).to(device)
    counts = Counter(key for _, key, _ in records)
    image_weights = torch.tensor([1 / counts[key] for _, key, _ in records], device=device)
    targets = torch.tensor(labels, dtype=torch.float32, device=device)
    predictions = {}
    for fold in FOLDS:
        train_mask = photo_folds != fold
        base, unknown, yes = clinical_features(records, metadata, train_mask)
        mask = torch.as_tensor(train_mask, device=device)
        held_out = [record for record, held in zip(records, ~train_mask, strict=True) if held]
        extras_a = np.column_stack([base, unknown, yes])
        extras_b = np.column_stack([base, yes])
        fold_predictions = {}
        for model, extras in zip(MODELS, (extras_a, extras_b), strict=True):
            clinical = torch.as_tensor(extras, dtype=embeddings.dtype, device=device)
            features = torch.cat([embeddings, clinical], dim=1)
            probabilities = fit_head(features, targets, image_weights, mask, device, initial)
            grouped = defaultdict(list)
            for (_, key, label), probability in zip(held_out, probabilities, strict=True):
                if not 0 <= probability <= 1:
                    raise ValueError(f"Invalid probability for {key}, {model}")
                fold_predictions.setdefault(key, {
                    "patient": key[0], "lesion_id": key[1], "fold": fold, "label": label,
                    "biopsied": metadata[key]["biopsed"] == "True",
                })
                grouped[key].append(probability)
            for key, values in grouped.items():
                if len(values) != counts[key]:
                    raise ValueError(f"Lesion images split across folds: {key}")
                fold_predictions[key][model] = mean(values)
        if predictions.keys() & fold_predictions.keys():
            raise ValueError(f"Duplicate out-of-fold lesion in fold {fold}")
        predictions.update(fold_predictions)
        print(f"Fold {fold}: {len(fold_predictions)} lesions", flush=True)

    if predictions.keys() != metadata.keys():
        raise ValueError("Out-of-fold predictions do not cover the metadata lesion keys")
    return predictions


def summarize(rows, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["patient"]].append(row)
    positives = sum(row["label"] for row in rows)
    negatives = len(rows) - positives
    result = {
        "lesions": len(rows), "patients": len(groups),
        "positive_lesions": positives, "negative_lesions": negatives,
    }
    if not positives or not negatives:
        # Log loss remains defined for a nonempty single-class subset.
        scores = {}
        for model in MODELS:
            probabilities = np.clip([row[model] for row in rows], 1e-7, 1 - 1e-7)
            loss = -np.log(probabilities if positives else 1 - probabilities)
            scores[model] = {"auroc": None, "log_loss": float(loss.mean()) if rows else None}
        delta = scores["model_a"]["log_loss"] - scores["model_b"]["log_loss"] if rows else None
        return dict(result, metrics=scores, delta_a_minus_b={"auroc": None, "log_loss": delta},
                    patient_bootstrap_95_ci=None, status="insufficient_class_support_for_auroc")

    scores = {model: metrics(rows, model) for model in MODELS}
    patients = list(groups.values())
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(BOOTSTRAP_DRAWS):
        draw = [row for i in rng.integers(len(patients), size=len(patients)) for row in patients[i]]
        if 0 < sum(row["label"] for row in draw) < len(draw):
            deltas.append(metrics(draw, "model_a") - metrics(draw, "model_b"))
    intervals = np.percentile(deltas, [2.5, 97.5], axis=0).T.tolist() if deltas else [None, None]
    return dict(
        result,
        metrics={model: dict(zip(METRICS, values.tolist())) for model, values in scores.items()},
        delta_a_minus_b=dict(zip(METRICS, (scores["model_a"] - scores["model_b"]).tolist())),
        patient_bootstrap_95_ci=dict(zip(METRICS, intervals)),
        bootstrap_attempted=BOOTSTRAP_DRAWS,
        bootstrap_valid=len(deltas),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--images", required=True, nargs="+", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Local ResNet18 DEFAULT checkpoint. No download is performed.")
    parser.add_argument("--output", required=True, type=Path, help="New JSON results path")
    parser.add_argument("--predictions-output", type=Path, help="New CSV predictions path")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    csv_path = args.predictions_output or args.output.with_name(f"{args.output.stem}_predictions.csv")
    if csv_path.resolve() == args.output.resolve():
        parser.error("JSON and CSV paths must differ")
    for path in (args.output, csv_path):
        if path.exists() or path.is_symlink() or not path.parent.is_dir():
            parser.error(f"Output must be a fresh path in an existing directory: {path}")
    for path in (args.metadata, args.folds, args.checkpoint):
        if not path.is_file():
            parser.error(f"Missing input file: {path}")
    for directory in args.images:
        if not directory.is_dir():
            parser.error(f"Missing image directory: {directory}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    with args.checkpoint.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    expected_prefix = Path(ResNet18_Weights.DEFAULT.url).stem.rsplit("-", 1)[1]
    if not checkpoint_sha256.startswith(expected_prefix):
        parser.error("Local checkpoint does not match the ResNet18 DEFAULT weights hash prefix")

    records, metadata, folds, labels = load_inputs(args.metadata, args.folds, args.images)
    predictions = predict(records, metadata, folds, labels, args.checkpoint, torch.device(args.device))
    rows = list(predictions.values())
    missing = {key for key, fields in metadata.items() if any(fields[s] == "UNK" for s in SYMPTOMS)}
    result = {
        "models": {
            "model_a": "Image + Age + Site + Unknown Indicators + Symptoms",
            "model_b": "Image + Age + Site + Symptoms",
        },
        "encoding": {
            "symptoms": list(SYMPTOMS), "yes": {"True": 1, "False": 0, "UNK": 0},
            "unknown_indicator_model_a_only": {"True": 0, "False": 0, "UNK": 1},
            "interpretation": "Encoding ablation only. Other features can still carry missingness information.",
        },
        "images": len(records),
        "device": args.device, "torch": str(torch.__version__),
        "checkpoint_sha256": checkpoint_sha256,
        "seed": 42,
        "fold_lesions": dict(sorted(Counter(row["fold"] for row in rows).items())),
        "bootstrap_method": "Paired patient-cluster percentile, fixed out-of-fold predictions",
        "all_lesions": summarize(rows, 42),
        "biopsy_confirmed_lesions": summarize([row for row in rows if row["biopsied"]], 43),
        "exploratory_missingness_subsets": {
            "no_unknown_symptoms": summarize([row for key, row in predictions.items() if key not in missing], 44),
            "any_unknown_symptom": summarize([row for key, row in predictions.items() if key in missing], 45),
        },
    }
    payload = json.dumps(result, indent=2, allow_nan=False)
    with args.output.open("x", encoding="utf-8") as summary, csv_path.open(
        "x", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        summary.write(payload + "\n")


if __name__ == "__main__":
    main()
