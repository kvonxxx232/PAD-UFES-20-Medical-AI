"""Read the existing V1.1 split and reuse V1.0's frozen, offline image encoder."""

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from pad_ufes20_curate import COHORT_REFERENCE, file_sha256, require_match, select_columns
from pad_v11_fold_inputs import load_curation
from pad_v11_images import _inventory
from pad_v11_metadata import SYMPTOMS
from pad_v11_nested import Photo


# Fixed Stage 3 artifacts. The summary authenticates the inputs as well as the split.
SPLITS_SHA256 = "b115e884ea549b60c09a8a61a81212cd36bc25b2fb38ab9a5f42dfc848f7b013"
FOLD_SUMMARY_SHA256 = "226a2acf50ab24823fb024a91721cbf2a036220937f784e29894776c32556696"
# Published in the V1.0 reproducibility notes.
CHECKPOINT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def _snapshot(path: Path, expected: str, description: str) -> bytes:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"{description} SHA-256 mismatch.")
    return raw


def _fold_assignments(raw, keys, quarantine):
    """Validate the saved assignments on *all* curated keys before restriction."""
    if keys.keys() & quarantine:
        raise ValueError("Quarantined key in curated cohort.")
    patients, components, folds = {}, {}, {}
    complete_labels = {fold: set() for fold in range(1, 6)}
    for row in select_columns(raw, ("patient_id", "lesion_id", "fold")):
        key = (row["patient_id"], row["lesion_id"])
        if key in folds or key not in keys:
            raise ValueError("Repeated or non-curated key in fixed split.")
        fold = int(row["fold"])
        if fold not in complete_labels:
            raise ValueError("Invalid outer fold number.")
        component, label, complete = keys[key]
        if (patients.setdefault(key[0], fold) != fold
                or components.setdefault(component, fold) != fold):
            raise ValueError("Patient or linked patient group crosses saved outer folds.")
        if complete:
            complete_labels[fold].add(label)
        folds[key] = fold
    if folds.keys() != keys.keys() or any(labels != {0, 1} for labels in complete_labels.values()):
        raise ValueError("Incomplete outer key coverage or missing primary class in an outer fold.")
    return folds


@dataclass(frozen=True)
class VerifiedPNG:
    path: Path
    sha256: str


def _verified_images(inventory_raw, image_dirs):
    digests = {row["img_id"]: row["file_sha256"]
               for row in select_columns(inventory_raw, ("img_id", "file_sha256"))}
    try:
        paths = _inventory(digests, image_dirs)
    except ValueError:
        raise ValueError("PNG filesystem inventory differs from the Stage 2 manifest.") from None
    images = {}
    for name, path in paths.items():
        if file_sha256(path) != digests[name]:
            raise ValueError("A PNG differs from its Stage 2 SHA-256.")
        images[name] = VerifiedPNG(path, digests[name])
    return images


def load_inputs(curation: Path, metadata: Path, review: Path, fold_dir: Path,
                image_dirs: Sequence[Path]):
    """Authenticate production inputs without splitting, decoding images or fitting.

    Return ordered complete-case photos, their paths and expected digests, and
    aggregate provenance. Patient and lesion identifiers stay in process memory.
    """
    keys, quarantine, provenance = load_curation(curation, metadata, review)
    summary = json.loads(_snapshot(fold_dir / "fold_summary.json", FOLD_SUMMARY_SHA256,
                                   "Fixed outer-fold summary"))
    if (summary["input_provenance"] != provenance or summary["splits_sha256"] != SPLITS_SHA256
            or summary["protocol"] != "protocol-v1.1-frozen" or summary["level"] != "outer"
            or summary["n_splits"] != 5 or summary["seed"] != 20261124
            or not summary["shuffle"] or summary["splitter"] != "StratifiedGroupKFold"
            or summary["stratum"] != "2 * binary_label + int(symptom_complete)"):
        raise ValueError("Fixed outer-fold provenance differs from the verified curation or protocol.")
    folds = _fold_assignments(_snapshot(fold_dir / "splits.csv", SPLITS_SHA256,
                                        "Fixed outer-fold assignments"), keys, quarantine)

    inventory_digest = provenance["stage2_output_sha256"]["image_inventory.csv"]
    inventory = _snapshot(curation / "image_inventory.csv", inventory_digest, "Stage 2 PNG inventory")
    images = _verified_images(inventory, image_dirs)
    clinical = _snapshot(metadata, provenance["metadata_sha256"], "Verified metadata")
    photos = []
    columns = ("img_id", "patient_id", "lesion_id", "age", "region", *SYMPTOMS)
    binary = {"True": 1, "False": 0}
    for row in select_columns(clinical, columns):
        key = (row["patient_id"], row["lesion_id"])
        if key not in keys or not keys[key][2]:
            continue
        group, label, _ = keys[key]
        # Complete-case validation already checked each token on every photo.
        symptoms = tuple(binary[row[field]] for field in SYMPTOMS)
        photos.append(Photo(image_id=row["img_id"], patient_id=key[0], lesion_id=key[1],
                            group=group, fold=folds[key], label=label, age=int(row["age"]),
                            site=row["region"], symptoms=symptoms))
    photos.sort(key=lambda p: (p.patient_id, p.lesion_id, p.image_id))
    primary = {(p.patient_id, p.lesion_id): p.label for p in photos}
    counts = {"keys": len(primary), "patients": len({p.patient_id for p in photos}),
              "images": len(photos), "positive": sum(primary.values()),
              "negative": len(primary) - sum(primary.values())}
    require_match(counts, COHORT_REFERENCE["primary"])
    stage2 = json.loads(_snapshot(curation / "summary.json", provenance["stage2_summary_sha256"],
                                  "Stage 2 curation summary"))
    provenance = {
        **provenance, "source_archive_sha256": stage2["source_archive_sha256"],
        "fold_summary_sha256": FOLD_SUMMARY_SHA256, "splits_sha256": SPLITS_SHA256,
        "verified_pngs": len(images), "curated_keys": len(keys),
        "quarantined_keys": len(quarantine), "primary": counts,
    }
    return photos, {p.image_id: images[p.image_id] for p in photos}, provenance


def verify_checkpoint(checkpoint: Path) -> str:
    digest = file_sha256(checkpoint)
    if digest != CHECKPOINT_SHA256:
        raise ValueError("Local checkpoint differs from the published V1.0 ResNet18 SHA-256.")
    return digest


class _ImageRecords:
    """Adapt checksum-verified bytes to the path-or-stream interface of V1.0 Photos."""

    def __init__(self, photos, images):
        self.photos = photos
        self.images = images

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, index):
        photo = self.photos[index]
        image = self.images[photo.image_id]
        raw = _snapshot(image.path, image.sha256, "PNG at extraction")
        return io.BytesIO(raw), (photo.patient_id, photo.lesion_id), photo.label


def extract_embeddings(photos, paths, checkpoint: Path, batch_size: int = 32,
                       device: str = "cpu") -> np.ndarray:
    """One ordered, shared inference pass using the verified offline V1.0 weights."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision.models import ResNet18_Weights, resnet18
    from pad_ufes20_full import Photos

    if not photos or batch_size < 1:
        raise ValueError("A non-empty photo list and positive batch size are required.")
    checkpoint_raw = _snapshot(checkpoint, CHECKPOINT_SHA256, "V1.0 ResNet18 checkpoint")
    encoder = resnet18(weights=None)
    encoder.load_state_dict(torch.load(io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=True))
    encoder.fc = torch.nn.Identity()
    encoder.requires_grad_(False).to(device).eval()
    dataset = Photos(_ImageRecords(photos, paths), ResNet18_Weights.DEFAULT.transforms())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.inference_mode():
        embeddings = torch.cat([encoder(batch.to(device)).cpu() for batch in loader]).numpy()
    if embeddings.shape != (len(photos), 512) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid frozen ResNet18 embeddings.")
    return embeddings
