"""Primary V1.1 metrics on paired held-out predictions, one row per lesion key."""

import numpy as np


PROBABILITY_CLIP = 1e-7
BOOTSTRAP_SEED = 20261125
BOOTSTRAP_RESAMPLES = 2000


def lesion_losses(labels, probabilities):
    """Natural-log losses after clipping the *aggregated* positive probabilities."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or not labels.size or probabilities.shape != labels.shape:
        raise ValueError("Expected equally sized, non-empty vectors of lesion labels and probabilities")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Lesion labels must be binary")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Lesion probabilities must be finite and between zero and one inclusive")
    clipped = np.clip(probabilities, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    return np.where(labels == 1, -np.log(clipped), -np.log1p(-clipped))


def _bootstrap(patient_ids, losses_a, losses_b):
    """Sum losses within patient once, then repeat whole patient contributions per draw."""
    patients, membership, sizes = np.unique(patient_ids, return_inverse=True, return_counts=True)
    sums_a = np.bincount(membership, weights=losses_a)
    sums_b = np.bincount(membership, weights=losses_b)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    differences = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        draw = rng.integers(len(patients), size=len(patients))
        n_lesions = sizes[draw].sum()
        differences[index] = sums_b[draw].sum() / n_lesions - sums_a[draw].sum() / n_lesions
    lower, upper = map(float, np.percentile(differences, [2.5, 97.5], method="linear"))
    if upper < 0:
        interpretation = "B has lower log loss"
    elif lower > 0:
        interpretation = "B has higher log loss"
    else:
        interpretation = "Interval includes zero; direction is inconclusive"
    return {
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "unit": "patient",
        "paired": True,
        "generator": "PCG64",
        "percentile_method": "linear",
        "ci_95": [lower, upper],
        "width": upper - lower,
        "interpretation": interpretation,
        "scope": "Conditional on the fixed out-of-fold fits and splits",
    }


def _loss_summary(losses_a, losses_b):
    mean_a = float(np.mean(losses_a))
    mean_b = float(np.mean(losses_b))
    return {"log_loss_A": mean_a, "log_loss_B": mean_b, "delta_log_loss": mean_b - mean_a}


def primary_report(oof):
    """Evaluate five complete-case outer folds from typed, paired lesion-level OOF rows."""
    keys = set()
    patient_folds = {}
    for row in oof:
        patient, lesion = row["patient_id"], row["lesion_id"]
        if not all(isinstance(value, str) and value.strip() for value in (patient, lesion)):
            raise ValueError("Missing patient or lesion identifier in OOF predictions")
        if (patient, lesion) in keys:
            raise ValueError("Duplicate lesion key in OOF predictions")
        keys.add((patient, lesion))
        fold, n_images = row["outer_fold"], row["n_images"]
        if type(fold) is bool or not isinstance(fold, (int, np.integer)) or not 1 <= fold <= 5:
            raise ValueError("Invalid outer fold in OOF predictions")
        if type(n_images) is bool or not isinstance(n_images, (int, np.integer)) or n_images < 1:
            raise ValueError("Expected a positive image count per lesion")
        if patient_folds.setdefault(patient, fold) != fold:
            raise ValueError("Patient appears in more than one OOF fold")
    if set(patient_folds.values()) != set(range(1, 6)):
        raise ValueError("OOF predictions must include all five outer folds")
    rows = sorted(oof, key=lambda row: (row["patient_id"], row["lesion_id"]))
    patients = np.array([row["patient_id"] for row in rows])
    folds = np.array([row["outer_fold"] for row in rows])
    labels = np.array([row["label"] for row in rows])
    losses_a = lesion_losses(labels, [row["p_A"] for row in rows])
    losses_b = lesion_losses(labels, [row["p_B"] for row in rows])
    by_fold = []
    for fold in range(1, 6):
        selected = folds == fold
        if set(labels[selected]) != {0, 1}:
            raise ValueError("Every outer fold must contain both primary classes")
        by_fold.append({
            "outer_fold": fold,
            "n_keys": int(selected.sum()),
            "n_patients": len(set(patients[selected])),
            **_loss_summary(losses_a[selected], losses_b[selected]),
        })
    return {
        "n_keys": len(rows),
        "n_patients": len(patient_folds),
        "n_images": sum(int(row["n_images"]) for row in rows),
        **_loss_summary(losses_a, losses_b),
        "folds": by_fold,
        "n_folds_favouring_B": sum(fold["delta_log_loss"] < 0 for fold in by_fold),
        "bootstrap": _bootstrap(patients, losses_a, losses_b),
    }
