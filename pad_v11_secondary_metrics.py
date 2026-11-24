"""Prespecified S1, S3 and S6 summaries of verified paired H1 lesion predictions."""

import numpy as np
from scipy.optimize import root
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from pad_v11_primary import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, lesion_losses


def _vectors(rows):
    if len({(r["patient_id"], r["lesion_id"]) for r in rows}) != len(rows):
        raise ValueError("Repeated lesion key in OOF predictions.")
    labels = np.array([r["label"] for r in rows])
    return labels, [np.array([r["p_" + model] for r in rows]) for model in ("A", "B")]


def _pair(a, b):
    return {"A": a, "B": b, "delta": None if a is None or b is None else b - a}


def _auroc(labels, probabilities, weights=None):
    if np.unique(labels if weights is None else labels[weights > 0]).size < 2:
        return None
    return float(roc_auc_score(labels, probabilities, sample_weight=weights))


def _interval(draws):
    defined = [value for value in draws if value is not None]
    bounds = np.percentile(defined, [2.5, 97.5], method="linear").tolist() if defined else None
    return {"ci_95": bounds, "defined_draws": len(defined), "undefined_draws": len(draws) - len(defined)}


def s1_summary(oof):
    """Patient multiplicities reproduce explicit lesion repetitions on the H1 bootstrap draws."""
    rows = sorted(oof, key=lambda r: (r["patient_id"], r["lesion_id"]))
    if not rows:
        raise ValueError("S1 requires lesion predictions")
    labels, probabilities = _vectors(rows)
    patients, membership = np.unique([r["patient_id"] for r in rows], return_inverse=True)
    squared_errors = [(p - labels) ** 2 for p in probabilities]
    auroc = _pair(*[_auroc(labels, p) for p in probabilities])
    brier = _pair(*[float(np.mean(errors)) for errors in squared_errors])
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    auroc_draws, brier_draws = [], []
    for _ in range(BOOTSTRAP_RESAMPLES):
        draw = rng.integers(len(patients), size=len(patients))
        weights = np.bincount(draw, minlength=len(patients))[membership]
        roc_a, roc_b = [_auroc(labels, p, weights) for p in probabilities]
        auroc_draws.append(None if roc_a is None else roc_b - roc_a)
        brier_draws.append(float(np.average(squared_errors[1] - squared_errors[0], weights=weights)))
    return {
        "n_keys": len(rows), "n_patients": len(patients),
        "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
        "auroc": {**auroc, **_interval(auroc_draws)}, "brier": {**brier, **_interval(brier_draws)},
    }


def _biopsy_group(rows):
    labels, probabilities = _vectors(rows)
    return {
        "n_keys": len(rows), "n_patients": len({r["patient_id"] for r in rows}),
        "n_images": sum(r["n_images"] for r in rows), "positive": int(labels.sum()),
        "negative": int(len(rows) - labels.sum()), "positive_fraction": float(labels.mean()),
        "auroc": _pair(*[_auroc(labels, p) for p in probabilities]),
        "log_loss": _pair(*[float(lesion_losses(labels, p).mean()) for p in probabilities]),
        "brier": _pair(*[float(np.mean((labels - p) ** 2)) for p in probabilities]),
    }


def s3_summary(oof, biopsy_keys):
    """Use only the supplied all-photos-biopsied complete keys from verified metadata."""
    rows = [r for r in oof if (r["patient_id"], r["lesion_id"]) in biopsy_keys]
    if not rows or len(rows) != len(biopsy_keys):
        raise ValueError("S3 membership does not match the complete-case OOF keys")
    folds = []
    for fold in sorted({r["outer_fold"] for r in oof}):
        subset = [r for r in rows if r["outer_fold"] == fold]
        if not subset:
            raise ValueError("S3 has an empty outer fold")
        folds.append({"outer_fold": fold, **_biopsy_group(subset)})
    return {"pooled": _biopsy_group(rows), "folds": folds}


def reliability_bins(labels, probabilities):
    """Fixed decimal tenths, left-closed and right-open except for the upper endpoint 1."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if (labels.ndim != 1 or labels.shape != probabilities.shape or not np.isin(labels, (0, 1)).all()
            or not np.all((probabilities >= 0) & (probabilities <= 1))):
        raise ValueError("Expected equal-length binary labels and probabilities in [0, 1]")
    edges = np.arange(11) / 10
    bin_ids = np.searchsorted(edges[1:-1], probabilities, side="right")
    bins = []
    for i in range(10):
        selected = bin_ids == i
        count = int(selected.sum())
        positive = int(labels[selected].sum())
        bins.append({
            "lower": float(edges[i]), "upper": float(edges[i + 1]), "count": count,
            "positive": positive, "mean_predicted": float(probabilities[selected].mean()) if count else None,
            "observed": positive / count if count else None,
        })
    return bins


def _calibration(labels, probabilities):
    bins = reliability_bins(labels, probabilities)
    if np.unique(labels).size != 2 or not np.all((probabilities > 0) & (probabilities < 1)):
        raise ValueError("Calibration regression requires both classes and finite original logits")
    logits = logit(probabilities)
    positive, negative = logits[labels == 1], logits[labels == 0]
    if positive.min() >= negative.max() or negative.min() >= positive.max():
        raise ValueError("No identifiable finite calibration slope and intercept")
    design = np.column_stack((np.ones(len(labels)), logits))

    def score_and_jacobian(coefficients):
        fitted = expit(design @ coefficients)
        variance = fitted * (1.0 - fitted)
        return design.T @ (fitted - labels), design.T @ (variance[:, None] * design)

    solution = root(score_and_jacobian, [0.0, 0.0], jac=True, tol=1e-10)
    if (not solution.success or not np.isfinite(solution.x).all()
            or np.max(np.abs(solution.fun)) > 1e-7):
        raise ValueError("Joint unpenalised calibration regression did not converge")
    return {
        "intercept": float(solution.x[0]), "slope": float(solution.x[1]),
        "mean_predicted": float(probabilities.mean()), "observed": float(labels.mean()), "bins": bins,
    }


def s6_summary(oof):
    """Diagnose calibration on original OOF probabilities without recalibrating any prediction."""
    labels, probabilities = _vectors(oof)
    return {model: _calibration(labels, p) for model, p in zip(("A", "B"), probabilities)}
