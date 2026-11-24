"""Frozen V1.1 primary A/B heads on aligned frozen image embeddings and complete photos."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from pad_ufes20_folds import N_SPLITS, SEED
from pad_v11_metadata import REGIONS
from pad_v11_primary import lesion_losses


C_GRID = (0.001, 0.01, 0.1, 1, 10)
MODELS = ("A", "B")


@dataclass(frozen=True)
class Photo:
    image_id: str
    patient_id: str
    lesion_id: str
    group: str
    fold: int
    label: int
    age: float
    site: str
    symptoms: tuple[int, int, int]  # grew, changed, bleed

    @property
    def key(self):
        return self.patient_id, self.lesion_id


@dataclass(frozen=True)
class Partition:
    train: tuple[int, ...]  # Photo indices in the caller's embedding matrix
    held_out: tuple[int, ...]


@dataclass(frozen=True)
class OuterSplit:
    fold: int
    outer: Partition
    inner: tuple[Partition, ...]


def plan_splits(photos: Sequence[Photo]) -> list[OuterSplit]:
    """Use saved outer folds and create only inner folds on sorted complete training keys."""
    rows = defaultdict(list)
    images = set()
    patient_groups, group_folds = {}, {}
    for i, p in enumerate(photos):
        if not all((p.image_id, p.patient_id, p.lesion_id, p.group)) or p.image_id in images:
            raise ValueError("Empty identifier or repeated image ID in primary inputs.")
        images.add(p.image_id)
        if p.fold not in range(1, N_SPLITS + 1) or p.label not in (0, 1):
            raise ValueError("Invalid saved fold or binary label.")
        if (not 0 <= p.age <= 120 or not float(p.age).is_integer() or p.site not in REGIONS
                or len(p.symptoms) != 3 or any(x not in (0, 1) for x in p.symptoms)):
            raise ValueError("Invalid or incomplete primary clinical inputs.")
        if patient_groups.setdefault(p.patient_id, p.group) != p.group:
            raise ValueError("A patient has inconsistent component assignments.")
        if group_folds.setdefault(p.group, p.fold) != p.fold:
            raise ValueError("A patient component crosses saved outer folds.")
        if rows[p.key]:
            first = photos[rows[p.key][0]]
            if (p.label, p.age, p.site, p.symptoms) != (first.label, first.age, first.site, first.symptoms):
                raise ValueError("Inconsistent primary clinical data within a lesion key.")
        rows[p.key].append(i)

    keys = sorted(rows)
    first_photo = {key: photos[rows[key][0]] for key in keys}
    for indices in rows.values():
        indices.sort(key=lambda i: photos[i].image_id)

    def partition(train_keys, val_keys):
        for side in (train_keys, val_keys):
            if {first_photo[k].label for k in side} != {0, 1}:
                raise ValueError("Infeasible partition: a training or held-out set lacks a class.")
        if {first_photo[k].group for k in train_keys} & {first_photo[k].group for k in val_keys}:
            raise ValueError("Patient component overlap in training and held-out sets.")
        return Partition(tuple(i for k in train_keys for i in rows[k]),
                         tuple(i for k in val_keys for i in rows[k]))

    plan = []
    for fold in range(1, N_SPLITS + 1):
        train = [k for k in keys if first_photo[k].fold != fold]
        test = [k for k in keys if first_photo[k].fold == fold]
        outer = partition(train, test)
        groups = [first_photo[k].group for k in train]
        labels = [first_photo[k].label for k in train]
        if len(set(groups)) < N_SPLITS:
            raise ValueError("Too few independent outer-training groups for inner folds.")
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        inner = tuple(partition([train[i] for i in fit], [train[i] for i in val])
                      for fit, val in splitter.split(np.zeros(len(train)), labels, groups))
        if sorted(i for part in inner for i in part.held_out) != sorted(outer.train):
            raise ValueError("Inner validation does not cover the outer training set exactly once.")
        plan.append(OuterSplit(fold, outer, inner))
    return plan


@dataclass(frozen=True)
class ClinicalEncoder:
    age_mean: float
    age_scale: float
    sites: tuple[str, ...]

    @classmethod
    def fit(cls, training):
        ages = np.array([p.age for p in training], dtype=np.float64)
        return cls(float(ages.mean()), float(ages.std(ddof=0)) or 1.,
                   tuple(sorted({p.site for p in training})))

    def transform(self, photos, embeddings, model):
        if model not in MODELS:
            raise ValueError("Unknown primary model.")
        age = [(p.age - self.age_mean) / self.age_scale for p in photos]
        sites = np.array([[p.site == s for s in self.sites] for p in photos])
        blocks = [embeddings, age, sites]
        if model == "B":
            blocks.append(np.array([p.symptoms for p in photos]))
        return np.column_stack(blocks)


def training_weights(photos):
    counts = Counter(p.key for p in photos)
    return np.array([1. / counts[p.key] for p in photos])


def mean_probabilities(photos, probabilities):
    """Average positive-class photo probabilities without logit transformation or clipping."""
    p = np.asarray(probabilities)
    if p.shape != (len(photos),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid per-image predicted probabilities.")
    by_key = defaultdict(list)
    for photo, prob in zip(photos, p):
        by_key[photo.key].append(prob)
    return {key: float(np.mean(values, dtype=np.float64)) for key, values in by_key.items()}


class FitFailure(ValueError):
    def __init__(self, diagnostics):
        super().__init__("Logistic fit failed or convergence was unconfirmed; inspect fit diagnostics.")
        self.diagnostics = diagnostics


def _fit_predict(train_x, held_x, labels, weights, c, context, diagnostics):
    estimator = LogisticRegression(C=c, solver="lbfgs", l1_ratio=0.0, fit_intercept=True,
                                   class_weight=None, tol=1e-6, max_iter=5000)
    failure = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            estimator.fit(train_x, labels, sample_weight=weights)
        except Exception as error:
            failure = error
    iterations = [int(n) for n in getattr(estimator, "n_iter_", ())]
    reached = any(n >= estimator.max_iter for n in iterations)
    convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    confirmed = failure is None and bool(iterations) and not reached and not convergence_warning
    diagnostics.append({
        **context, "C": c, "n_iter": iterations, "iteration_cap_reached": reached,
        "converged": confirmed, "warnings": [
            {"category": w.category.__name__, "message": str(w.message)} for w in caught],
        "error": repr(failure) if failure is not None else None,
    })
    if not confirmed:
        raise FitFailure(diagnostics) from failure
    if list(estimator.classes_) != [0, 1]:
        raise ValueError("Unexpected class order for positive-class prediction.")
    return estimator.predict_proba(held_x)[:, 1]


def select_c(scores):
    """Smallest C among exact ties for the minimum unrounded pooled key loss."""
    if set(scores) != set(C_GRID) or not np.isfinite(list(scores.values())).all():
        raise ValueError("Incomplete grid or invalid pooled validation losses.")
    return min(C_GRID, key=lambda c: (scores[c], c))


def _design_matrices(photos, embeddings, part):
    train = [photos[i] for i in part.train]
    held = [photos[i] for i in part.held_out]
    train_images = embeddings[list(part.train)]
    held_images = embeddings[list(part.held_out)]
    encoder = ClinicalEncoder.fit(train)
    labels = np.array([p.label for p in train])
    weights = training_weights(train)
    for model in MODELS:
        yield (model, encoder.transform(train, train_images, model),
               encoder.transform(held, held_images, model), labels, weights, held)


def run_nested(photos: Sequence[Photo], embeddings: np.ndarray):
    """Return one paired OOF row per lesion, pooled inner grid scores and all fit diagnostics."""
    embeddings = np.asarray(embeddings)
    if (embeddings.ndim != 2 or embeddings.shape[0] != len(photos)
            or embeddings.shape[1] == 0 or not np.isfinite(embeddings).all()):
        raise ValueError("Invalid or misaligned image embedding matrix.")
    plan = plan_splits(photos)  # Validate all partitions before fitting a head.
    labels_by_key = {p.key: p.label for p in photos}
    counts = Counter(p.key for p in photos)
    oof, tuning, diagnostics = {}, [], []
    for split in plan:
        # Every C/model uses the same held-out training keys in the same folds.
        predictions = {(model, c): {} for model in MODELS for c in C_GRID}
        for inner_fold, part in enumerate(split.inner, start=1):
            for model, train_x, held_x, labels, weights, held in _design_matrices(photos, embeddings, part):
                for c in C_GRID:
                    context = {"outer_fold": split.fold, "inner_fold": inner_fold, "model": model}
                    probs = _fit_predict(train_x, held_x, labels, weights, c, context, diagnostics)
                    predictions[model, c].update(mean_probabilities(held, probs))

        keys = sorted({photos[i].key for i in split.outer.train})
        y = np.array([labels_by_key[k] for k in keys])
        chosen = {}
        for model in MODELS:
            scores = {}
            for c in C_GRID:
                pooled = predictions[model, c]
                if pooled.keys() != set(keys):
                    raise ValueError("Incomplete inner lesion probability coverage.")
                scores[c] = float(lesion_losses(y, [pooled[k] for k in keys]).mean())
            chosen[model] = select_c(scores)
            tuning.extend({"outer_fold": split.fold, "model": model, "C": c,
                           "pooled_log_loss": scores[c], "selected": c == chosen[model],
                           "n_keys": len(keys)} for c in C_GRID)
        for model, train_x, held_x, labels, weights, held in _design_matrices(photos, embeddings, split.outer):
            context = {"outer_fold": split.fold, "inner_fold": None, "model": model}
            probs = _fit_predict(train_x, held_x, labels, weights, chosen[model], context, diagnostics)
            for key, mean in mean_probabilities(held, probs).items():
                row = oof.setdefault(key, {"patient_id": key[0], "lesion_id": key[1],
                                           "outer_fold": split.fold, "label": labels_by_key[key],
                                           "n_images": counts[key]})
                if row["outer_fold"] != split.fold or f"p_{model}" in row:
                    raise ValueError("Repeated outer held-out predictions.")
                row[f"p_{model}"] = mean
    if oof.keys() != counts.keys() or any("p_A" not in r or "p_B" not in r for r in oof.values()):
        raise ValueError("Incomplete paired OOF predictions.")
    return [oof[k] for k in sorted(oof)], tuning, diagnostics
