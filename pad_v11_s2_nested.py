"""Frozen S2 nested CV. Train on all curated keys and score complete keys only."""

from collections import Counter, defaultdict
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from pad_ufes20_folds import N_SPLITS, SEED
from pad_v11_metadata import REGIONS
from pad_v11_nested import (C_GRID, MODELS, OuterSplit, Partition, Photo, _design_matrices,
                            _fit_predict, mean_probabilities, select_c)
from pad_v11_primary import lesion_losses


def plan_s2(photos: Sequence[Photo], complete_keys: set[tuple[str, str]]) -> list[OuterSplit]:
    """Check saved outer assignments and construct S2 inner splits on all training keys.

    Each Partition.train includes every training photo. Partition.held_out includes
    only scored complete photos. Group separation is checked before this restriction.
    """
    indices = defaultdict(list)
    seen_images, patient_groups, group_folds = set(), {}, {}
    for i, p in enumerate(photos):
        if not all((p.image_id, p.patient_id, p.lesion_id, p.group)) or p.image_id in seen_images:
            raise ValueError("Missing or duplicate S2 identifier.")
        seen_images.add(p.image_id)
        if p.fold not in range(1, N_SPLITS + 1) or p.label not in (0, 1):
            raise ValueError("Invalid saved outer fold or binary outcome.")
        if (not 0 <= p.age <= 120 or not float(p.age).is_integer() or p.site not in REGIONS
                or len(p.symptoms) != 3 or any(v not in (0, 1) for v in p.symptoms)):
            raise ValueError("Invalid S2 age, site or encoded symptom value.")
        if patient_groups.setdefault(p.patient_id, p.group) != p.group:
            raise ValueError("A patient belongs to multiple groups.")
        if group_folds.setdefault(p.group, p.fold) != p.fold:
            raise ValueError("A patient group crosses saved outer folds.")
        if indices[p.key]:
            first = photos[indices[p.key][0]]
            if (p.age, p.site, p.label) != (first.age, first.site, first.label):
                raise ValueError("Inconsistent age, site or label within an S2 key.")
            if p.key in complete_keys and first.symptoms != p.symptoms:
                raise ValueError("Conflicting symptoms within a complete S2 key.")
        indices[p.key].append(i)
    if not complete_keys or not complete_keys <= indices.keys():
        raise ValueError("Missing or unrecognised complete keys in S2.")

    keys = sorted(indices)
    first = {k: photos[indices[k][0]] for k in keys}
    for views in indices.values():
        views.sort(key=lambda i: photos[i].image_id)

    def partition(train, validation):
        if {first[k].group for k in train} & {first[k].group for k in validation}:
            raise ValueError("Training and unfiltered held-out groups overlap.")
        scored = [k for k in validation if k in complete_keys]
        if any({first[k].label for k in side} != {0, 1} for side in (train, scored)):
            raise ValueError("S2 training or complete-case validation lacks a class.")
        return Partition(tuple(i for k in train for i in indices[k]),
                         tuple(i for k in scored for i in indices[k]))

    plans = []
    for fold in range(1, N_SPLITS + 1):
        train = [k for k in keys if first[k].fold != fold]
        test = [k for k in keys if first[k].fold == fold]
        outer = partition(train, test)
        groups = [first[k].group for k in train]
        if len(set(groups)) < N_SPLITS:
            raise ValueError("Too few S2 outer-training groups for five inner folds.")
        strata = [2 * first[k].label + int(k in complete_keys) for k in train]
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        inner, validated = [], []
        for fit, val in splitter.split(np.zeros(len(train)), strata, groups):
            if not np.array_equal(np.sort(np.concatenate((fit, val))), np.arange(len(train))):
                raise ValueError("Inner split does not partition every curated outer-training key.")
            inner.append(partition([train[i] for i in fit], [train[i] for i in val]))
            validated.extend(val)
        if sorted(validated) != list(range(len(train))):
            raise ValueError("Inner validation coverage differs from the full outer-training set.")
        plans.append(OuterSplit(fold, outer, tuple(inner)))
    return plans


def run_s2_nested(photos: Sequence[Photo], embeddings: np.ndarray, complete_keys: set[tuple[str, str]]):
    """Fit all S2 heads with H1's preprocessing, weights, model and convergence checks."""
    embeddings = np.asarray(embeddings)
    if (embeddings.ndim != 2 or embeddings.shape[0] != len(photos)
            or embeddings.shape[1] == 0 or not np.isfinite(embeddings).all()):
        raise ValueError("Invalid or misaligned S2 embeddings.")
    plans = plan_s2(photos, complete_keys)
    by_key = {p.key: p for p in photos}
    counts = Counter(p.key for p in photos)
    oof, tuning, diagnostics = {}, [], []
    for split in plans:
        predictions = {(model, c): {} for model in MODELS for c in C_GRID}
        for inner, part in enumerate(split.inner, 1):
            for model, x, validation_x, y, weights, held in _design_matrices(photos, embeddings, part):
                context = {"outer_fold": split.fold, "inner_fold": inner, "model": model}
                for c in C_GRID:
                    probs = _fit_predict(x, validation_x, y, weights, c, context, diagnostics)
                    predictions[model, c].update(mean_probabilities(held, probs))

        scored_keys = sorted(k for k in complete_keys if by_key[k].fold != split.fold)
        scored_labels = [by_key[k].label for k in scored_keys]
        selected = {}
        for model in MODELS:
            grid = {}
            for c in C_GRID:
                probabilities = predictions[model, c]
                if probabilities.keys() != set(scored_keys):
                    raise ValueError("S2 tuning predictions differ from complete outer-training keys.")
                grid[c] = float(lesion_losses(scored_labels, [probabilities[k] for k in scored_keys]).mean())
            selected[model] = select_c(grid)
            tuning.extend({"outer_fold": split.fold, "model": model, "C": c,
                           "pooled_log_loss": grid[c], "selected": c == selected[model],
                           "n_keys": len(scored_keys)} for c in C_GRID)

        for model, x, test_x, y, weights, held in _design_matrices(photos, embeddings, split.outer):
            context = {"outer_fold": split.fold, "inner_fold": None, "model": model}
            probs = _fit_predict(x, test_x, y, weights, selected[model], context, diagnostics)
            for key, probability in mean_probabilities(held, probs).items():
                row = oof.setdefault(key, {"patient_id": key[0], "lesion_id": key[1],
                                           "label": by_key[key].label, "outer_fold": split.fold,
                                           "n_images": counts[key]})
                if f"p_{model}" in row or row["outer_fold"] != by_key[key].fold:
                    raise ValueError("Duplicate or misplaced S2 held-out prediction.")
                row[f"p_{model}"] = probability
    if oof.keys() != complete_keys or any("p_A" not in row or "p_B" not in row for row in oof.values()):
        raise ValueError("Incomplete S2 paired OOF prediction coverage.")
    return [oof[k] for k in sorted(oof)], tuning, diagnostics
