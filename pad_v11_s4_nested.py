"""Complete-case S4 refits on H1's fixed patient splits and frozen embeddings."""

from collections import Counter

import numpy as np

from pad_ufes20_folds import N_SPLITS
from pad_v11_nested import C_GRID, MODELS, _design_matrices, _fit_predict, mean_probabilities, select_c
from pad_v11_primary import lesion_losses


FACTORS = (1, 3, 10)


def _check_plan(photos, plan):
    """Catch stale indices, omitted photos and patient/component leakage before fitting."""
    if [split.fold for split in plan] != list(range(1, N_SPLITS + 1)):
        raise ValueError("Invalid S4 outer fold plan.")

    def check(part, pool):
        fit, held = set(part.train), set(part.held_out)
        if (len(fit) != len(part.train) or len(held) != len(part.held_out)
                or fit & held or fit | held != pool):
            raise ValueError("An S4 partition repeats or misassigns a photo.")
        for field in ("patient_id", "group"):
            if {getattr(photos[i], field) for i in fit} & {getattr(photos[i], field) for i in held}:
                raise ValueError("An S4 partition splits a patient or linked component.")
        if any({photos[i].label for i in side} != {0, 1} for side in (fit, held)):
            raise ValueError("An S4 partition lacks a class.")

    indices = set(range(len(photos)))
    for split in plan:
        check(split.outer, indices)
        if set(split.outer.held_out) != {i for i, p in enumerate(photos) if p.fold == split.fold}:
            raise ValueError("S4 outer test differs from the fixed assignment.")
        outer_train = set(split.outer.train)
        for part in split.inner:
            check(part, outer_train)
        if (len(split.inner) != N_SPLITS
                or sorted(i for part in split.inner for i in part.held_out) != sorted(outer_train)):
            raise ValueError("S4 inner validation must cover every outer-training photo once.")


def _partition_predictions(photos, embeddings, part, fold, inner, factor, candidates, diagnostics):
    image_width = embeddings.shape[1]
    for model, x, held_x, y, weights, held in _design_matrices(photos, embeddings, part):
        # H1 constructs fresh matrices as [image, standardized age, site, optional symptoms].
        # The LR intercept is fitted internally and has no column in either matrix.
        x[:, image_width:] *= factor
        held_x[:, image_width:] *= factor
        context = {"outer_fold": fold, "inner_fold": inner, "model": model, "factor": factor}
        for c in candidates[model]:
            probs = _fit_predict(x, held_x, y, weights, c, context, diagnostics)
            yield model, c, mean_probabilities(held, probs)


def run_factor(photos, embeddings, plan, factor):
    """Run one prespecified multiplier, independently tuning A and B on the supplied H1 plan."""
    if type(factor) is bool or factor not in FACTORS:
        raise ValueError("The frozen S4 multipliers are 1, 3 and 10.")
    embeddings = np.asarray(embeddings)
    if (embeddings.ndim != 2 or embeddings.shape[0] != len(photos) or embeddings.shape[1] == 0
            or not np.isfinite(embeddings).all()):
        raise ValueError("Invalid or misaligned S4 image embeddings.")
    _check_plan(photos, plan)
    by_key = {p.key: p for p in photos}
    counts = Counter(p.key for p in photos)
    search = {model: C_GRID for model in MODELS}
    oof, tuning, diagnostics = {}, [], []
    for split in plan:
        predictions = {(model, c): {} for model in MODELS for c in C_GRID}
        for inner, part in enumerate(split.inner, 1):
            for model, c, means in _partition_predictions(
                    photos, embeddings, part, split.fold, inner, factor, search, diagnostics):
                predictions[model, c].update(means)

        training_keys = sorted({photos[i].key for i in split.outer.train})
        y = [by_key[k].label for k in training_keys]
        chosen = {}
        for model in MODELS:
            scores = {}
            for c in C_GRID:
                pooled = predictions[model, c]
                if pooled.keys() != set(training_keys):
                    raise ValueError("Incomplete inner validation lesion coverage for S4.")
                scores[c] = float(lesion_losses(y, [pooled[k] for k in training_keys]).mean())
            chosen[model] = select_c(scores)
            tuning.extend({"factor": factor, "outer_fold": split.fold, "model": model, "C": c,
                           "pooled_log_loss": scores[c], "selected": c == chosen[model],
                           "n_keys": len(y)} for c in C_GRID)

        refit = {model: (c,) for model, c in chosen.items()}
        for model, _, means in _partition_predictions(
                photos, embeddings, split.outer, split.fold, None, factor, refit, diagnostics):
            for key, probability in means.items():
                row = oof.setdefault(key, {"patient_id": key[0], "lesion_id": key[1],
                                           "outer_fold": split.fold, "label": by_key[key].label,
                                           "n_images": counts[key]})
                if row["outer_fold"] != split.fold or f"p_{model}" in row:
                    raise ValueError("Repeated S4 held-out lesion.")
                row[f"p_{model}"] = probability
    if oof.keys() != counts.keys() or any("p_A" not in r or "p_B" not in r for r in oof.values()):
        raise ValueError("Incomplete S4 paired predictions.")
    return [oof[k] for k in sorted(oof)], tuning, diagnostics
