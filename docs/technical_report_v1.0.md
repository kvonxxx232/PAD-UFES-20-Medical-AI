# PAD-UFES-20 Medical AI technical report

Version 1.0.0

## Research questions and cohort

Can structured clinical information improve cancer classification from smartphone photographs? Do explicit unknown symptom indicators add predictive information beyond images, age, site, and affirmative symptom responses?

The positive class combines BCC, SCC, and MEL. The negative class combines ACK, NEV, and SEK. The local cohort contains 2,298 images, 1,373 patients, and 1,891 patient–lesion keys. Of these keys, 833 are positive and 1,058 are negative. The biopsy-confirmed subset contains 1,047 keys from 783 patients, including all 833 positive keys and 214 negative keys. The biopsy subset is an evaluation subset of the same out-of-fold predictions.

The official dataset reports 1,641 lesions. Locally, 1,641 distinct `lesion_id` values yield 1,891 `(patient_id, lesion_id)` combinations. Exactly 250 lesion numbers occur under two patients, and 197 of these pairs of records differ in binary label. Every `img_id` maps to one combination key. Single-column de-duplication would merge distinct patient records. The difference is consistent with an upstream count based on distinct lesion numbers. The official counting rule and a clinical identity mapping remain unverified. [Official dataset](https://data.mendeley.com/datasets/zr7vgbcyr2/1) · [Dataset paper](https://doi.org/10.1016/j.dib.2020.106221)

## Methods

- **Validation.** Five fixed patient-level folds contain 379, 379, 379, 377, and 377 analysis keys. Each fold is held out once. Patients and their images stay in one fold, and the classification heads use only training-fold labels.
- **Image features.** Clinical PNGs are converted to RGB and transformed with the torchvision ResNet18 default pretrained-weight transforms. A frozen ImageNet-pretrained ResNet18 in evaluation mode extracts features with its final classification layer replaced by an identity layer. Within each run, the models and folds share a single feature extraction pass.
- **Clinical features.** Each training partition determines the age mean and standard deviation and the site vocabulary. For `grew`, `changed`, and `bleed`, the affirmative indicator is 1 for `True` and 0 for `False` or `UNK`. The separate unknown indicator is 1 only for `UNK`.
- **Training.** A separately fitted linear head produces a single cancer logit. Heads share the initial weights for image features and the initial bias, under a PyTorch seed of 42. Clinical coefficients start at zero. Each head uses 100 full-batch AdamW steps with learning rate 0.01, weight decay 0.01, and binary cross-entropy with logits. Each photo receives weight 1 divided by its key's photo count so each key has equal total training weight.
- **Evaluation.** Mean sigmoid probability across a key's held-out photos gives its cancer score. Pooled AUROC and Log Loss use the held-out predictions from all five folds. Log Loss clips probabilities to `[1e-7, 1 - 1e-7]`. Paired percentile intervals use 2,000 bootstrap resamples of patients, retaining their eligible keys and using the same draw for both models. These intervals reflect variation in the evaluated patients at fixed trained predictions.

## Original evaluation

| Model | All AUROC | All Log Loss | Biopsy AUROC | Biopsy Log Loss |
|---|---:|---:|---:|---:|
| Image | 0.848643 | 0.484758 | 0.711472 | 0.600574 |
| Image + age + site | 0.861780 | 0.465143 | 0.733488 | 0.577685 |
| Image + age + site + unknown indicators | 0.898879 | 0.409293 | 0.771892 | 0.520276 |
| Full model / Model A | 0.919889 | 0.375021 | 0.809578 | 0.475962 |
| Model B | 0.879581 | 0.439335 | 0.761996 | 0.547493 |

The full model includes all image and clinical features described above. Its original per-fold AUROCs were 0.919472, 0.931222, 0.908485, 0.923714, and 0.917518. The archived summary contains the complete per-fold results. [Five-fold summary](../pad_ufes20_five_fold_results.json)

### Missingness ablation and potential bias

Model A is the full model and reproduces its saved predictions. Model B excludes the explicit unknown indicators while retaining the three affirmative symptom indicators and the other inputs. This encoding ablation leaves other predictors free to carry indirect information about symptom missingness.

| Evaluation set | A−B AUROC (95% CI) | A−B Log Loss (95% CI) |
|---|---|---|
| All keys | +0.04031 [0.03307, 0.04777] | −0.06431 [−0.07454, −0.05465] |
| Biopsy-confirmed keys | +0.04758 [0.03245, 0.06358] | −0.07153 [−0.08911, −0.05488] |

The exploratory missingness subgroup with at least one `UNK` contains 342 positive keys out of 389. The complete subgroup contains 491 positive keys out of 1,502. The A−B Log Loss differences are −0.23418 and −0.02032, respectively. At a fixed exploratory threshold of 0.5, moving from B to A recovers 73 cancer predictions and adds 12 false positives in the missing subgroup. In the complete subgroup, 33 previously detected cancers become false negatives. The case mix and the threshold affect these comparisons. Missing symptom values may be associated with clinical selection or documentation. These results establish a within-cohort predictive association and leave its cause and transferability open. [Ablation summary](../pad_ufes20_missingness_ablation_results.json)

### Melanoma false negatives

MEL is part of the positive class. It is represented by 52 photographs across 36 patient–lesion keys from 36 patients. All 36 keys are biopsy-confirmed. The following analysis uses the original held-out predictions, with a score of at least 0.5 counted as positive.

| Model | MEL keys classified as cancer | MEL false negatives | MEL sensitivity |
|---|---:|---:|---:|
| Image | 11/36 | 25 | 30.6% |
| Image + age + site | 12/36 | 24 | 33.3% |
| Image + age + site + unknown indicators | 12/36 | 24 | 33.3% |
| Full model / Model A | 14/36 | 22 | 38.9% |
| Model B | 12/36 | 24 | 33.3% |

This is sensitivity for classifying melanoma cases as cancer in a binary task. The threshold is exploratory and has no independent clinical validation. The small MEL sample limits interpretation.

## Exact-duplicate image audit

SHA-256 hashing of the local PNG files found 15 byte-identical pairs associated with 24 keys from 12 patients. Each pair has different image IDs and patient–lesion keys, and the two records belong to the same patient and fold. Twelve pairs differ in diagnostic category and seven differ in binary label. The audit detected no exact image pair spanning folds. Near-duplicates, including crops and re-encoded images, were outside its scope. The conflicts leave the correct clinical mapping and labels unresolved.

## Post-hoc data quality sensitivity analysis

All 24 affected keys were excluded from evaluation using the saved out-of-fold predictions. The training data and fitted models were the original ones, so this analysis measures sensitivity to evaluation inclusion only. All-key evaluation retained 1,867 keys and the biopsy-confirmed analysis retained 1,027.

| Model | All AUROC | All Log Loss | Biopsy AUROC | Biopsy Log Loss |
|---|---:|---:|---:|---:|
| Image | 0.850285 | 0.482473 | 0.714609 | 0.598283 |
| Image + age + site | 0.863719 | 0.462386 | 0.737638 | 0.574643 |
| Image + age + site + unknown indicators | 0.901619 | 0.404962 | 0.777670 | 0.514283 |
| Full model / Model A | 0.922394 | 0.370959 | 0.814865 | 0.470111 |
| Model B | 0.881221 | 0.436807 | 0.765451 | 0.544621 |

| Evaluation set | Filtered A−B AUROC (95% CI) | Filtered A−B Log Loss (95% CI) |
|---|---|---|
| All retained keys | +0.04117 [0.03402, 0.04837] | −0.06585 [−0.07559, −0.05615] |
| Retained biopsy-confirmed keys | +0.04941 [0.03396, 0.06415] | −0.07451 [−0.09129, −0.05670] |

The full model retained higher AUROC and lower Log Loss than the image baseline. Model A also retained higher AUROC and lower Log Loss than Model B in both sets. The effect of resolving these conflicts before model fitting remains unmeasured.

## Limitations and intended use

The evaluation is retrospective and internal to one dataset. Biopsy confirmation selects a subset with a different class mix. Some negative labels are based on clinical diagnosis, and the duplicate-image audit identifies unresolved label conflicts. Symptom missingness and other metadata may reflect local ascertainment and documentation. The observed performance gains do not establish causal effects, identify a leakage mechanism, or demonstrate external validity. No external or prospective validation was performed. The code and metrics are intended for research and education. See the [reproducibility notes](reproducibility_notes.md) for the remaining barriers to exact independent replication.
