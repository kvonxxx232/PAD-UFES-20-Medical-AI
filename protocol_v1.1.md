# Protocol v1.1: Dataset Audit and Complete-case Symptom Study

**Status: FROZEN.** Final inspection was completed on 2026-11-24. This revision records the completed image audit and metadata preflight. The preflight tested fold feasibility in memory. The production split and model fits are future work.

## 0. Status of the evidence

The V1.0 results and the pattern of unknown symptom values (`UNK`) were observed on this dataset before this protocol was written. V1.1 is a pre-specified follow-up on previously explored data. Its findings are limited to internal validation. V1.0 remains a historical baseline. Report V1.1 separately because the curation, training procedure and fold assignments differ.

## 1. Research question

Among lesions with all three symptom fields recorded, do the observed values of `grew`, `changed` and `bleed` improve cancer classification beyond image, age and anatomical site?

## 2. Hypothesis and interpretation

The primary estimand is **Delta Log Loss = LogLoss(B) − LogLoss(A)**, pooled over held-out symptom-complete patient–lesion keys, with equal weight per key. A is the image, age and site model. B adds the three symptoms. Lower log loss is better. The sole formal hypothesis of interest is an improvement with B (Delta < 0). Interpret the two-sided 95% interval below using its unrounded bounds.

| Interval for Delta | Interpretation within the evaluated cohort |
|---|---|
| Upper bound < 0 | B has lower log loss. An incremental predictive benefit was detected. |
| Lower bound ≤ 0 ≤ upper bound | The interval includes zero, including when either bound equals zero. The direction of an increment is inconclusive. Report the bounds and width to show the range of compatible effects. |
| Lower bound > 0 | B has higher log loss. Adding symptoms worsened predictive performance on this endpoint. |

There is **no equivalence margin**. An interval containing zero cannot establish equivalence or explain the source of the V1.0 gain. The numerical tolerance in Section 11 applies only to reproducing a computation. Secondary and exploratory analyses are descriptive and carry no additional formal tests or multiplicity adjustment.

## 3. Data and eligibility

- Use PAD-UFES-20, Mendeley Data version 1. Record the SHA-256 of the metadata and each source image archive. The two local metadata copies used in the audit had the same SHA-256, `14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527`.
- The analysis unit is the ordered pair `(patient_id, lesion_id)`. The original inventory contained 2,298 PNGs and 1,891 such keys. The 1,641 distinct lesion numbers in the local metadata match the published lesion count. Reuse of a lesion number across patients does not establish clinical identity. The upstream counting convention remains unverified.
- Complete the label-blinded image audit in Section 4 and record its pair and key sets before inspecting diagnoses or biopsy status. Then check the metadata by patient–lesion key. Keep the duplicate decisions fixed throughout these checks.
- Validate age as an integer in the preflight range of 0–120 years. Validate `region` against the 14 recognised site codes: ABDOMEN, ARM, BACK, CHEST, EAR, FACE, FOOT, FOREARM, HAND, LIP, NECK, NOSE, SCALP and THIGH. The six valid detailed diagnoses map to binary cancer labels: BCC, SCC and MEL are positive (1), and ACK, NEV and SEK are negative (0). Check detailed diagnoses as well as binary labels for agreement across all photos of each key.
- For a missing, invalid or conflicting age, site or diagnosis, stop processing before generating folds. Document the key, field and offending values for resolution. This includes conflicting detailed diagnoses that map to the same binary class. Resume only after a documented resolution. Silent imputation, overwriting and selection of the first record are prohibited.
- A **symptom-complete** key has an observed True or False value for each of `grew`, `changed` and `bleed` on every photo, with agreement within each symptom. Both True and False for the same symptom in one key trigger whole-key quarantine under the symptom-conflict rule. An ordinary `UNK` or missing response without contradiction makes the key incomplete and ineligible for primary training, tuning and evaluation. Incompleteness alone leaves the key in the curated cohort and never triggers duplicate-image quarantine.
- Primary inputs are the image embedding, age and anatomical site (`region`), plus the three symptom values for B. Exclude Fitzpatrick skin type, `biopsed`, missingness indicators, `itch`, `hurt`, `elevation`, diameters, smoking, drinking, socioeconomic and ancestry fields from both primary models.

## 4. Image duplicate audit and pre-fold curation

1. Match each metadata image ID to one readable PNG. For the image audit, use only the image bytes, decoded pixels and image/patient/lesion identifiers. Keep diagnosis, biopsy status, model predictions, performance measures and fold numbers out of candidate selection and visual adjudication.
2. Across the full inventory, compute SHA-256 on original file bytes. Independently compare decoded RGB content across **all images**, preserving native width, height, orientation and channel order. Group by dimensions and a digest of the unresized row-major RGB bytes, then verify dimensions and actual RGB byte equality within matching groups. Record each unordered pair across different patient–lesion keys once. Give file-byte matches precedence over RGB-only matches in the pair counts.
3. Compute a 64-bit pHash per image using the ImageHash `phash` convention (`hash_size=8`, `highfreq_factor=4`). Convert to grayscale, resize to 32 × 32 with Lanczos, apply the type-II DCT along both axes, and compare the top-left 8 × 8 coefficients, including DC, with their median using strict greater-than. Count differing bits. For distinct files across different keys, a Hamming distance **≤8 triggers label-blinded visual review only**. Preserve the historical candidate denominator by excluding SHA-256 matches and retaining RGB-only matches for reconciliation. Record one of three visual judgments per pair: likely near-duplicate, visually similar but distinct, or uncertain. Document corresponding local details such as markings, skin texture or background. A distance or a visual flag alone does not establish identical content.
4. **Quarantine every key on every side of a confirmed byte-identical or decoded RGB-identical relationship across different keys**, including all photos of each affected key, independently of diagnosis or label agreement. Take the union of keys across pairs and reasons. A visually similar pair with unequal decoded RGB pixels remains flagged pending documented label-blinded adjudication. Record the reviewer, local image evidence and disposition. Retain unresolved pairs in the candidate log without automatic exclusion. Record visually distinct dispositions there as well.
5. **Completed audit findings to reproduce** are 15 SHA-256-identical pairs involving 24 unique keys and one additional RGB-identical pair involving two new keys. The two files in the additional pair have different SHA-256 digests and decode to identical RGB pixels at the same native dimensions. Together these are **16 distinct pairs, 32 images, 13 patients and 26 unique patient–lesion keys**. All pairs were within patient and within the same original V1.0 fold. The ≤8 review covered 35 non-byte-identical cross-key pairs, including the one RGB-only pair. The other 34 were judged visually similar but distinct. These counts describe the completed audit on the recorded inputs. They are regression checks, never quotas for exclusion.
6. Apply the symptom-conflict rule in Section 3 as a separate reason. At implementation, retain sorted image-pair evidence and a sorted key-level exclusion manifest with every applicable rule ID. Report rule-specific counts, overlaps and the union. The 26-key finding is the **duplicate-only** union before combining exclusion reasons. Recompute the relationships from the PNGs on a clean run and compare pair IDs, key IDs, counts and manifest hashes. Investigate disagreements before generating folds.
7. Build patient groups from patient IDs and any confirmed cross-patient identical-content links established **before** quarantine, using connected components. Assign all retained keys of each component to one fold. Candidate-only and uncertain pairs create no group edges. The completed audit found no confirmed cross-patient link, so each component here is a single patient.

Reference SHA-256 digests from the completed duplicate audit permit a check of membership as well as counts. For pairs, put the two full image filenames in lexicographic order. For keys, retain the field order `patient_id`, `lesion_id`. In each set, sort the distinct two-string tuples lexicographically, join the fields with one tab and terminate every row with LF. Hash the resulting UTF-8 bytes without a header.

| Duplicate-only reference set | Rows | SHA-256 |
|---|---:|---|
| Confirmed image pairs | 16 | `e0456e137f1c3c3c1b8f0fa20f0c5aa6908e4111dec340d8963bd53abaf3fff8` |
| Affected patient–lesion keys | 26 | `2d1a9611aff51a12845f8b27f389d6fd46ab7aca577bed5c357ee828592ae25d` |

### Completed metadata preflight

The subsequent metadata check covered all 1,891 original keys. There were zero keys with missing, invalid or conflicting age or site. There were zero within-key conflicts in detailed diagnosis, binary cancer label or biopsy status, and zero True/False contradictions for any of the three symptoms. Of 389 keys with at least one `UNK`, two also belonged to the independently established 26-key duplicate set. Quarantining all photos of those 26 keys removed 42 images. Ordinary symptom missingness accounted for the remaining 387 keys outside the primary cohort.

| Population on the verified inputs | Keys | Patients | Images | Positive keys | Negative keys |
|---|---:|---:|---:|---:|---:|
| Curated cohort after quarantine | 1,865 | 1,368 | 2,256 | 816 | 1,049 |
| Primary symptom-complete cohort | 1,478 | 1,115 | 1,854 | 475 | 1,003 |
| Symptom-incomplete portion of the curated cohort | 387 | 318 | 402 | 341 | 46 |
| S3 biopsy-confirmed subset of the primary cohort | 638 | 508 | 903 | 475 | 163 |

Patients can contribute both complete and incomplete keys. S3 is nested within the primary cohort.

## 5. Patient-grouped splits

Use one row per retained key, in lexicographic order of the two identifier strings. On all curated keys, use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261124)` with the groups from Section 4. The tested four-stratum encoding is `2 * binary_label + int(symptom_complete)`, with True coded as 1 and False as 0. Assign each photo the fold of its key. Restrict these common outer folds to the population required for each analysis.

The completed in-memory preflight used scikit-learn 1.9.1 and NumPy 2.5.5. Each curated key appeared in one outer validation fold. All five outer validation folds contained all four strata. All 25 primary and all 25 S2 inner validation folds contained both classes among their symptom-complete keys. Patients were disjoint between training and validation at each split, and each outer test patient's records were absent from all corresponding inner splits. No split file was saved in the preflight.

When implemented, save and hash the V1.1 split table. Verify group disjointness, absence of quarantined keys, one held-out assignment per eligible key, class support for fitting, and counts by fold, class and completeness. Report infeasible partitions for methodological review. Fix the folds once, before model fitting, and retain them for every analysis.

## 6. Models and complete-case retraining

- Use the V1.0 frozen ImageNet-pretrained ResNet18 in evaluation mode, with its final classification layer replaced by the identity and the same RGB preprocessing and pretrained-weight transforms. Share one extraction pass across models. Fit an L2-regularised logistic regression head for A and for B.
- Use scikit-learn 1.9.1 `LogisticRegression(C=C, solver="lbfgs", l1_ratio=0.0, fit_intercept=True, class_weight=None, tol=1e-6, max_iter=5000)`. Pass the inverse-photo-count weights defined below as `sample_weight` to every inner fit and outer refit, including S2 and S4. For each fit record the model, fold, C, S4 factor if applicable, and `n_iter_`. Capture and report warnings, including `ConvergenceWarning`. Check convergence status and whether the iteration cap was reached. Document and resolve any unconfirmed convergence before using that fit for C selection or reporting predictions.
- Standardise age using the mean and population standard deviation across training photos, as in V1.0. Use a scale of one for a constant training age. Learn the site one-hot vocabulary from the same training partition. Encode an unseen test or validation site as an all-zero site vector. Encode the three observed symptoms as binary for B. Fit any data-dependent preprocessing anew inside each inner training split and during outer refitting. Give each training photo weight `1 / number of training photos in its key`, so every key contributes total weight one.
- For each outer fold, restrict **both** primary models to symptom-complete keys in the outer training patients. Construct five patient-grouped inner folds on these keys, using the same sorted-key convention, stratifying on binary outcome with shuffle seed **20261124**. Share inner splits between A and B. Select C separately for each model from **[0.001, 0.01, 0.1, 1, 10]**. For each C, pool the inner held-out predictions, aggregate to one probability per key as below and minimise equal-key **lesion-level log loss**. Use the smaller C for an exact tie in the unrounded score.
- Refit each head at its selected C on all symptom-complete outer-training keys. Predict each photo of the symptom-complete outer-test keys once. For both inner scoring and outer evaluation, take the **arithmetic mean of the per-image predicted positive-class probabilities** within each key. For log loss, clip this mean to `[1e-7, 1 − 1e-7]` and use natural logarithms. Use the pooled out-of-fold lesion predictions for the primary comparison.

## 7. Statistics

Report the pooled Delta, its two-sided percentile 95% CI, each outer-fold Delta and the number of folds favouring B. For the CI use **2,000 paired patient-cluster bootstrap resamples with seed 20261125**. Sample from the sorted unique evaluated patient IDs, with replacement, drawing as many patients as were evaluated. Carry all evaluated lesions of each drawn patient and the paired A and B predictions together. A patient drawn twice contributes all their lesions twice. Recompute the difference of the two equally lesion-weighted mean losses on each draw. Use NumPy's PCG64 generator and the 2.5th and 97.5th percentiles with linear interpolation. This interval describes variation among evaluated patients conditional on the fixed out-of-fold fits and splits. It omits uncertainty from model retraining or choosing a new partition.

## 8. Secondary analyses

- **S1.** Report B minus A for AUROC and Brier score on primary lesion predictions, with the same paired bootstrap draws. Higher AUROC and lower Brier score favour B. For an AUROC draw containing only one class, record it as undefined and report the number of defined draws used for the descriptive interval.
- **S2.** Train A and B on all curated outer-training keys. For B, encode an unknown symptom on a photo as 0 without adding an indicator. For inner tuning, train on all curated inner-training keys and score the symptom-complete inner-validation keys. Use the five-fold patient-grouped splitter, seed and four-stratum encoding from Section 5 on the full outer-training population. Evaluate on exactly the same symptom-complete outer-test keys as the primary analysis and report any disagreement. Use the same grid, weighting and probability aggregation throughout.
- **S3.** Describe the primary models' existing out-of-fold predictions on symptom-complete keys with `biopsed=True` on every photo. Exclude a key with inconsistent biopsy status from S3 until documented resolution. This restriction is specific to S3. Report subset size and class composition and use the primary model fits.
- **S4. Clinical-feature scaling sensitivity.** Repeat the complete-case A/B comparison with a common clinical-feature multiplier in **{1, 3, 10}**. After the usual preprocessing, multiply standardised age and every site column in both models, and the three symptom columns in B, by that factor. Keep the image embedding and intercept on their original scales. Apply the same factor at training and prediction, with no subsequent restandardisation. Each feature coefficient retains its L2 penalty. At a fixed C, increasing the multiplier weakens the effective penalty on the clinical effect expressed in the original units. Reuse the primary inner and outer splits and reselect C for each model and factor. Report Delta at all three factors. Factor 1 reproduces the primary comparison.
- **S6.** Describe calibration of each primary model's pooled out-of-fold lesion probabilities with a slope, intercept and reliability diagram. Use 10 equal-width probability bins over [0, 1], left-closed and right-open except for the last bin, which includes 1. For every bin report its key count. For occupied bins plot the mean predicted probability against the observed positive fraction. Mark empty bins with count zero and omit their point. Differences in subgroup prevalence limit comparisons of calibration-in-the-large.
- **S7.** Describe missingness in the full curated cohort before restriction to complete cases. Report Table 1 by symptom-unknown status (age, sex, site, biopsy status, diagnosis and photos per key) and the number of symptom fields with any unknown entry (0–3) against cancer rate. Give each key an any-UNK indicator equal to 1 if any of the three symptom fields is unknown or missing on any photo, and 0 otherwise. Among patients with at least two retained keys, report the patient count as the denominator and, as the numerator, the number whose keys all share one status (all 0 or all 1). Report the numerator divided by the denominator, with each patient counted once. This is a descriptive concordance statistic and supplies no evidence of a specific missingness mechanism.

## 9. Optional exploratory work

The former **S5** permutation analysis is optional and exploratory. If undertaken, document the exchangeability assumptions, strata and handling of sparse strata before running it. Permute the three symptoms jointly at key level within the specified strata, retaining the original draft's 200 permutations as a descriptive comparison with observed Delta. The comparison addresses only a partial null. Choosing age bands is not a prerequisite for the primary protocol. Label other work beyond the listed analyses, including image-based prediction of unknown status within class, as exploratory.

## 10. Limits on interpretation

Claims concern association, incremental prediction and uncertainty within PAD-UFES-20. Complete-case retraining targets lesions with recorded symptoms and can select a population with a different clinical or documentation profile. Biopsy confirmation is selective, and some negative reference diagnoses are clinical. Exact-image quarantine resolves analytic eligibility while the underlying clinical identity and correct lesion assignment can remain unknown. Blinded pHash review can miss transformed images outside the threshold. Prior exploration of V1.0, internal cross-validation and the fixed-fit bootstrap limit inference. The study cannot establish causality, a missingness shortcut, external validity or clinical utility. Report all prespecified results, including null or unfavourable ones.

## 11. Reproducibility

For the future V1.1 implementation, record the Python and package versions (including Pillow, NumPy, SciPy, scikit-learn, PyTorch and torchvision), the encoder checkpoint hash, the image and metadata checksums, audit/manifest and split hashes, seeds and selected C values. Provide one command for a clean-environment rerun of the curation, splits and prespecified tables. The deterministic image-pair and excluded-key sets and fold assignments must match on the same inputs. For the **primary Delta Log Loss** require `abs(Delta_rerun − Delta_reference) ≤ 0.001` log-loss units using unrounded values. Report both estimates and the absolute difference. This tolerance has no role in scientific interpretation of the CI.

## 12. Out of scope for V1.1

Fine-tuning, feature or modality dropout, recalibration and external validation belong to separately documented work.

## 13. Final inspection and freeze

Final inspection confirmed the eligibility rules, model specification and descriptive analyses against the agreed decisions and verified preflight counts. The annotated tag `protocol-v1.1-frozen` identifies the authorised protocol before production fold generation and model fitting. Document subsequent amendments with their date and reason.
