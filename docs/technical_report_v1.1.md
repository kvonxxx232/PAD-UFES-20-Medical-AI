# PAD-UFES-20 V1.1 — Dataset Audit and Complete-case Symptom Study

**Research technical report · Draft for methodological review · 24 November 2026**

## Summary

Among lesions with growth, change and bleeding recorded, did those three observations improve prediction of the dataset's binary cancer label beyond the photograph, age and anatomical site? In 1,478 eligible lesion keys from 1,115 patients, the image–age–site model had a pooled out-of-fold (OOF) log loss of **0.422859**. Adding the three symptoms reduced it to **0.355035**. The prespecified difference, **Δ = LogLoss(B) − LogLoss(A), was −0.067824** (paired patient-bootstrap 95% CI **−0.084488 to −0.050819**). All five outer folds favoured B.

The image audit identified 16 pairs with identical content across different lesion keys. Quarantine of all 26 affected keys left 1,865 keys for analysis. Among these, the cancer-positive fraction was 32.14% when all three symptoms were recorded and 88.11% when any was unknown. The primary result therefore applies to a selected population. V1.1 is a prespecified follow-up internal analysis of data already explored in V1.0. Independent confirmatory validation and clinical use remain outside the evidence presented here. [Frozen V1.1 protocol](../protocol_v1.1.md)

## 1. Historical motivation from V1.0

The V1.0 release examined 1,891 patient–lesion keys before V1.1 curation. Its missingness ablation kept image, age, site and affirmative symptom indicators in both models. One model also received explicit indicators that a symptom was unknown. Across the historical OOF keys, adding those indicators changed AUROC by **+0.04031** (95% CI +0.03307 to +0.04777) and log loss by **−0.06431** (95% CI −0.07454 to −0.05465). In the exploratory breakdown, 342/389 keys with an unknown symptom and 491/1,502 symptom-complete keys were cancer-positive. Thus the fact that a response was recorded carried predictive information in that dataset. [V1.0 technical report](technical_report_v1.0.md) · [Archived V1.0 ablation results](../pad_ufes20_missingness_ablation_results.json)

V1.0 also found identical image files under different lesion keys and assessed their exclusion after fitting. V1.1 moves the label-independent audit before fitting. On the resulting cohort it asks whether observed symptom values still contribute when every training and evaluation key has all three responses. H1 retrains both models on complete cases and excludes explicit missingness indicators. The earlier ablation, patient folds, analytic cohort and training procedure differ, so the V1.0 estimates above serve as historical motivation. All remaining comparisons in this report are within V1.1.

## 2. Data, audit and eligibility

PAD-UFES-20 version 1 contains smartphone photographs of skin lesions with clinical metadata. The binary outcome groups basal cell carcinoma (BCC), squamous cell carcinoma (SCC) and melanoma (MEL) as positive, and actinic keratosis (ACK), nevus (NEV) and seborrheic keratosis (SEK) as negative. ACK is a precancerous lesion category. Its negative label here is specific to the binary outcome defined for this experiment. The source describes the cancer cases as biopsy-proven and allows clinical diagnoses for some other lesions. It reports 1,641 lesions. In the local metadata, 1,641 distinct lesion numbers map to 1,891 distinct ordered patient–lesion keys because numbers recur across patients. V1.1 uses the ordered key throughout. The metadata alone cannot establish whether records sharing a lesion number refer to the same physical lesion or how the source counted those lesions. [PAD-UFES-20, Mendeley Data, version 1](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

### Image audit before clinical reconciliation

All 2,298 expected PNGs were matched and read. The audit used the image content and the identifiers needed to group records. Diagnosis, biopsy status, predictions and fold numbers were withheld from candidate selection and review.

| Audit step | Finding | Disposition |
| --- | --- | --- |
| Original PNG bytes | 15 cross-key pairs matched by SHA-256 and byte comparison, involving 24 unique keys | Quarantine every affected key, including its other photographs |
| Decoded image content | One additional pair had different file bytes and identical RGB bytes at the same native dimensions, involving two further keys | Apply the same whole-key rule |
| Visual candidate review | 35 non-byte-identical cross-key pairs had 64-bit pHash Hamming distance ≤8. These included the RGB-identical pair. An AI-assisted preliminary review classified the other 34 as visually similar but distinct. | The preliminary classifications were recorded without independent human sign-off. Similarity alone triggered no exclusion. |
| Deterministic union | **16 pairs, 32 images, 26 keys, 13 patients** | **42 photos removed**, including 10 additional photos belonging to affected keys |

RGB comparisons covered the full inventory at native resolution. Dimensions and a digest identified possible matches, followed by comparison of the decoded RGB bytes. Byte matches took precedence in counting. For pHash, the implementation converted to grayscale, resized to 32 × 32 with Lanczos interpolation, applied a type-II DCT on both axes and compared the top-left 8 × 8 coefficients, including DC, with their median. The threshold initiated label-blinded visual review. The recovered records document the preliminary visual assessment for each pair. All 16 deterministic pairs were within patient and within the same original V1.0 fold. None created a cross-patient grouping link.

### Metadata checks and cohort flow

After fixing the duplicate decisions, the pipeline checked age, site, detailed diagnosis, binary label, biopsy status and each symptom across photos of each key. No key had missing, invalid or conflicting age, site or diagnosis values. There were no within-key binary-label or biopsy conflicts and no contradictory True/False symptom records. A missing, invalid or conflicting age, site or diagnosis would halt processing. Contradictory symptom values would quarantine the whole key. An ordinary unknown response makes a key incomplete and leaves it in the curated cohort. Biopsy inconsistency would suspend eligibility for the biopsy-confirmed analysis pending resolution.

| Population | Patients | Lesion keys | Photos | Positive keys | Negative keys |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original inventory | 1,373 | 1,891 | 2,298 | 833 | 1,058 |
| Curated after quarantine | 1,368 | 1,865 | 2,256 | 816 | 1,049 |
| Symptom-complete, H1 population | 1,115 | 1,478 | 1,854 | 475 | 1,003 |
| Symptom-incomplete, retained | 318 | 387 | 402 | 341 | 46 |
| Biopsy-confirmed and complete, S3 subset | 508 | 638 | 903 | 475 | 163 |

A complete key required an observed and consistent True or False value for each of `grew`, `changed` and `bleed` on every photograph. Of 389 original keys with any unknown symptom, two had already qualified for duplicate quarantine. Sixty-five patients contributed both complete and incomplete retained keys. S3 is nested within the complete cohort.

## 3. Model fitting and statistical analysis

**Patient grouping and outer folds.** Connected components were defined by patient identity and any confirmed cross-patient identical-content links established before quarantine. Here each component contained one patient. The outer split used one row per curated key, sorted lexicographically by the two identifier strings. Five-fold `StratifiedGroupKFold` used patient components as groups, shuffling with seed **20261124** and four strata encoded by `2 * binary_label + int(symptom_complete)`. The saved assignment placed 373 curated keys in each fold. Each photograph inherited its key's fold. H1 and the secondary evaluations restricted these same outer assignments to their specified populations.

**Image and clinical inputs.** A frozen ImageNet-pretrained ResNet18 in evaluation mode converted each RGB photo to a 512-dimensional embedding. The pipeline used the pretrained torchvision transforms, including resizing the shorter edge to 256 pixels, a 224 × 224 centre crop and ImageNet channel normalisation. An identity layer replaced the final classifier. Compared models shared the embeddings within each run. The image encoder underwent **no fine-tuning**.

| V1.1 primary model | Inputs |
| --- | --- |
| A | Image embedding, standardised age and one-hot anatomical site |
| B | All A inputs plus the three observed binary values for growth, change and bleeding |

Age used the current training photos' mean and population standard deviation, with scale one for constant age. Site levels were learned on those same photos. An unseen evaluation site mapped to all zeros. Each inner training split and outer refit fitted its own preprocessing. Neither primary model used Fitzpatrick skin type, biopsy status, missingness indicators, additional symptoms or the remaining clinical and demographic metadata.

**Nested selection and refitting.** For each H1 outer fold, five inner folds split only symptom-complete outer-training keys, stratified by the binary label and grouped by patient component. The inner split used the same sorted-key convention and seed 20261124. A and B shared the partitions and separately selected `C` from **0.001, 0.01, 0.1, 1, 10**. For every candidate, inner validation predictions were pooled across folds. The score was the mean lesion-level log loss with equal weight per key. An exact tie in the unrounded score selected the smaller C. Each head was then refitted on the entire eligible outer-training set. All five outer and 25 H1 inner partitions passed patient/component isolation and class-support checks.

The head was scikit-learn 1.9.1 `LogisticRegression` with `solver="lbfgs"`, L2 (`l1_ratio=0.0`), `fit_intercept=True`, `class_weight=None`, `tol=1e-6` and `max_iter=5000`. Each training photo received weight one divided by the number of training photos in its key, giving each key total weight one. The positive probabilities of a key's held-out photos were averaged arithmetically. Log loss used natural logarithms after clipping that mean to [10⁻⁷, 1 − 10⁻⁷]. Each eligible key contributed one paired A/B OOF prediction.

**Primary estimand and uncertainty.** The sole prespecified primary hypothesis (H1) was that adding observed symptoms would reduce pooled lesion-level log loss. We report **Δ = LogLoss(B) − LogLoss(A)**. The two-sided percentile interval used 2,000 paired patient-cluster bootstrap resamples, with NumPy PCG64 seed **20261125**. Each draw sampled 1,115 times with replacement from sorted evaluated patient IDs. Every draw of a patient carried all that patient's evaluated keys and both predictions, including repeats. The statistic retained equal weight per sampled lesion. Bounds used the 2.5th and 97.5th percentiles with linear interpolation.

The frozen rule interprets an upper bound below zero as evidence of lower log loss with B, an interval containing or touching zero as inconclusive about direction, and a lower bound above zero as evidence of higher log loss with B. The protocol defines no equivalence margin. These intervals condition on the learned heads and fixed folds. All secondary analyses are descriptive, with no additional formal hypothesis tests.

## 4. Primary result (H1)

The primary comparison used all 1,478 complete test keys exactly once, including 475 positive and 1,003 negative keys. The pooled difference was **−0.067824** (95% CI **−0.084488 to −0.050819**). Its unrounded upper bound was below zero, meeting the prespecified criterion for incremental predictive benefit in this cohort.

| Outer fold | Patients | Keys | Positive keys | Log loss A | Log loss B | B − A | C for A | C for B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 224 | 296 | 95 | 0.436752 | 0.347922 | -0.088830 | 0.01 | 0.1 |
| 2 | 223 | 296 | 95 | 0.413056 | 0.353122 | -0.059935 | 0.01 | 0.1 |
| 3 | 223 | 295 | 95 | 0.444051 | 0.355490 | -0.088560 | 0.01 | 0.1 |
| 4 | 224 | 296 | 95 | 0.393565 | 0.354599 | -0.038966 | 0.1 | 0.1 |
| 5 | 221 | 295 | 95 | 0.426958 | 0.364073 | -0.062885 | 0.1 | 0.1 |
| **Pooled** | 1,115 | 1,478 | 475 | 0.422859 | 0.355035 | -0.067824 | — | — |

All 250 inner fits and 10 outer refits converged, with zero fit warnings and 26–1,261 iterations. C selection was completed using only inner validation data. The primary comparison is the pooled lesion estimate. The individual folds describe its consistency across the fixed split.

## 5. Prespecified secondary analyses

### S1. Discrimination and probability error

| Pooled measure | A | B | B − A | Descriptive 95% CI for B − A |
| --- | ---: | ---: | ---: | --- |
| AUROC | 0.873569 | 0.909961 | +0.036392 | +0.024831 to +0.048307 |
| Brier score | 0.134570 | 0.110884 | −0.023687 | −0.030059 to −0.017105 |

S1 used the original H1 lesion probabilities and the same 2,000 paired patient draws. All draws contained both classes, so all 2,000 AUROC differences were defined. B ranked positive lesions above negative lesions more often and had lower mean squared probability error. Both comparisons favour B on the same complete cases as H1.

### S2. Train on all curated keys and evaluate on the H1 test keys

A and B trained on both complete and incomplete outer-training keys. B encoded each unknown symptom as zero on the affected photo, as it did for False, while retaining observed True values and omitting missingness indicators. Five inner folds stratified on the four label/completeness strata and grouped on patient components. All inner-training keys entered the fits. Only complete inner-validation keys contributed to C selection. The 1,478 outer-test keys matched H1 exactly by key, patient, fold, label and photo count.

| Outer fold | Training keys (incomplete) | Complete test keys | Log loss A | Log loss B | B − A | C for A | C for B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,492 (310) | 296 | 0.475063 | 0.409573 | -0.065490 | 0.1 | 0.1 |
| 2 | 1,492 (310) | 296 | 0.429492 | 0.372123 | -0.057369 | 0.1 | 0.1 |
| 3 | 1,492 (309) | 295 | 0.427590 | 0.396052 | -0.031538 | 0.1 | 0.1 |
| 4 | 1,492 (310) | 296 | 0.412927 | 0.373172 | -0.039755 | 0.01 | 0.1 |
| 5 | 1,492 (309) | 295 | 0.439704 | 0.409974 | -0.029731 | 0.1 | 0.1 |

Pooled log loss was **0.436960** for A and **0.392164** for B, giving **Δ = −0.044796**. B had lower loss in all five folds. Relative to H1, the absolute reduction in loss was smaller by 0.023029 on the same evaluation keys. This comparison changes the training population and introduces a shared encoding for False and unknown. It cannot isolate the effect of that encoding. All 260 fits converged with zero warnings (30–1,390 iterations).

### S3. Biopsy-confirmed complete cases

S3 used the existing H1 predictions for keys with a recorded biopsy on every photo. It retained all 475 positive keys and 163 of 1,003 negative keys, for **638 keys from 508 patients**. The positive fraction rose from 32.14% in H1 to 74.45%. The subgroup used the original H1 model fits.

| Metric on biopsy-confirmed complete keys | A | B | B − A |
| --- | ---: | ---: | ---: |
| AUROC | 0.739787 | 0.808473 | +0.068686 |
| Log loss | 0.673371 | 0.579732 | -0.093638 |
| Brier score | 0.230020 | 0.190836 | -0.039184 |

| Outer fold | Keys | Positive | Negative | Δ AUROC | Δ log loss | Δ Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 118 | 95 | 23 | +0.105721 | -0.111284 | -0.053189 |
| 2 | 126 | 95 | 31 | +0.026146 | -0.069011 | -0.025373 |
| 3 | 131 | 95 | 36 | +0.077193 | -0.107204 | -0.052956 |
| 4 | 131 | 95 | 36 | +0.071930 | -0.073572 | -0.036744 |
| 5 | 132 | 95 | 37 | +0.061451 | -0.107823 | -0.028601 |

B improved all three scores in each fold. Both models had lower AUROC and higher probability losses in this subset than across all complete keys. The positive cases are the same, but the set of negatives against which they are ranked has changed. The class mix also changes the contribution of positive and negative cases to average loss. The larger subgroup differences do not establish that biopsy status modifies the symptom effect. This overlapping subset provides internal descriptive evidence only.

### S4. Clinical-feature scaling sensitivity

After fitting the usual training-partition preprocessing, we multiplied standardised age and all site columns by the same factor in both models, and the three symptom columns by that factor in B. The embedding and fitted intercept retained their original scales. The same transformation was applied at prediction, with no further standardisation. Every feature coefficient retained its L2 penalty. At a fixed C, scaling up a clinical feature weakens the effective penalty on its coefficient expressed in the original units. Each model selected C again at each factor using the H1 inner folds.

| Clinical multiplier | Log loss A | Log loss B | B − A | Folds favouring B | Iteration range |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.422859 | 0.355035 | -0.067824 | 5/5 | 26–1,261 |
| 3 | 0.413324 | 0.350838 | -0.062486 | 5/5 | 25–1,358 |
| 10 | 0.407337 | 0.338475 | -0.068862 | 5/5 | 33–2,165 |

| Outer fold | Δ at s = 1 | C at s = 1 (A / B) | Δ at s = 3 | C at s = 3 (A / B) | Δ at s = 10 | C at s = 10 (A / B) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.088830 | 0.01 / 0.1 | -0.091736 | 0.01 / 0.01 | -0.106399 | 0.01 / 0.01 |
| 2 | -0.059935 | 0.01 / 0.1 | -0.074643 | 0.01 / 0.01 | -0.092131 | 0.01 / 0.01 |
| 3 | -0.088560 | 0.01 / 0.1 | -0.064912 | 0.01 / 0.01 | -0.060306 | 0.01 / 0.01 |
| 4 | -0.038966 | 0.1 / 0.1 | -0.019933 | 0.01 / 0.1 | -0.020101 | 0.01 / 0.01 |
| 5 | -0.062885 | 0.1 / 0.1 | -0.061209 | 0.1 / 0.1 | -0.065331 | 0.01 / 0.01 |

The unrounded factor-one Δ and original H1 Δ were both **−0.06782443964243512**, an absolute difference of **0.0** against the frozen numerical tolerance of 0.001. The two OOF files were byte-identical. H1 retained its predictions, but did not persist its embedding matrix. S4 regenerated the embeddings once with the verified H1 weights, transforms, photo order and CPU batch size, and shared the matrix across all three factors. All 780 fits converged with zero warnings. The loss advantage had the same direction across factors, with some variation in magnitude. These comparisons neither establish equivalence between factors nor select an optimal clinical weight.

### S6. Calibration of the original OOF probabilities

For each model, we fitted an unpenalised logistic regression of the observed binary label on the logit of the original OOF probability, with one equally weighted row per lesion. The model was `logit Pr(Y = 1 | p) = α + β logit(p)`, with intercept and slope estimated jointly. Ideal values are α = 0 and β = 1. The joint intercept has a different interpretation from an intercept estimated with slope fixed at one. All original probabilities were strictly between zero and one, so their logits needed no clipping. We used the fitted coefficients solely to assess calibration. All reported probabilities retain their original OOF values.

| Model | Calibration intercept | Calibration slope | Mean predicted positive-class probability | Observed positive fraction |
| --- | ---: | ---: | ---: | ---: |
| A | +0.007649 | +1.043197 | 32.3604% | 32.1380% |
| B | -0.012399 | +0.976798 | 32.1432% | 32.1380% |

The reliability-diagram coordinates below use the mean prediction and observed positive fraction in each of ten fixed equal-width bins. Bins are left-closed and right-open, with 1 included in the last bin. All bins were occupied for both models.

| Probability bin | A keys | A mean predicted | A observed | B keys | B mean predicted | B observed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [0.0, 0.1) | 403 | 4.67% | 2.98% | 585 | 3.64% | 2.91% |
| [0.1, 0.2) | 268 | 14.58% | 6.72% | 195 | 14.76% | 15.38% |
| [0.2, 0.3) | 193 | 24.66% | 30.05% | 127 | 24.60% | 29.92% |
| [0.3, 0.4) | 127 | 34.52% | 40.16% | 76 | 34.82% | 32.89% |
| [0.4, 0.5) | 101 | 45.00% | 51.49% | 75 | 44.58% | 36.00% |
| [0.5, 0.6) | 95 | 55.01% | 58.95% | 63 | 54.66% | 61.90% |
| [0.6, 0.7) | 73 | 64.83% | 65.75% | 67 | 65.17% | 68.66% |
| [0.7, 0.8) | 86 | 75.09% | 77.91% | 65 | 75.06% | 80.00% |
| [0.8, 0.9) | 59 | 84.76% | 86.44% | 80 | 84.95% | 85.00% |
| [0.9, 1.0] | 73 | 94.97% | 84.93% | 145 | 95.87% | 91.72% |

Both slopes were close to one. On the logit scale, A's slope above one suggests that its predicted positive-class probabilities were slightly too close together, while B's slope below one suggests that its predictions were slightly too far apart. A single slope obscures local departures. In the highest bin, A predicted 94.97% on average and 62/73 keys (84.93%) were positive. B predicted 95.87% and 133/145 (91.72%) were positive. In the 0.4–0.5 bin A underestimated the event fraction while B overestimated it. The models assign different keys to a given bin. These estimates describe calibration in the same internal test predictions and do not validate a recalibrated model or establish calibration in a population with a different prevalence.

### S7. Descriptive epidemiology of missingness

S7 used all **1,865** retained keys before the primary restriction. A symptom counted as unknown for a key if it was unknown or missing on any of its photos. Each distinct symptom field counted at most once. The 387 keys with any unknown represented **20.75%** of the curated cohort. Their cancer-positive fraction was **88.11%**, compared with **32.14%** among complete keys. All 387 had a biopsy, compared with 638/1,478 (43.17%) complete keys. The full distribution of age, sex, site, diagnosis and photos per key appears in Table 1 in the appendix.

| Unknown fields per key | Keys | Positive | Negative | Positive fraction |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1,478 | 475 | 1,003 | 32.14% |
| 1 | 10 | 6 | 4 | 60.00% |
| 2 | 371 | 330 | 41 | 88.95% |
| 3 | 6 | 5 | 1 | 83.33% |

Of the 1,865 keys, **385 (20.64%)** had unknown growth, **379 (20.32%)** unknown change and **6 (0.32%)** unknown bleeding. In 371 of the 387 incomplete keys (95.87%), only growth and change were unknown. The one-field and three-field categories had just 10 and 6 keys, respectively.

Among **346 patients** with at least two retained keys, 237 had only complete keys, 44 had only incomplete keys and 65 had both. The fraction whose keys all shared one any-unknown status was **281/346 (81.21%)**, counting each patient once. Patient identifiers were used solely for grouping. Concordance describes the distribution of recorded fields and does not identify the timing or cause of missingness.

## 6. Interpretation and study limits

**What the four measures answer.** AUROC summarises how often a positive key ranks above a negative key, counting ties as one half. The numerical probability attached to that ranking can still be inaccurate. Log loss scores the probability assigned to the observed label and penalises confident errors particularly strongly. Brier score is the mean squared difference between predicted probability and the binary label. Both probability scores incorporate more than calibration, and their averages depend on the population being evaluated. Calibration asks whether observed event fractions agree with stated probabilities. Even when the mean predicted positive-class probability is close to the overall positive fraction, the mean prediction in a particular bin can exceed that bin's observed positive fraction. None of these measures, on its own, establishes an appropriate threshold or a clinical benefit.

**What the symptom comparison supports.** On complete cases, adding the three recorded responses to image, age and site reduced log loss for the predicted positive-class probabilities on held-out keys. The direction persisted under the prespecified changes to the training population and clinical feature scale. H1 does not separate the contributions of the three symptoms or estimate the effect of eliciting a symptom in clinical practice. The six prespecified secondary analyses provide context for H1. The optional exploratory permutation analysis (S5) was not undertaken.

**Documentation and selection.** S7 shows that unknown symptom status co-occurred with biopsy confirmation and a different diagnostic mix. Restricting to complete keys removed 341 cancer-positive and 46 cancer-negative keys. These counts help explain why complete-case retraining targets a particular documentation profile. They cannot distinguish clinical selection, documentation practices or other explanations. Neither the historical ablation nor the V1.1 results establish a causal missingness mechanism or demonstrate that a classifier learned a missingness shortcut. Selective biopsy and clinical reference diagnoses for some negatives further limit comparisons between evaluation populations.

**Residual image and modelling uncertainty.** Quarantine resolves how duplicated image content enters this analysis. The correct clinical lesion assignment and label for ambiguous source records remain unresolved. Transformed copies outside the perceptual-hash review threshold could have escaped detection. The label-blinded, AI-assisted preliminary review of pHash candidates had no independent human sign-off. Patient grouping reduces direct overlap but cannot establish independence from every shared acquisition or documentation practice. The fitted models used a frozen encoder and linear logistic heads throughout. H1's interval describes patient sampling variation conditional on those fitted models and this partition. Variation from training again on a different sample or choosing new folds is outside its scope.

**Scope of validation.** The protocol was frozen before V1.1 production fitting, after the dataset and V1.0 findings had been examined. Prespecification strengthens the interpretation of the stated H1 comparison within this cohort. Evidence of transportability, prospective performance, clinical thresholds and utility would require separate studies. This report documents a dataset audit and an internal symptom study.

## 7. Reproducibility and evidence available for review

The protocol freeze tag `protocol-v1.1-frozen` points to `67ebe7c830d8be409f16bcf54ab593a8609e97c9`. The H1 training run records a clean local code commit, `c62f33e1ce752df0f235a5f664a3425d8b6bafe5`. S2 and S4 record that base commit **and** separate source-file fingerprints for their local analysis code. Those additional sources are pending version control at the time of drafting. The runs used Python 3.12.10, NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.9.1, Pillow 13.0.0, PyTorch 2.15.0 and torchvision 0.29.2. The encoder ran on CPU with a batch size of 32.

| Reproduction object | SHA-256 |
| --- | --- |
| Frozen protocol text | `f208036381abd1c355d0b4be2e096b0cc01f4b826c5d4889b40f3ca0d763a39d` |
| Original metadata used in the audit | `14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527` |
| Confirmed image-pair set | `e0456e137f1c3c3c1b8f0fa20f0c5aa6908e4111dec340d8963bd53abaf3fff8` |
| Duplicate-key union | `2d1a9611aff51a12845f8b27f389d6fd46ab7aca577bed5c357ee828592ae25d` |
| Fixed V1.1 outer split table | `b115e884ea549b60c09a8a61a81212cd36bc25b2fb38ab9a5f42dfc848f7b013` |
| ImageNet ResNet18 checkpoint | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| Original H1 paired OOF file | `4687d825180a99967a3e4a9f32267526e5f8f4c0be23d0cc394fc2badff1ce3b` |

The frozen protocol specifies the canonical ordering and serialisation for the pair and key hashes. Curation retained per-PNG checksums and private case-level evidence. Hashes of the three original source archives remain unavailable, so archive-level provenance is incomplete. The outer assignment, model inputs, source fingerprints where recorded, saved OOF tables, inner scores and fit diagnostics supported the checks for this draft. H1, S2 and each S4 factor had 250 inner fits and ten outer refits, all with confirmed convergence and zero warnings.

S1 and S6 were recalculated from the sealed H1 OOF probabilities. S3 additionally used biopsy status from the verified metadata and retained-key membership. S7 used the verified metadata and Stage 2 membership. These four analyses were reconstructed in memory from existing inputs. They have no separate archived result packages. The project has stage-specific entry points for curation, splitting and model fitting. A tested single clean-environment command that reproduces the entire chain and all prespecified tables remains outstanding. The aggregate tables here can be checked without refitting the classifiers. [Protocol and reproduction requirements](../protocol_v1.1.md)

## 8. Figures proposed for the reviewed report

| Figure | Content |
| --- | --- |
| 1. Cohort and audit flow | Trace 2,298 photos and 1,891 keys through the two duplicate rules, whole-key quarantine, completeness restriction and nested biopsy subset. Show both photo and key counts. |
| 2. Primary effect by outer fold | Plot the five fold-specific H1 log loss differences and the pooled difference against zero. Add the patient-bootstrap interval only to the pooled point. |
| 3. Training and scaling sensitivities | Display H1 and S2 on the common complete test set in one panel. Show the three prespecified S4 multipliers and their fold-specific differences in another. |
| 4. Reliability diagrams | Plot each occupied bin's mean prediction against its observed positive fraction for each model. Use common axes, an identity line and counts from S6. |
| 5. Symptom recording and case mix | Show the positive fraction and denominator at each unknown-field count, with a separate panel for biopsy proportions in complete and incomplete keys. |

## Appendix. Table 1 — Curated cohort by symptom recording status (S7)

Categorical entries are counts and percentages of **all keys in the column**, including those with unrecorded sex. Each key contributes once to the age, site, sex, diagnosis and photograph summaries. The patient row counts distinct people within a column and includes 65 people who appear in both. SD denotes the sample standard deviation. Q1 and Q3 denote the 25th and 75th percentiles.

| Characteristic | Complete | Missing |
| --- | ---: | ---: |
| Lesion keys | 1,478 | 387 |
| Patients | 1,115 | 318 |
| Photographs | 1,854 | 402 |
| Cancer-positive keys | 475 (32.1%) | 341 (88.1%) |
| Age, years, mean (SD) | 59.85 (16.23) | 61.52 (14.73) |
| Age, median [Q1, Q3] | 62 [51, 72] | 63 [53, 72] |
| Sex, female | 394 (26.7%) | 195 (50.4%) |
| Sex, male | 372 (25.2%) | 192 (49.6%) |
| Sex, unrecorded | 712 (48.2%) | 0 (0.0%) |
| Biopsied | 638 (43.2%) | 387 (100.0%) |
| Biopsy status, False | 840 (56.8%) | 0 (0.0%) |
| Diagnosis, ACK | 632 (42.8%) | 6 (1.6%) |
| Diagnosis, BCC | 384 (26.0%) | 259 (66.9%) |
| Diagnosis, MEL | 23 (1.6%) | 13 (3.4%) |
| Diagnosis, NEV | 182 (12.3%) | 33 (8.5%) |
| Diagnosis, SCC | 68 (4.6%) | 69 (17.8%) |
| Diagnosis, SEK | 189 (12.8%) | 7 (1.8%) |
| Photographs per key, mean (SD) | 1.254 (0.515) | 1.039 (0.206) |
| Keys with one photograph | 1,145 (77.5%) | 373 (96.4%) |
| Site, abdomen | 25 (1.7%) | 5 (1.3%) |
| Site, arm | 124 (8.4%) | 26 (6.7%) |
| Site, back | 145 (9.8%) | 44 (11.4%) |
| Site, chest | 164 (11.1%) | 50 (12.9%) |
| Site, ear | 43 (2.9%) | 19 (4.9%) |
| Site, face | 378 (25.6%) | 96 (24.8%) |
| Site, foot | 6 (0.4%) | 7 (1.8%) |
| Site, forearm | 288 (19.5%) | 44 (11.4%) |
| Site, hand | 94 (6.4%) | 12 (3.1%) |
| Site, lip | 13 (0.9%) | 7 (1.8%) |
| Site, neck | 49 (3.3%) | 22 (5.7%) |
| Site, nose | 84 (5.7%) | 40 (10.3%) |
| Site, scalp | 16 (1.1%) | 2 (0.5%) |
| Site, thigh | 49 (3.3%) | 13 (3.4%) |

The descriptive standardised mean differences (Missing minus Complete, divided by the square root of the average group variance) were **+0.108** for age and **−0.549** for photographs per key. The large number of unrecorded sex entries among symptom-complete keys warrants particular care when interpreting the sex distribution.
