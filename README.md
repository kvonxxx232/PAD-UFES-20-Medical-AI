# PAD-UFES-20 Medical AI

## V1.1 — Dataset Audit and Complete-case Symptom Study

V1.0 found that an unknown symptom response carried information about the recorded cancer label. Among lesions with growth, change and bleeding recorded on every photograph, do the **observed responses** add predictive value beyond the image, age and anatomical site?

Before fitting, a label-blinded audit of 2,298 photographs identified 15 cross-key pairs with identical file bytes and one additional pair with identical decoded RGB pixels. Quarantine of all 26 affected patient–lesion keys left 1,865 keys. A 64-bit perceptual hash flagged candidates for review. The preliminary AI-assisted review of 34 other candidates had no independent human sign-off. Only confirmed identical content triggered image-based quarantine.

The primary analysis used only symptom-complete cases for training and evaluation. **Model A** used frozen ImageNet ResNet18 features, age and site. **Model B** added growth, change and bleeding. Five fixed outer folds held out entire patients. Within each outer training set, five patient-grouped inner folds selected the regularisation parameter `C` for each L2 logistic regression model by lesion-level log loss. Clinical preprocessing was fitted within each training partition. Each lesion contributed equal training weight, and its held-out prediction was the mean of its photo probabilities.

The pooled out-of-fold predictions covered **1,478 symptom-complete lesion keys from 1,115 patients**.

| Lesion-level measure | Model A | Model B |
| --- | ---: | ---: |
| Log loss (H1, primary) | 0.422859 | 0.355035 |
| AUROC (S1, descriptive) | 0.873569 | 0.909961 |

Adding the recorded symptoms lowered log loss in all five folds. The pooled **Δ log loss (B − A) was −0.067824** (95% CI **[−0.084488, −0.050819]**, 2,000 paired patient-cluster bootstrap resamples). The interval is conditional on these fits and folds. Six prespecified secondary analyses examined discrimination and probability error, training population, biopsy-confirmed cases, clinical-feature scaling, calibration and symptom missingness.

This was a prespecified internal follow-up on previously explored data. Symptom completeness selected a cohort with a different case mix. The association with cancer labels leaves the cause of missingness unresolved. Biopsy was selective, some negative labels relied on clinical diagnosis, and transformed duplicate images may have escaped detection. The image encoder remained frozen. External and prospective validation and clinical utility remain untested. Differences in curation, fold assignments and training methods prevent attributing score changes between V1.0 and V1.1 to a model upgrade. Intended use is research and education.

After the technical report was drafted, a clean-environment run reproduced the H1, S2 and S4 out-of-fold predictions byte-for-byte. The reproduction guide gives the tested single command and identifies the private historical evidence it requires. [V1.1 reproduction guide](docs/reproducibility_v1.1.md) · [V1.1 technical report](docs/technical_report_v1.1.md) · [Frozen research protocol](protocol_v1.1.md)

### Research figures

1. [Dataset curation and cohort flow](docs/figures/figure_1_dataset_curation_and_cohort_flow.svg)
2. [Primary effect by outer fold](docs/figures/figure_2_primary_effect_by_outer_fold.svg)
3. [Training population and clinical-feature scaling sensitivity](docs/figures/figure_3_training_population_and_clinical_feature_scaling_sensitivity.svg)
4. [Reliability diagrams for Models A and B](docs/figures/figure_4_reliability_diagrams.svg)
5. [Symptom missingness and case composition](docs/figures/figure_5_symptom_missingness_and_case_composition.svg)

## V1.0 Research Baseline

**Version 1.0.0 · Research code and evaluation summaries**

This project uses smartphone photographs and clinical metadata from PAD-UFES-20 to distinguish the dataset’s cancer group (BCC, SCC, MEL) from its non-cancer group (ACK, NEV, SEK). It asks whether age, anatomical site, and reported symptoms add predictive value beyond images, and how much of that value comes from explicitly encoding unknown symptom responses.

### Data and validation

The project evaluated 2,298 images from 1,373 patients. The dataset authors report 1,641 lesions. Here, 1,641 distinct `lesion_id` values produce **1,891 unique `(patient_id, lesion_id)` keys** because 250 lesion numbers each occur under two patients. The documentation gives the official total but leaves its counting rule unresolved. Results in this repository use patient–lesion keys as the analysis unit. [PAD-UFES-20 dataset, version 1](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

The model uses a **frozen ImageNet-pretrained ResNet18** to extract image features and a separately trained linear classification head. Each patient belongs to one of five fixed folds. Clinical preprocessing and head fitting use the training folds. Held-out photo probabilities are averaged within each patient–lesion key. The reported pooled metrics combine held-out predictions from all five folds.

### Main results

| Evaluation set | Image AUROC | Full-model AUROC | Image Log Loss | Full-model Log Loss |
|---|---:|---:|---:|---:|
| All 1,891 keys | 0.8486 | 0.9199 | 0.4848 | 0.3750 |
| 1,047 biopsy-confirmed keys | 0.7115 | 0.8096 | 0.6006 | 0.4760 |

The full model includes image, age, site, unknown indicators, and symptom responses. In the Missingness Ablation, Model A retains the unknown indicators and Model B removes them while retaining the symptom response columns. Across all keys, A−B AUROC was **+0.0403** (paired patient bootstrap 95% CI **[0.0331, 0.0478]**). This shows added predictive information in the evaluated cohort. Missing responses may reflect clinical or recording workflows and may behave differently in other settings. The ablation leaves indirect missingness signals available in the other inputs.

### Melanoma and data quality

The original evaluation included **36 melanoma (MEL) keys from 36 patients, represented by 52 images**. At the exploratory threshold of 0.5, the image baseline classified **11/36 (30.6%)** as cancer and the full model classified **14/36 (38.9%)** as cancer. This threshold had no independent clinical validation.

A SHA-256 audit identified **15 pairs of byte-identical images** associated with **24 analysis keys**. Seven pairs had conflicting cancer labels. Each exact pair stayed within a patient and fold. Near-duplicate images were outside the audit scope. An evaluation-only sensitivity analysis excluded the 24 keys from the existing predictions. The full model still outperformed the image baseline, and Model A still outperformed Model B, in both evaluation sets. The original training data were used for all reported predictions.

### Reproduction and use

See the [technical report](docs/technical_report_v1.0.md) for the methods, tables, and separate post-hoc results. See the [reproducibility notes](docs/reproducibility_notes.md) for the environment, dataset, and commands. Exact independent reproduction still depends on obtaining the original patient-fold assignments or recovering their generation procedure. A clean-environment reproduction has yet to be verified.

This is a retrospective, internally validated research project. Dataset selection, label quality, class imbalance, and symptom recording may limit generalizability. There is no external or prospective evaluation in this release. Intended use is research and education. Clinical diagnosis, triage, and treatment decisions are outside the intended use.

## License

The [MIT License](LICENSE) applies only to this project's original code. The original PAD-UFES-20 data and medical images are governed by the dataset's license. Third-party pretrained weights are governed by their respective terms. [PAD-UFES-20 dataset and license](https://data.mendeley.com/datasets/zr7vgbcyr2/1)
