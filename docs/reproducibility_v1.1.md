# Reproducing the PAD-UFES-20 V1.1 study

The release check reproduced the frozen analyses in a clean Python environment. One command started from the original metadata and 2,298 extracted PNGs, regenerated the fixed folds, refitted H1, S2 and S4, and reproduced the prespecified tables. It required the private historical review and the H1 reference archive. The five figures were verified separately. This guide records release checks completed after the dated technical report. [Single-command run](#run-all-prespecified-analyses-with-one-command) · [Figure commands](#reproduce-the-five-figures) · [Technical report](technical_report_v1.1.md) · [Frozen protocol](../protocol_v1.1.md)

## Environment and source data

The release reproduction used CPython **3.12.10** on **macOS 14.7.1 arm64**, NumPy **2.5.5**, Pillow **13.0.0**, SciPy **1.18.1**, scikit-learn **1.9.1**, PyTorch **2.15.0** and torchvision **0.29.2**. The primary requirements include the curation, fold-generation and V1.0 encoder dependencies. The files pin the six direct Python packages. The installation below also pins the transitive dependencies used on this platform. OS libraries, fonts and the PNG renderer are separate. [Primary requirements](../requirements-v1.1-primary.txt) · [Fold requirements](../requirements-v1.1-folds.txt) · [Curation requirements](../requirements-v1.1-curation.txt) · [Encoder requirements](../requirements.txt)

Run the examples in **Bash from the root of a Git checkout** containing the launcher, analysis code, report and figure scripts. Training checks that its output is untracked and Git-ignored. The protocol is tagged `protocol-v1.1-frozen`. Stage 4B recorded code commit `c62f33e1ce752df0f235a5f664a3425d8b6bafe5`. Commit `bc0b151a03dbf018f562da3b05c4651d2a2058f7` archives the S2, S4 and secondary analysis code.

For a fresh environment on the tested platform, start in a shell with Python 3.12.10 available. Pick an unused environment name. The verified installation used pip 25.0.1 and 19 packages from PyPI wheels.

```bash
set -euo pipefail
umask 077
test ! -e .venv-v11
unset PYTHONPATH
export PYTHONNOUSERSITE=1
python3.12 -I -c 'import sys; assert sys.version_info[:3] == (3, 12, 10)'
python3.12 -I -m venv .venv-v11
source .venv-v11/bin/activate
python -m pip --isolated install --no-cache-dir --only-binary=:all: \
  --index-url https://pypi.org/simple -r requirements-v1.1-primary.txt \
  cloudpickle==3.1.3 filelock==4.2.12 fsspec==2026.9.0 Jinja2==3.1.6 \
  joblib==1.6.0 MarkupSafe==3.0.4 mpmath==1.3.0 narwhals==2.29.0 \
  networkx==3.7 setuptools==84.2.1 sympy==1.14.0 threadpoolctl==3.8.0 \
  typing-extensions==4.16.0
python -m pip check
```

Obtain the original metadata CSV and the three image parts from PAD-UFES-20 **version 1** (DOI `10.17632/zr7vgbcyr2.1`). Keep the downloaded metadata bytes and extract the PNGs with their original filenames. The audit inventories the three directories recursively. If the original archives are available, retain them and record their SHA-256 values locally. The three archive hashes from the historical analysis are unavailable. Its metadata digest and per-PNG inventory allow checks on the extracted inputs. [Dataset and licence](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

Obtain the official `ResNet18_Weights.IMAGENET1K_V1` checkpoint separately. The installed torchvision package exposes its download address as `ResNet18_Weights.IMAGENET1K_V1.url` in `torchvision.models`. The V1.1 CLI reads a local checkpoint and verifies its SHA-256. The encoder ran on CPU in batches of 32 with frozen weights. [Torchvision ResNet18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

Set these paths once in the same shell. Adapt the data and checkpoint locations to your copy. `ARCHIVE` contains the original private records. Choose an unused `RUN` name under the repository's ignored `private/` directory for new outputs.

```bash
DATA_DIR=data/PAD-UFES-20
METADATA="$DATA_DIR/metadata.csv"
IMAGE_DIRS=("$DATA_DIR/imgs_part_1" "$DATA_DIR/imgs_part_2" "$DATA_DIR/imgs_part_3")
CHECKPOINT=weights/resnet18-f37072fd.pth
ARCHIVE=private
REVIEW="$ARCHIVE/v1.1-review/recovered_review.json"
H1_REF="$ARCHIVE/v1.1-primary-stage4b"
RUN=private/reproduction-01
CURATION="$RUN/curation"
FOLDS="$RUN/outer-folds"
```

## Private inputs and reference checks

Restore archived files from an authorised local backup or request private access from the project maintainer. Keep the images, clinical metadata, pair-level review evidence, split assignments and lesion-level predictions out of Git. The public report and figures contain aggregates.

| Directory under `ARCHIVE` | Contents and recovery |
| --- | --- |
| `v1.1-curation/` | Stage 2 regenerates the PNG inventory, confirmed pairs, duplicate keys, pHash queue, quarantine manifest, curated keys and their summary from the metadata and three image parts. |
| `v1.1-review/` | Restore the exact `recovered_review.json` from the historical label-blinded review. It maps all 34 pending pairs to dispositions, pair-specific evidence and source records from the AI-assisted preliminary review. There was no independent human sign-off. Stage 2 creates the queue only. Stage 3 stops if this separate record is missing or fails its digest or evidence checks. |
| `v1.1-outer-folds/` | Stage 3 regenerates `splits.csv` and `fold_summary.json` from the verified curation, metadata and review. The historical V1.0 patient-fold table has a different assignment. |
| `v1.1-primary-stage4b/` | The original H1 package contains the paired OOF file, inner scores, fit diagnostics, summary and a separate `run_provenance.json`. H1 training writes the first four. S2, S4, the secondary CLI and Figures 2 and 3 check the digests of the original H1 OOF, summary and run record. Restore the exact archived bytes for these checks. |
| `v1.1-secondary-s2/`, `v1.1-secondary-s4/` | S2 and S4 training generate new run packages. To reproduce Figure 3, restore the original summaries and OOF files, including S4 subdirectories `s1/`, `s3/` and `s10/`. The plot authenticates the archived summaries, which include run-specific provenance such as timestamps. |

| Reference object | SHA-256 |
| --- | --- |
| Original metadata used in V1.1 | `14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527` |
| 16 confirmed image pairs, canonical set | `e0456e137f1c3c3c1b8f0fa20f0c5aa6908e4111dec340d8963bd53abaf3fff8` |
| 26 duplicate-related lesion keys, canonical set | `2d1a9611aff51a12845f8b27f389d6fd46ab7aca577bed5c357ee828592ae25d` |
| ResNet18 checkpoint | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| Recovered private review | `f264121493e53b0602c0c35f847224518d466ee486466f9cea09dfc3aa4e69a9` |
| Fixed V1.1 outer split table | `b115e884ea549b60c09a8a61a81212cd36bc25b2fb38ab9a5f42dfc848f7b013` |
| Original H1 paired OOF file | `4687d825180a99967a3e4a9f32267526e5f8f4c0be23d0cc394fc2badff1ce3b` |
| Original H1 summary | `c6362c4f367ce128525070e0af44a715434906e130ee1d8ebc0ab9ea1cb32f46` |
| Original H1 execution record | `2e330227a2cc725c393b0e60eab66fa8d95ed49f58eb401e514600d18b05a169` |

## Run all prespecified analyses with one command

Use the paths defined above and restore the historical review and H1 reference first. The dry run checks the protocol, software versions, metadata, every PNG, checkpoint, review, historical H1 files and output destination. It reads the report and records its digest. The output path must be new. [Reproduction launcher](../reproduce_v11.py)

```bash
ARGS=(--metadata "$METADATA" --images "${IMAGE_DIRS[@]}" --checkpoint "$CHECKPOINT"
      --review "$REVIEW" --primary-reference "$H1_REF"
      --report docs/technical_report_v1.1.md --output-dir "$RUN")
python -B reproduce_v11.py --dry-run "${ARGS[@]}"
python -B reproduce_v11.py "${ARGS[@]}"
```

The second invocation repeats the input checks, curates the raw data, generates and verifies the fixed folds, then fits H1. It compares the unrounded primary Δ against the archived run before fitting S2 and the three S4 factors. Finally, it computes S1, S3, S6 and S7 and checks them against the report. The individual commands below also support separate stage runs. Choose a different unused run directory for those commands.

Only `completed.json` marks the full run complete. It records both H1 estimates and their absolute difference, the secondary report comparison and the digests of the newly generated outputs. The H1, S2 and S4 summaries contain the pooled and fold-specific estimates and selected C values. The four other analyses are in `secondary.json`. The private command journal records each invocation and exit code. A failed check stops subsequent stages and leaves the available logs for inspection.

The historical record authenticates the original H1 execution. The launcher links the new H1 OOF and summary together with that original record in a private reference directory and records the source of each file. The downstream CLIs require the new OOF and summary to match their historical hashes exactly. A new H1 fit can satisfy the protocol’s numerical tolerance while failing this additional archive check. In that case the H1 comparison remains available and the full run stops.

## Repeat curation, splitting and model fitting

### Stage 2 — Curate the data

```bash
python pad_ufes20_curate.py --metadata "$METADATA" --images "${IMAGE_DIRS[@]}" \
  --output-dir "$CURATION"
```

The program checks file-byte and native-size decoded RGB equality across different patient–lesion keys before reconciling clinical fields. A Hamming distance of 8 or less between 64-bit pHashes flags pairs for label-blinded review. The frozen reference has 16 confirmed pairs, 26 quarantined keys and 1,865 retained keys, of which 1,478 are symptom-complete. A successful `summary.json` records checksums for the private tables. The protocol specifies the pair-set and key-set digests. [Frozen protocol](../protocol_v1.1.md)

The command above matches the historical run's archive-hash field. To record a new download's three ZIP hashes in a separate curation run, supply `--archives ZIP1 ZIP2 ZIP3` in part order and choose another output directory. This changes the summary digest even if the extracted PNGs match. Downstream checks also authenticate the historical summaries.

### Stage 3 — Generate and verify the fixed outer folds

```bash
python pad_ufes20_folds.py --metadata "$METADATA" --curation "$CURATION" \
  --review "$REVIEW" --output-dir "$FOLDS"
python pad_ufes20_folds.py --verify --metadata "$METADATA" --curation "$CURATION" \
  --review "$REVIEW" --output-dir "$FOLDS"
```

The generator uses five-fold `StratifiedGroupKFold` on sorted lesion keys, grouping linked patients and stratifying on `2 * binary_label + int(symptom_complete)` with shuffle seed **20261124**. It checks patient and group disjointness and complete coverage. Compare the printed split digest with the table above. The second command regenerates the table in memory and checks the saved bytes and summary. To verify an existing archive, run just the second command with `--curation "$ARCHIVE/v1.1-curation"` and `--output-dir "$ARCHIVE/v1.1-outer-folds"`.

### Stage 4B — Run or compare the primary complete-case fit

With the verified Stage 2 and Stage 3 directories, define the common fit inputs. `preflight` checks the inputs and split feasibility. `train` fits the models and requires a fresh private destination.

```bash
FIT_INPUTS=(--metadata "$METADATA" --images "${IMAGE_DIRS[@]}" --checkpoint "$CHECKPOINT"
            --curation "$CURATION" --review "$REVIEW" --fold-dir "$FOLDS")
python pad_ufes20_primary.py preflight "${FIT_INPUTS[@]}"
python pad_ufes20_primary.py train "${FIT_INPUTS[@]}" --output-dir "$RUN/h1"
python pad_ufes20_primary.py compare "$H1_REF/summary.json" "$RUN/h1/summary.json"
```

H1 holds out one fixed outer fold at a time. Five patient-grouped inner folds on complete training keys select `C` from `[0.001, 0.01, 0.1, 1, 10]` by pooled lesion-level log loss. Age and site preprocessing uses each training partition. Every key has equal total training weight across its photos. Both heads use the same frozen ResNet18 features, and each held-out key receives the mean of its photo probabilities. Inspect the private inner scores, fit diagnostics and final summary. The comparison command requires two distinct completed summaries with matching input provenance and checks that the absolute difference in unrounded primary Δ log loss is at most **0.001**.

### Run S2 and S4

Use the same `FIT_INPUTS` and the **archived original H1** as the evaluation reference. A new H1 fit within the numerical tolerance does not supply the original provenance record. Each of these `train` commands writes to its own fresh directory.

```bash
python pad_ufes20_s2.py preflight "${FIT_INPUTS[@]}" --primary-dir "$H1_REF"
python pad_ufes20_s2.py train "${FIT_INPUTS[@]}" --primary-dir "$H1_REF" --output-dir "$RUN/s2"
python pad_ufes20_s4.py preflight "${FIT_INPUTS[@]}" --primary-dir "$H1_REF"
python pad_ufes20_s4.py train "${FIT_INPUTS[@]}" --primary-dir "$H1_REF" --output-dir "$RUN/s4"
```

S2 trains on all curated outer-training keys, with False and unknown symptoms both encoded as zero in B. Inner validation scores and outer evaluation use complete keys. S4 repeats complete-case training at clinical-feature multipliers **1, 3 and 10**, reselecting `C` on the H1 inner folds for each model and multiplier. Both analyses score the same held-out keys as H1.

## Recompute S1, S3, S6 and S7

With the archived inputs restored, this command prints aggregate JSON and checks results against the technical report at its printed precision. It requires the original metadata and private records. It reads neither raw images nor encoder weights and does not refit A or B.

```bash
python pad_ufes20_secondary.py --metadata "$METADATA" --curation "$ARCHIVE/v1.1-curation" \
  --review "$REVIEW" --fold-dir "$ARCHIVE/v1.1-outer-folds" --primary-dir "$H1_REF" \
  --report docs/technical_report_v1.1.md
```

S1 recomputes AUROC and Brier score with paired patient bootstrap differences. S3 restricts the original OOF rows to the biopsy-confirmed complete-case subset. S6 estimates diagnostic calibration intercepts and slopes and reports ten fixed equal-width bins with the original probabilities. S7 summarises symptom recording and case mix on all retained keys. An input or report mismatch exits with a nonzero status.

## Reproduce the five figures

The report's SVGs can be viewed in a browser. Each plotting command creates both an SVG and a PNG and needs a PNG renderer. Use a new figure directory for each verification so the reviewed figures stay intact. Figure 4's script is at the repository root. All five read existing aggregates or recompute them from archived inputs.

**PNG rendering.** The release check used the clean Python environment with Corretto JDK 11.0.26 and existing Apache Batik JARs distributed with MATLAB. It did not start MATLAB. For the Java route, set `BATIK_CLASSPATH` to the Batik JARs and dependencies and put the JDK on `PATH`. The scripts also support `rsvg-convert` on `PATH` or an importable CairoSVG with its Cairo runtime. Those two routes require no MATLAB software or `BATIK_CLASSPATH` and were inspected in source only. Different fonts or renderers can change the output bytes. [Shared renderer](figures/plot_figure_1_cohort_flow.py) · [Figure 4 renderer](../plot_figure_4_reliability.py)

```bash
FIGURES=$(mktemp -d)
python docs/figures/plot_figure_1_cohort_flow.py --summary "$ARCHIVE/v1.1-curation/summary.json" \
  --output-dir "$FIGURES"
python docs/figures/plot_figure_2_primary_effect.py --primary-dir "$H1_REF" --output-dir "$FIGURES"
python docs/figures/plot_figure_3_sensitivity.py --private-dir "$ARCHIVE" --output-dir "$FIGURES"
python plot_figure_4_reliability.py --oof "$H1_REF/oof_predictions.csv" \
  --run-summary "$H1_REF/summary.json" --output-dir "$FIGURES"
python docs/figures/plot_figure_5_symptom_missingness.py --metadata "$METADATA" \
  --curation "$ARCHIVE/v1.1-curation" --output-dir "$FIGURES"
```

The example above uses the original private directory layout. The clean-environment validation substituted the following inputs, retaining the plotters' original authentication checks throughout.

| Figure | Inputs supplied in the clean-environment rendering |
| --- | --- |
| 1 · Cohort flow | New Stage A curation summary, frozen protocol and report. |
| 2 · Primary effect | New Stage A H1 OOF and summary, plus the historical H1 execution record. |
| 3 · Training population and scaling | New H1, S2 and three S4 OOF files, new H1 and three factor summaries, plus the historical H1 execution record and original S2/S4 top-level summaries. The plotter checks the exact top-level summary bytes. Substituting each new top-level summary was separately tested and rejected. Their start time, finish time and Git base commit differed. Their computational results matched. |
| 4 · Reliability | New Stage A H1 OOF and summary, with all 20 bins compared with the report. |
| 5 · Missingness | New Stage A curation, authenticated original metadata, frozen protocol and report. |

To plot directly from a successful single-command run, use `$RUN/curation` for Figures 1 and 5, `$RUN/authenticated-h1` for Figure 2, and `$RUN/h1` for Figure 4. For Figure 3, provide the following private input layout. These commands accept an archive directory and run directory under the project root, as in the examples above. They reference the untouched historical top-level summaries and take the predictions and factor results from the new run.

```bash
VIEW=$(mktemp -d "$RUN/figure3-inputs.XXXXXX")
NEW=$(cd "$RUN" && pwd -P)
REF=$(cd "$ARCHIVE" && pwd -P)
mkdir "$VIEW/v1.1-secondary-s2" "$VIEW/v1.1-secondary-s4"
ln -s "$NEW/authenticated-h1" "$VIEW/v1.1-primary-stage4b"
ln -s "$REF/v1.1-secondary-s2/summary.json" "$VIEW/v1.1-secondary-s2/summary.json"
ln -s "$NEW/s2/oof_predictions.csv" "$VIEW/v1.1-secondary-s2/oof_predictions.csv"
ln -s "$REF/v1.1-secondary-s4/summary.json" "$VIEW/v1.1-secondary-s4/summary.json"
for factor in 1 3 10; do
  ln -s "$NEW/s4/s$factor" "$VIEW/v1.1-secondary-s4/s$factor"
done
FIGURES=$(mktemp -d)
python docs/figures/plot_figure_3_sensitivity.py --private-dir "$VIEW" --output-dir "$FIGURES"
```

## What has been reproduced

| Evidence | Scope of verification |
| --- | --- |
| Curation and fixed folds · Stage A passed | From the 2,298 extracted PNGs and original metadata, the clean environment regenerated all six curation CSVs, the curation summary, the outer split and its summary byte-for-byte. There were 16 confirmed cross-key duplicate pairs and 26 quarantined keys. The 1,865 retained keys covered 1,368 patients and 2,256 photographs. Patient and linked-group overlaps across folds were both zero. |
| H1 · Stage A passed | The fresh fit on 1,478 complete keys reproduced the historical OOF, inner scores and diagnostics byte-for-byte. New and historical unrounded Δ were both **−0.06782443964243512**, an absolute difference of **0.0** against the frozen tolerance of **0.001**. The pooled loss, five fold estimates and patient-bootstrap interval were also independently recalculated from the new OOF. |
| S2 and S4 · Stage B passed | The clean-environment fits reproduced all four historical OOF files, inner scores and diagnostics byte-for-byte. Unrounded A/B losses and Δ matched for every fold and pooled comparison. All 1,040 fits converged without warnings. |
| S1, S3, S6 and S7 · Stage B passed | The secondary CLI used the Stage A OOF and authenticated cohort inputs. Eight report tables (66 data rows) and 14 additional numerical passages matched at printed precision. |
| Single-command run · Stage C passed | A read-only dry run checked 2,298 PNGs and left the requested output directory unused. A subsequent invocation ran all 11 steps from curation to the secondary tables. All five sets of paired OOF predictions, inner scores and fit diagnostics matched the historical bytes. All 1,300 fits converged without warnings. The unrounded H1 Δ difference was 0.0. |
| Figures · Stage C passed | All five figures matched the report numerically. The Stage A/B outputs and the later single-command outputs were each used to regenerate the five SVGs and five PNGs. Both sets matched the reviewed files byte-for-byte with the tested renderer. Decoded PNG pixels matched, and all five renders were inspected for labels, layout and readability. |

For the single-command run, the secondary CLI checked eight report tables (66 data rows) and 14 additional numerical passages. A separate check compared five more tables (24 rows) covering cohort flow, H1, S2 and S4 against the new outputs. The 97 automated regression tests passed. Two additional launcher checks confirmed that an existing destination and a review file with the wrong digest both stop the dry run.

The verified scope is a clean Python environment on the recorded Apple Silicon platform, using the original extracted files and authorised private evidence. The three historical source ZIP checksums remain unavailable. The Stage 3 gate depends on the archived AI-assisted review, the downstream analyses authenticate the historical H1 execution record, and Figure 3 requires the archived S2 and S4 top-level summaries. Keep these private, along with case-level files and command logs. Reproduction from a fresh upstream download alone, cross-platform numerical replication and external independent clinical validation remain unverified.
