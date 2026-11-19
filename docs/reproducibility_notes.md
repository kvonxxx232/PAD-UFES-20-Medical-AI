# Reproducibility notes

## Data and patient folds

Obtain the original PAD-UFES-20 metadata CSV and three PNG archives from the source record and extract the images locally. The scripts expect exact `img_id` filenames, with one metadata row per image in the local copy. Keep the source CSV row order for comparison with the archived predictions. The experiment used 2,298 images. The local metadata copy has SHA-256 `14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527`. The checksum identifies the local bytes used in the analysis. The equivalence of those bytes to a fresh official download remains to be checked. [PAD-UFES-20 dataset, version 1](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

The experiments consumed an existing local `pad_ufes20_patient_folds.csv` with columns `patient_id,fold` and one assignment per patient. The table contains 1,373 assignments across folds 1–5. Its SHA-256 is `b347e94b30a2bb8f574f166e899ab529baa619d3d79aec4d3a7967894e706b4c`. The codebase contains no fold generator or documented recipe for recovering the original assignments. The model seed of 42 controls model initialization and cannot recover these assignments. The table is excluded from the default public release pending a sharing decision. An approved copy of that table or a verified exact generator is required to reproduce the historical fold-level results.

## Environment and weights

The observed environment used Python 3.12.10 with PyTorch 2.15.0, torchvision 0.29.2, NumPy 2.5.5, and Pillow 13.0.0. The reported runs used CPU. [requirements.txt](../requirements.txt) pins the direct dependencies observed in that environment. It does not lock transitive dependencies or establish compatible builds across platforms.

The model uses torchvision's `ResNet18_Weights.DEFAULT` and its associated transforms. The historical local checkpoint was `resnet18-f37072fd.pth`, SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. Obtain the official torchvision checkpoint separately and verify this digest before running. The main script can download default weights if the torchvision cache is empty. To run it offline, place the checkpoint under `$TORCH_HOME/hub/checkpoints/` or the equivalent default cache location. The ablation script takes the local checkpoint path explicitly.

## Run from the project root

The paths below are examples. Set them to your local files, including the authorized fold table. Pick fresh output filenames because the scripts refuse to overwrite existing outputs.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

DATA_DIR=data
FOLDS=data/pad_ufes20_patient_folds.csv
export TORCH_HOME="$PWD/.cache/torch"
CHECKPOINT="$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth"
mkdir -p outputs

python pad_ufes20_full.py \
  --metadata "$DATA_DIR/metadata.csv" --folds "$FOLDS" \
  --images "$DATA_DIR/imgs_part_1" "$DATA_DIR/imgs_part_2" "$DATA_DIR/imgs_part_3" \
  --device cpu --output outputs/full.json --predictions-output outputs/full_predictions.csv

python pad_ufes20_missingness_ablation.py \
  --metadata "$DATA_DIR/metadata.csv" --folds "$FOLDS" \
  --images "$DATA_DIR/imgs_part_1" "$DATA_DIR/imgs_part_2" "$DATA_DIR/imgs_part_3" \
  --device cpu --checkpoint "$CHECKPOINT" \
  --output outputs/ablation.json --predictions-output outputs/ablation_predictions.csv
```

Expected counts are 1,891 unique prediction keys overall, with 379, 379, 379, 377, and 377 in folds 1–5. Compare the aggregate results with the archived [five-fold summary](../pad_ufes20_five_fold_results.json) and [ablation summary](../pad_ufes20_missingness_ablation_results.json).

## Remaining verification

- Verify installation, CPU execution, and output comparisons in a fresh environment. No cross-platform numerical tolerance has been established.
- Confirm data bytes against the upstream archive, resolve the upstream lesion counting rule, and document a shareable path to the exact fold assignments.
- The MEL and duplicate-image analyses were run from saved predictions. There is currently no tracked command that regenerates those tables. The technical report records their grouping, filtering, and scoring rules. Exact duplicate filtering uses the union of patient–lesion keys attached to any identical SHA-256 image group.
- The project code license is pending. The dataset's license is documented separately at its source. [PAD-UFES-20 dataset license](https://data.mendeley.com/datasets/zr7vgbcyr2/1)
