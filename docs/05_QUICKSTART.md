# Quickstart

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Data

The common path uses dataset names directly:

```bash
python scripts/prepare_datasets.py --dataset visdrone
python scripts/prepare_datasets.py --dataset tinyperson
```

Roots:

- `CAPE_DATA_ROOT`: default parent for raw and prepared data.
- `VISDRONE_RAW_ROOT`: existing VisDrone raw data.
- `TINYPERSON_RAW_ROOT`: existing TinyPerson raw data.
- `TINYPERSON_DOWNLOAD_URLS`: optional comma-separated TinyPerson archive URLs.

VisDrone uses best-effort public DET archives when raw data is missing.
TinyPerson is prepared automatically from a local raw root or user-supplied
download URLs because stable public mirrors are not guaranteed.

Manual TinyPerson COCO-style fallback:

```bash
python scripts/prepare_datasets.py \
  --dataset tinyperson \
  --train-images /path/to/train/images \
  --train-json /path/to/train.json \
  --val-images /path/to/val/images \
  --val-json /path/to/val.json \
  --no-download
```

## Train

```bash
python scripts/train.py --dataset visdrone
python scripts/train.py --dataset tinyperson
```

Baseline:

```bash
python scripts/train.py --dataset visdrone --model-mode baseline
python scripts/train.py --dataset tinyperson --model-mode baseline
```

Smoke train:

```bash
python scripts/train.py --dataset visdrone --smoke
python scripts/train.py --dataset tinyperson --smoke
```

## Evaluate

```bash
python scripts/evaluate.py \
  --dataset visdrone \
  --checkpoint outputs/checkpoints/best.pt \
  --export-optional-curves \
  --measure-latency
```

## Benchmark

```bash
python scripts/budget_sweep.py \
  --dataset visdrone \
  --checkpoint outputs/checkpoints/best.pt \
  --export-optional-curves
```

Smoke report without a dataset:

```bash
python scripts/smoke_report_generation.py --output-dir outputs/smoke_reports
```

## Runner

```bash
bash run_cape_multidataset.sh --stage prepare --dataset both
bash run_cape_multidataset.sh --stage train --dataset visdrone
bash run_cape_multidataset.sh --stage eval --dataset both --checkpoint outputs/checkpoints/best.pt
bash run_cape_multidataset.sh --stage benchmark --dataset both --checkpoint outputs/checkpoints/best.pt
```

## Required Outputs

The benchmark protocol always preserves:

- 4 tables: unified detection, SAR, operating points, CAPE budget ablation.
- 3 figures: precision-recall, recall vs FP/image, confidence threshold.
- Optional CSV curves: PR by size, miss rate vs FP/image, PR under budget.

CAPE remains hypothesis-centric. Dataset preparation and evaluation must not
introduce patch routing, tile selection, crop-and-redetect, or region-first
stage-2 detection.
