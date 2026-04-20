# CAPE-Det

CAPE-Det is a research prototype for tiny-person detection on VisDrone and
TinyPerson under one unified SAR-oriented evaluation protocol.

The implementation follows `docs/` as the source of truth:

- Hypothesis-centric: CAPE refines compact human hypotheses, not patches,
  tiles, crops, or second-stage regions.
- Compositional: each hypothesis mixes learned primitive footprints.
- Degradation-aware: rendered footprints model tiny-person evidence after
  smoothing/blur-like degradation.
- Budgeted: refinement budget is allocated over hypotheses.
- Unified: VisDrone and TinyPerson share the same human-centric evaluator and
  benchmark schema.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The runner script performs the same setup automatically:

```bash
bash run_cape_multidataset.sh --stage prepare --dataset visdrone
```

## Dataset Roots

CAPE uses a raw cache and a prepared internal layout.

Environment variables:

- `CAPE_DATA_ROOT`: default root for `raw/` and `prepared/`.
- `VISDRONE_RAW_ROOT`: manual VisDrone raw root override.
- `TINYPERSON_RAW_ROOT`: manual TinyPerson raw root override.
- `TINYPERSON_DOWNLOAD_URLS`: optional comma-separated TinyPerson archive URLs.

Default prepared layout:

```text
data/
  raw/
    visdrone/
    tinyperson/
  prepared/
    visdrone/
      images/{train,val,test}/
      labels/{train,val,test}/
      metadata/visdrone_prepared.yaml
    tinyperson/
      images/{train,val,test}/
      labels/{train,val,test}.json
      metadata/tinyperson_prepared.yaml
```

VisDrone auto-download uses public VisDrone DET archives mirrored by
Ultralytics when no raw root is present. TinyPerson public mirrors are not
stable; CAPE auto-prepares from `TINYPERSON_RAW_ROOT`, manual COCO-style paths,
or URLs supplied through `TINYPERSON_DOWNLOAD_URLS`.

## One-Command Dataset Preparation

```bash
python scripts/prepare_datasets.py --dataset visdrone
python scripts/prepare_datasets.py --dataset tinyperson
python scripts/prepare_datasets.py --dataset both
```

Manual TinyPerson fallback:

```bash
python scripts/prepare_datasets.py \
  --dataset tinyperson \
  --train-images /path/to/train/images \
  --train-json /path/to/train.json \
  --val-images /path/to/val/images \
  --val-json /path/to/val.json \
  --no-download
```

## Training

Dataset names resolve automatically, prepare data if needed, validate the
prepared layout, and then continue into training.

```bash
python scripts/train.py --dataset visdrone
python scripts/train.py --dataset tinyperson
```

Baseline experiments are available with:

```bash
python scripts/train.py --dataset visdrone --model-mode baseline
python scripts/train.py --dataset tinyperson --model-mode baseline
```

The default settings are single-GPU friendly: batch size 2, AMP enabled,
gradient clipping enabled, small CNN/FPN backbone, `K=128` hypotheses, `A=64`
active hypotheses, and `T=3` refinement steps. Training skips prediction
decoding and final NMS by default because losses consume raw detector outputs
and CAPE hypotheses directly.

## Evaluation And Reports

```bash
python scripts/evaluate.py \
  --dataset visdrone \
  --checkpoint outputs/checkpoints/best.pt \
  --reports-dir outputs/reports \
  --figures-dir outputs/figures \
  --export-optional-curves \
  --measure-latency
```

Budget sweep for Table 4:

```bash
python scripts/budget_sweep.py \
  --dataset visdrone \
  --checkpoint outputs/checkpoints/best.pt \
  --reports-dir outputs/reports \
  --figures-dir outputs/figures \
  --export-optional-curves
```

Smoke report generation without datasets or torch:

```bash
python scripts/smoke_report_generation.py --output-dir outputs/smoke_reports
```

The reporting utilities emit CSV and markdown for exactly four tables:

1. `table1_unified_detection.csv` and `.md`
2. `table2_search_and_rescue.csv` and `.md`
3. `table3_operating_points.csv` and `.md`
4. `table4_budget_cape_ablation.csv` and `.md`

They emit PNG and CSV for exactly three required figures:

1. `fig1_precision_recall.png` and `.csv`
2. `fig2_recall_vs_fp_per_image.png` and `.csv`
3. `fig3_confidence_threshold.png` and `.csv`

Optional curve CSVs are written with `--export-optional-curves`:

- `pr_by_size.csv`
- `miss_rate_vs_fp_per_image.csv`
- `pr_under_budget.csv`

## Runner Scripts

```bash
bash run_cape_multidataset.sh --stage prepare --dataset both
bash run_cape_multidataset.sh --stage train --dataset visdrone
bash run_cape_multidataset.sh --stage train --dataset tinyperson
bash run_cape_multidataset.sh --stage eval --dataset both --checkpoint outputs/checkpoints/best.pt
bash run_cape_multidataset.sh --stage benchmark --dataset both --checkpoint outputs/checkpoints/best.pt
```

Bootstrap from outside a checkout:

```bash
bash bootstrap_cape_multidataset.sh --repo-dir cape --stage prepare --dataset visdrone
```

## Label Protocol

Primary benchmark mode is `human_unified_single`.

- VisDrone `pedestrian` maps to `person`.
- VisDrone `people` maps to `person` when `visdrone_people_policy: merge`,
  or is ignored when `visdrone_people_policy: ignore`.
- TinyPerson `person` maps to `person`.

Secondary mode is `human_split`, where VisDrone preserves `pedestrian` and
`people`, while TinyPerson remains `person`.

## Validation

```bash
python -m compileall cape_det scripts tests
python -m pytest -q
python scripts/sanity_check_dataset.py --dataset visdrone --split train
python scripts/sanity_check_dataset.py --dataset tinyperson --split train
python scripts/train.py --dataset visdrone --smoke
python scripts/train.py --dataset tinyperson --smoke
```

## Current Scope

This is a runnable research prototype, not a production detector. FLOPs and
energy/image are optional and remain `NaN` when unsupported. TinyPerson download
depends on a user-supplied mirror or local raw data, while VisDrone has a
best-effort public archive path.
