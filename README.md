# CAPE-Det

CAPE-Det is a research prototype for tiny-person detection on VisDrone and
TinyPerson under one unified SAR-oriented evaluation protocol.

The implementation follows the docs in `docs/`:

- CAPE is hypothesis-centric: the unit of computation is a compact human
  hypothesis, not a patch, tile, crop, or second-stage region.
- CAPE is compositional: each hypothesis mixes learned primitives.
- CAPE is degradation-aware: rendered footprints model tiny-person evidence
  after blur/smoothing.
- CAPE is budgeted over hypotheses: refinement depth is allocated to active
  hypotheses only.
- VisDrone and TinyPerson share the same primary human-centric evaluation.

## Repository Structure

```text
configs/              YAML configs for datasets, models, experiments, ablations
cape_det/datasets/    VisDrone/TinyPerson parsers and unified label mapping
cape_det/models/      Lightweight baseline detector and CAPE branch
cape_det/losses/      Detector, CAPE, matching, and composite losses
cape_det/metrics/     Unified AP/SAR/threshold metrics, tables, curves, latency
cape_det/trainers/    Training loop, builders, checkpoints
cape_det/utils/       Config, logging, NMS, profiling, visualization helpers
scripts/              CLI entrypoints
tests/                Minimal parser, model, loss, evaluator, reporting tests
```

## Dataset Layout

Set dataset paths in `configs/datasets/visdrone.yaml` and
`configs/datasets/tinyperson.yaml`.

Expected VisDrone layout:

```text
VisDrone/
  images/train/*.jpg
  images/val/*.jpg
  annotations/train/*.txt
  annotations/val/*.txt
```

Expected TinyPerson layout is COCO-like:

```text
TinyPerson/
  images/train/*.jpg
  images/val/*.jpg
  annotations/train.json
  annotations/val.json
```

## Label Mapping

Primary benchmark mode is `human_unified_single`.

- VisDrone `pedestrian` maps to `person`.
- VisDrone `people` maps to `person` when `visdrone_people_policy: merge`,
  or is ignored when `visdrone_people_policy: ignore`.
- TinyPerson `person` maps to `person`.

Secondary mode is `human_split`, where VisDrone preserves `pedestrian` and
`people`, while TinyPerson remains `person`.

## Training

Baseline:

```bash
python3 scripts/train.py --config configs/experiments/visdrone_baseline.yaml
python3 scripts/train.py --config configs/experiments/tinyperson_baseline.yaml
```

CAPE hybrid:

```bash
python3 scripts/train.py --config configs/experiments/visdrone_cape.yaml
python3 scripts/train.py --config configs/experiments/tinyperson_cape.yaml
```

The default settings are single-GPU friendly: batch size 2, AMP enabled,
gradient clipping enabled, small CNN/FPN backbone, `K=128` hypotheses,
`A=64` active hypotheses, `T=3` refinement steps.

Training skips prediction decoding, merging, and final NMS by default because
losses consume raw detector outputs and CAPE hypotheses directly. Set
`model.decode_during_train: true` only when a debugging run needs train-time
decoded predictions.

## Evaluation and Reporting

```bash
python3 scripts/evaluate.py \
  --config configs/experiments/visdrone_cape.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --reports-dir outputs/reports \
  --figures-dir outputs/figures \
  --export-optional-curves
python3 scripts/make_benchmarks.py \
  --config configs/experiments/visdrone_cape.yaml \
  --predictions outputs/eval/predictions.json \
  --targets outputs/eval/targets.json \
  --reports-dir outputs/reports \
  --figures-dir outputs/figures \
  --metrics-output outputs/reports/metrics.json \
  --export-optional-curves
python3 scripts/plot_figures.py \
  --metrics outputs/eval/metrics.json \
  --output-dir outputs/figures \
  --export-optional-curves
python3 scripts/budget_sweep.py \
  --config configs/experiments/ablations/budget_sweep.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --reports-dir outputs/reports \
  --figures-dir outputs/figures \
  --export-optional-curves
python3 scripts/smoke_report_generation.py --output-dir outputs/smoke_reports
```

The reporting utilities emit markdown and CSV for exactly four benchmark tables:

1. `table1_unified_detection.csv` and `.md`
2. `table2_search_and_rescue.csv` and `.md`
3. `table3_operating_points.csv` and `.md`
4. `table4_budget_cape_ablation.csv` and `.md`

They also emit PNG and CSV for exactly three required figures:

1. `fig1_precision_recall.png` and `.csv`
2. `fig2_recall_vs_fp_per_image.png` and `.csv`
3. `fig3_confidence_threshold.png` and `.csv`

When `--export-optional-curves` is passed, the same unified evaluator also
exports:

1. `pr_by_size.csv`
2. `miss_rate_vs_fp_per_image.csv`
3. `pr_under_budget.csv`

The smoke-report command creates a synthetic two-image subset with baseline and
CAPE rows, verifies every table column required by `docs/04_EVALUATION_PROTOCOL.md`,
and checks that all table/figure artifacts exist and are non-empty.

## Known Limitations

This is a proof-of-concept implementation. The renderer uses learned primitive
footprints and differentiable smoothing rather than full RGB image formation.
The value-head target is an approximate detached refinement utility. FLOPs and
energy/image are reported when optional runtime support is available and are
otherwise emitted as `NaN` without changing table schemas.
