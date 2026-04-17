# CAPE-Det Unified Evaluation Protocol

## Primary evaluation scope

Use one unified primary evaluation protocol across both VisDrone and TinyPerson.

### Unified label protocol

- Default main mode: `human_unified_single`
- Unified class: `person`

Mapping:
- VisDrone `pedestrian` -> `person`
- VisDrone `people` -> `person` (configurable: merge or ignore)
- TinyPerson `person` -> `person`

Secondary mode:
- `human_split`
- VisDrone keeps `pedestrian`, `people`
- TinyPerson keeps `person`

## Unified size bins

Default bins:
- `tiny`: area < 16^2
- `small`: 16^2 <= area < 32^2
- `medium_plus`: area >= 32^2

Report:
- AP_tiny
- AP_small
- AP_medium_plus
- Recall_tiny
- Recall_small

## Search-and-rescue metrics

Required:
- Pd (Probability of Detection)
- Miss Rate
- FP/image
- Recall@IoU0.3
- Recall@IoU0.5
- Latency (ms)
- FPS
- Energy/image (if feasible)
- Avg active hypotheses
- Avg refinement budget used

## Confidence-threshold analysis

Compute and export:
- Precision(conf)
- Recall(conf)
- F1(conf)
- FP/image(conf)
- MissRate(conf)
- Pd(conf)

Operating points:
- best F1 threshold
- high-recall threshold
- low-FP threshold

## Curves

Required plots:
- PR curve
- PR by size bin
- Recall vs FP/image
- Miss Rate vs FP/image
- PR under different budgets
- Precision/Recall/F1 vs confidence

## Required benchmark tables

### Table 1: Unified Detection Benchmark
- Dataset
- EvalMode
- AP50
- AP50_95
- AP75
- Precision
- Recall
- F1
- AR1
- AR10
- AR100
- AP_tiny
- AP_small
- AP_medium_plus
- Recall_tiny
- Recall_small
- Params
- FLOPs
- Latency_ms
- FPS

### Table 2: Search-and-Rescue Benchmark
- Dataset
- EvalMode
- Pd
- MissRate
- FP_per_image
- Recall_IoU_0_3
- Recall_IoU_0_5
- Latency_ms
- FPS
- Energy_per_image
- AvgActiveHypotheses
- AvgRefinementBudgetUsed

### Table 3: Operating Points by Confidence Threshold
- Dataset
- EvalMode
- Threshold_BestF1
- Precision_BestF1
- Recall_BestF1
- F1_BestF1
- Threshold_HighRecall
- Precision_HighRecall
- Recall_HighRecall
- FP_per_image_HighRecall
- Threshold_LowFP
- Precision_LowFP
- Recall_LowFP
- FP_per_image_LowFP

### Table 4: Budget / CAPE Ablation Benchmark
- Dataset
- BudgetMode
- MaxActiveHypotheses
- MaxRefinementSteps
- AvgActiveHypotheses
- AvgRefinementBudgetUsed
- AP_tiny
- Recall_tiny
- Pd
- FP_per_image
- Latency_ms
- FPS

## Required figures

### Figure 1
Precision-Recall figure:
- baseline vs CAPE
- optional size-bin curves

### Figure 2
Recall vs FP/image:
- baseline vs CAPE
- same style across both datasets

### Figure 3
Confidence-threshold figure:
- Precision vs confidence
- Recall vs confidence
- F1 vs confidence
- mark best F1, high recall, low FP points
