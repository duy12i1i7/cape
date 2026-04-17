# CAPE-Det Codex Main Prompt

```text
You are a senior research engineer in computer vision, efficient deep learning, and edge AI deployment.

Your task is to build a full, runnable proof-of-concept research codebase for a novel detector idea called:

CAPE-Det = Compositional Amortized Program Executor for Tiny-Person Detection

The implementation must benchmark on BOTH:
1. VisDrone
2. TinyPerson

The code must be suitable for research prototyping, ablation, benchmarking, and paper-style reporting.

The purpose of this codebase is to evaluate whether CAPE-Det is a feasible research direction for tiny-person detection in UAV imagery, especially under search-and-rescue-like operational constraints.

====================================================
0. IMPORTANT EXECUTION INSTRUCTIONS
====================================================

Do NOT generate everything at once.

Work in two phases.

PHASE A:
First produce ONLY:
- full repository file tree
- responsibilities of each module
- tensor shape plan for the major modules
- label mapping plan for VisDrone and TinyPerson
- training plan
- unified evaluation plan
- ablation plan
- risks and implementation choices

Then STOP and wait for confirmation.

PHASE B:
After confirmation, generate actual code incrementally in this order:
1. configs + dataset layer
2. baseline detector
3. CAPE branch
4. losses + matching
5. trainer + evaluator
6. benchmark + plotting + visualization
7. tests + README

For each stage:
- print file path
- then print full file content
- ensure imports remain consistent
- do not omit code
- do not leave fake TODOs
- if a previous file must be revised, print the full revised file again

====================================================
1. HIGH-LEVEL RESEARCH GOAL
====================================================

Implement a hybrid detector where:

- A lightweight conventional global detector handles generic detection.
- A novel CAPE branch specializes in tiny human detection.

CAPE must preserve this central research idea:

Detection is performed by amortized compositional hypothesis inference under a degradation-aware image formation model,
NOT by patch routing, crop-and-redetect, tile zoom-in, focused region refinement, cluster-first processing, or heavy dense feature pyramids alone.

The project must evaluate feasibility on:
- VisDrone
- TinyPerson

====================================================
2. WHAT CAPE MUST NOT BECOME
====================================================

Do NOT implement CAPE as:
- patch selection
- tile routing
- crop-based zoom-in
- region-based stage-2 detector
- cluster proposal detector
- query over cropped regions
- prior-guided patch extraction
- any patch-first or region-first logic

The computation unit of CAPE must be:
- human hypotheses / micro-programs
NOT image patches.

====================================================
3. DATASETS
====================================================

Support both:

1. VisDrone
   - native 10 classes
   - human-relevant classes include:
     - pedestrian
     - people

2. TinyPerson
   - person-centric tiny object dataset
   - person-only native mode is allowed

The implementation must support:
- training on VisDrone only
- training on TinyPerson only
- clean extension point for future joint human-focused training

====================================================
4. REQUIRED REPOSITORY STRUCTURE
====================================================

Create a clean research codebase like:

cape_det/
  configs/
    datasets/
    experiments/
    model/
  models/
    backbones/
    necks/
    heads/
    cape/
  datasets/
  trainers/
  losses/
  metrics/
  utils/
  scripts/
  tests/
  README.md
  requirements.txt

Keep code explicit, modular, readable, and hackable.

====================================================
5. DATASET ABSTRACTION
====================================================

Design the dataset layer FIRST.

Do NOT hardcode VisDrone-specific logic into the model.

Implement a unified dataset abstraction where each sample contains:
- image tensor
- boxes in xyxy
- class labels
- metadata:
  - image id
  - original image size
  - dataset name
  - optional ignore/crowd flags if available

Implement dataset adapters for:
- VisDrone
- TinyPerson

Need:
- annotation parsing
- path config
- train/val split support
- sanity-check parser scripts
- shared transforms where possible

====================================================
6. UNIFIED HUMAN LABEL PROTOCOL
====================================================

Implement BOTH of these label modes:

(A) human_unified_single
- one unified class: person

Mapping:
- VisDrone:
  - pedestrian -> person
  - people -> person (configurable: merge or ignore depending on experiment)
- TinyPerson:
  - person -> person

(B) human_split
- preserve distinct human-related classes when supported
- VisDrone:
  - pedestrian
  - people
- TinyPerson:
  - person

Default primary benchmark mode must be:
- human_unified_single

The CAPE branch must use a unified human-target interface.
The global detector may remain dataset-specific.

Use explicit mapping tables and config-driven behavior.
No magic constants.

====================================================
7. MODEL OVERVIEW
====================================================

Implement a HYBRID model with two branches:

----------------------------------------------------
A. GLOBAL BASE DETECTOR
----------------------------------------------------

Use a lightweight conventional detector.

Requirements:
- small CNN backbone
- transparent neck (simple FPN-like or similar)
- anchor-free head preferred
- support dataset-specific class count
- outputs:
  - classification
  - box regression
  - objectness if used

This branch must be able to run by itself as the baseline.

----------------------------------------------------
B. CAPE BRANCH
----------------------------------------------------

CAPE must perform human hypothesis inference.

Core properties:
- hypothesis-centric
- compositional
- degradation-aware
- refinement budget over hypotheses
- no patch routing

====================================================
8. CAPE BRANCH REQUIREMENTS
====================================================

Implement these required modules.

----------------------------------------------------
8.1 Hypothesis Seed Generator
----------------------------------------------------

Generate a small fixed number of initial human hypotheses from lightweight feature maps.

Each hypothesis should contain compact latent parameters such as:
- x, y center (continuous)
- scale
- aspect ratio
- elongation/orientation proxy
- primitive-mixture logits
- blur/degradation latent
- confidence logit
- optional nuisance latent

Target hypothesis dimension:
- around 10 to 20 parameters

----------------------------------------------------
8.2 Compositional Primitive Vocabulary
----------------------------------------------------

Implement a small learned vocabulary of human-like primitives.

Proof-of-concept:
- 3 to 6 primitives

Do NOT use hard-coded image templates.
Implement them through learnable primitive generators:
- small MLP and/or tiny decoder

A hypothesis may be a weighted mixture over primitive bases.

----------------------------------------------------
8.3 Differentiable Footprint Renderer
----------------------------------------------------

Implement a differentiable renderer that maps a human hypothesis into a compact evidence footprint.

This renderer should approximate how a tiny person appears after degradation.

Include simple differentiable degradation effects such as:
- Gaussian blur
- smoothing / downsample-like degradation
- optional motion blur if stable

It is NOT necessary to render full RGB.
Rendering to:
- occupancy footprint map
- evidence footprint map
- compact latent evidence
is acceptable.

----------------------------------------------------
8.4 Local Evidence Encoder
----------------------------------------------------

Extract local evidence directly from feature maps.

Do NOT crop image patches and rerun a detector.

Implement efficient local evidence extraction around hypothesis centers using feature sampling/pooling.

Need:
- local feature summary
- compatibility score between rendered footprint and feature evidence
- residual / mismatch signal used for refinement

----------------------------------------------------
8.5 Iterative Hypothesis Refiner
----------------------------------------------------

Run a small fixed number of refinement steps, e.g. 2 to 4.

At each step, update hypotheses using:
- current hypothesis parameters
- local evidence summary
- renderer residual / compatibility signal
- optional global context

This must remain hypothesis refinement, not region refinement.

----------------------------------------------------
8.6 Budgeted Refinement / Value Head
----------------------------------------------------

Implement a value head predicting expected utility of refining each hypothesis.

The budget is over hypotheses only.

Allowed proof-of-concept strategies:
- refine top-K hypotheses more deeply
- weight updates by value
- early-stop refinement of low-value hypotheses

Not allowed:
- patch extraction
- tile rerouting
- region-specific heavy re-inference

Make this explicit in code comments and README.

----------------------------------------------------
8.7 Final Readout
----------------------------------------------------

Decode final hypotheses into:
- bounding boxes
- confidence scores
- class logits for human targets

Support:
- VisDrone unified human mode
- VisDrone split human mode
- TinyPerson person mode

Merge CAPE predictions with global detector outputs before final NMS / decoding.

====================================================
9. TRAINING OBJECTIVES
====================================================

Implement a modular composite loss.

----------------------------------------------------
9.1 Global Detector Loss
----------------------------------------------------

Standard detection loss for the global detector:
- classification
- bbox regression
- objectness if used

----------------------------------------------------
9.2 CAPE Hypothesis Supervision
----------------------------------------------------

Assign human GT boxes to hypotheses.

Allowed assignment:
- Hungarian
- nearest-center
- top-k soft matching
- any simple transparent matching strategy

Supervise at least:
- center
- size
- confidence
- human class

----------------------------------------------------
9.3 Rendering / Footprint Consistency Loss
----------------------------------------------------

Encourage rendered footprints to match feature-derived local evidence.

Allowed:
- L1
- BCE
- focal-like loss
- cosine similarity loss
- contrastive compatibility loss

Choose the simplest stable option.

----------------------------------------------------
9.4 Hypothesis Parsimony / Sparsity Loss
----------------------------------------------------

Penalize redundant confident hypotheses.
Encourage compact explanations.

----------------------------------------------------
9.5 Value / Budget Calibration Loss
----------------------------------------------------

Train the value head to correlate with actual usefulness of refinement.

For proof-of-concept, approximate usefulness using detached improvement in:
- matching score
- box quality
- classification confidence
- rendering compatibility

Document clearly how this target is defined.

====================================================
10. ENGINEERING REQUIREMENTS
====================================================

The project must be runnable on a single GPU.

Need:
- mixed precision training
- gradient clipping option
- checkpoint save/load
- resume support
- validation loop
- deterministic seed support where possible
- logging
- parameter count
- FLOPs estimate if feasible
- latency benchmark scripts

Benchmark latency for:
- baseline only
- baseline + CAPE

====================================================
11. UNIFIED EVALUATION PROTOCOL
====================================================

Use ONE common primary evaluation protocol across BOTH VisDrone and TinyPerson.

Do NOT use different primary metrics for different datasets.

Official dataset-native metrics may be added as optional secondary reports,
but the main benchmark must use the same protocol for both datasets.

====================================================
12. UNIFIED SIZE-BIN PROTOCOL
====================================================

Use shared size bins across both datasets.

Default bins:
- tiny: area < 16^2
- small: 16^2 <= area < 32^2
- medium_plus: area >= 32^2

Make these configurable.

Report:
- AP_tiny
- AP_small
- AP_medium_plus
- Recall_tiny
- Recall_small

====================================================
13. SEARCH-AND-RESCUE (SAR) ORIENTED EVALUATION
====================================================

Add a SAR-oriented evaluation suite.

Required SAR metrics:
- Probability of Detection (Pd)
- Miss Rate
- False Positives per Image (FP/image)
- Recall@IoU0.3
- Recall@IoU0.5
- latency
- FPS
- energy per image if feasible
- average active CAPE hypotheses
- average refinement budget used

Support these operational modes:
1. Standard unified detection mode
2. SAR-sensitive mode emphasizing recall and miss rate
3. Budget-constrained mode:
   - fixed latency budget
   - fixed max active hypotheses
   - fixed refinement budget

====================================================
14. CONFIDENCE-THRESHOLD ANALYSIS
====================================================

Add explicit confidence-threshold analysis.

Do NOT rely only on AP or PR curve.

Compute and plot:
- Precision vs confidence threshold
- Recall vs confidence threshold
- F1 vs confidence threshold
- FP/image vs confidence threshold
- Miss Rate vs confidence threshold
- Pd vs confidence threshold
- optional: active CAPE hypotheses vs confidence threshold

Requirements:
- same logic for VisDrone and TinyPerson
- baseline vs CAPE comparison on same figure if possible
- support size-bin-specific analysis
- export raw values to CSV
- mark key operating points:
  - best F1 threshold
  - high-recall threshold
  - low-FP threshold
  - optional user-specified threshold

====================================================
15. CURVE-BASED EVALUATION
====================================================

Add plotting utilities for both datasets under the unified human protocol.

Required curves:
1. Precision-Recall curve
2. PR curves by size bin:
   - tiny
   - small
   - medium_plus
3. Recall vs FP/image
4. Miss Rate vs FP/image
5. PR curves under different CAPE refinement budgets
6. optional Precision/Recall/F1 vs confidence threshold plots

Requirements:
- use the same plotting and evaluation logic for both datasets
- support baseline vs CAPE comparison on same figure
- export plots to PNG
- export raw points to CSV
- annotate key operating points

====================================================
16. THE REQUIRED 4 BENCHMARK TABLES
====================================================

Generate utilities to automatically produce the following four tables in CSV and markdown format.

----------------------------------------------------
TABLE 1: Unified Detection Benchmark
----------------------------------------------------

Columns:
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

Use the same exact schema for both VisDrone and TinyPerson.

----------------------------------------------------
TABLE 2: Search-and-Rescue Benchmark
----------------------------------------------------

Columns:
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

Use the same exact schema for both datasets.

----------------------------------------------------
TABLE 3: Operating Points by Confidence Threshold
----------------------------------------------------

Columns:
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

The code must compute these operating points automatically.

----------------------------------------------------
TABLE 4: Budget / CAPE Ablation Benchmark
----------------------------------------------------

Columns:
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

This table is mandatory because CAPE is budget-aware.

====================================================
17. THE REQUIRED 3 FIGURES
====================================================

Generate utilities to automatically produce the following three paper-style figures in PNG and CSV.

----------------------------------------------------
FIGURE 1: Precision-Recall Figure
----------------------------------------------------

Must include:
- baseline vs CAPE PR curve
- same plotting logic for both datasets
- optional per-dataset figure or shared style
- optional separate curves by size bin:
  - tiny
  - small
  - medium_plus

----------------------------------------------------
FIGURE 2: Recall vs FP_per_image Figure
----------------------------------------------------

Must include:
- baseline vs CAPE
- same axes and style for both datasets
- useful for SAR operating trade-offs

----------------------------------------------------
FIGURE 3: Confidence-Threshold Figure
----------------------------------------------------

Must include:
- Precision vs confidence
- Recall vs confidence
- F1 vs confidence

Either:
- as three aligned plots
or
- as one figure with multiple labeled curves

The figure must support:
- baseline vs CAPE comparison
- export raw values to CSV
- marking operating points:
  - best F1
  - high recall
  - low FP

====================================================
18. TRAINING MODES
====================================================

Implement:
1. VisDrone baseline training
2. VisDrone CAPE training
3. TinyPerson baseline training
4. TinyPerson CAPE training

Optional extension:
5. joint human-focused training on both datasets

If joint training is too much for the first stable version:
- implement single-dataset modes first
- add a clean extension point

====================================================
19. CONFIG SYSTEM
====================================================

Create configs such as:

configs/datasets/visdrone.yaml
configs/datasets/tinyperson.yaml

configs/experiments/visdrone_baseline.yaml
configs/experiments/visdrone_cape.yaml
configs/experiments/tinyperson_baseline.yaml
configs/experiments/tinyperson_cape.yaml
configs/experiments/joint_humans_visdrone_tinyperson.yaml

Each config must expose:
- dataset name
- paths
- label mappings
- image size
- augmentation settings
- batch size
- optimizer settings
- evaluation mode
- size bin thresholds
- confidence threshold settings
- CAPE hyperparameters
- ablation toggles

====================================================
20. ABLATION SUPPORT
====================================================

Make these ablations easy to run:

1. global baseline only
2. baseline + hypothesis seeds only
3. baseline + seeds + renderer but no iterative refinement
4. baseline + full CAPE without value head
5. baseline + full CAPE
6. baseline + full CAPE + sparsity loss
7. baseline + full CAPE + value calibration
8. budget sweep:
   - varying active hypotheses
   - varying refinement steps
9. runs on VisDrone
10. runs on TinyPerson

Ablation toggles must truly change execution behavior.

====================================================
21. BENCHMARK / REPORTING UTILITIES
====================================================

Add scripts to:
- run evaluation
- dump all four tables
- dump all three figures
- export raw curves to CSV
- compare baseline vs CAPE
- compare multiple CAPE budget settings
- print summary metrics to console

====================================================
22. VISUALIZATION UTILITIES
====================================================

Add scripts to visualize:
- predicted boxes
- CAPE hypotheses
- rendered evidence footprints
- refinement trajectories
- baseline vs CAPE overlays

Support dataset selection:
- visdrone
- tinyperson

====================================================
23. TESTS
====================================================

Add real minimal tests for:
- dataset parsing
- label mapping
- forward pass
- loss computation
- dummy train step
- prediction decoding
- hypothesis rendering
- CAPE refinement loop
- benchmark table generation
- curve export

====================================================
24. README REQUIREMENTS
====================================================

README must explain:
- project overview
- CAPE intuition
- repository structure
- environment setup
- dataset preparation for VisDrone
- dataset preparation for TinyPerson
- training commands
- evaluation commands
- benchmark generation commands
- figure generation commands
- label mapping logic
- unified evaluation protocol
- SAR evaluation protocol
- confidence-threshold analysis
- known limitations
- debugging tips
- likely failure modes

====================================================
25. CODING STYLE REQUIREMENTS
====================================================

- Keep code explicit and readable
- Avoid giant monolithic files
- Prefer small reusable modules
- Use type hints where practical
- Add docstrings for nontrivial classes/functions
- Explain tensor shapes in comments for major modules
- Keep imports consistent
- Surface hyperparameters in configs
- Choose the simplest stable design preserving the CAPE idea

Do not output pseudocode only.
Do not stop at scaffolding.
When Phase B is requested, write runnable code.

====================================================
26. SELF-REVIEW STEP
====================================================

After generating the codebase, perform a strict self-review and report:
1. import/dependency issues
2. tensor shape risks
3. GT assignment risks
4. dataset-specific leakage into the model
5. accidental patch-based logic
6. fake ablation toggles
7. likely training instability
8. benchmark/report generation issues
9. exact fixes

If any file is broken, provide the full corrected content.

====================================================
27. FINAL EXPECTATION
====================================================

I want a runnable proof-of-concept codebase that can answer:

Is CAPE-Det a feasible research direction for tiny-person detection on VisDrone and TinyPerson under unified SAR-oriented evaluation?

Prioritize:
1. conceptual faithfulness
2. runnable implementation
3. clean experimentation
4. correct generation of the 4 tables and 3 figures
5. clarity over unnecessary complexity

Start now with PHASE A only:
- full repository file tree
- module responsibilities
- tensor shape plan
- dataset mapping plan
- training/evaluation plan
- the exact design for the 4 tables and 3 figures
- risks and implementation choices

Then stop and wait.
```
