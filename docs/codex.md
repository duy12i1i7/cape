“Using docs as source of truth, fix CAPE supervision only. Preserve hypothesis-centric design. Update CAPE internals to store seed and final readouts, normalize box/center/size losses, reverse footprint-loss detach so learned primitives train toward feature evidence, and make value calibration target detached refinement improvement. Add focused tests for these losses.”

“Fix dataset/evaluation unification. Preserve ignored VisDrone regions as ignore boxes, ensure TinyPerson/VisDrone label mapping remains config-driven, add validation that mapper num_classes equals model num_classes, and add tests for human_unified_single, VisDrone human_split, TinyPerson human_split, and ignored-region evaluation.”

“Finish benchmark protocol support from docs/04_EVALUATION_PROTOCOL.md. Update scripts/evaluate.py to emit all 4 tables and 3 figures, add a real budget sweep script that populates Table 4 across CAPE budget settings, and add optional CSV exports for PR-by-size, MissRate-vs-FP, and PR-under-budget curves.”

“Simplify training for a stable proof of concept. Avoid NMS/decoding during train unless requested, reduce default active hypotheses for smoke configs, keep value calibration disabled by default, and add ablation tests proving each toggle changes behavior.”

“Run the strict review prompt in docs/03_REVIEW_PROMPT.md after the above fixes. Report import/dependency issues, tensor shape risks, GT assignment risks, accidental patch-based logic, fake ablation toggles, benchmark generation issues, and exact fixes.”