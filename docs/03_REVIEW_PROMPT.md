# CAPE-Det Codex Review Prompt

```text
Review the entire CAPE-Det codebase as a strict research engineering reviewer.

Check for:
- import errors
- broken dependencies
- tensor shape inconsistencies
- dataset parsing bugs
- incorrect VisDrone/TinyPerson label mapping
- GT assignment bugs for CAPE hypotheses
- non-differentiable renderer components
- accidental patch-based logic
- ablation flags that do not truly change behavior
- incorrect generation of any of the 4 benchmark tables
- incorrect generation of any of the 3 required figures
- single-GPU usability issues
- likely training instability points

Then produce:
1. a bug list
2. exact fixes
3. revised full file contents for any broken files
```
