# Quickstart

## Suggested workflow with Codex

1. Paste `01_MAIN_PROMPT.md` into Codex.
2. Wait for PHASE A output.
3. Review the file tree, dataset mapping, tensor shapes, and the 4 tables + 3 figures plan.
4. Paste `02_PHASE_B_PROMPT.md`.
5. Let Codex generate the code stage by stage.
6. Paste `03_REVIEW_PROMPT.md`.
7. Apply any fixes returned by the review.

## What this bundle contains

- Main Codex prompt for CAPE-Det research implementation
- Phase B follow-up prompt for code generation
- Review prompt for debugging and cleanup
- Unified evaluation protocol summary
- This quickstart guide

## Notes

- The main benchmark is unified across VisDrone and TinyPerson.
- Search-and-rescue evaluation is built into the prompt.
- The prompt requires exactly 4 benchmark tables and 3 required figures.
- CAPE must remain hypothesis-centric and must not devolve into patch routing.
