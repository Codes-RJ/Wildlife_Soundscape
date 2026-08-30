# MASTER PROMPT FOR GEMINI

You are modifying an existing, mostly-complete Python + ESP32 research project.

Read ALL files in this handoff folder before editing the repository.

## Operating rules

1. Inspect the current repository before changing any file.
2. Work on a new Git branch, preferably:
   `research-upgrades`
3. Make changes phase-by-phase in the exact order in `02_IMPLEMENTATION_PLAN.md`.
4. Preserve protocol v4 and all ESP32 wire structures.
5. Do not modify firmware unless a compile/runtime error directly proves it is necessary.
6. Do not alter the existing TDOA sign convention:
   pair B-A means `(distance_B - distance_A) / c`.
7. Do not reset, compress, or reinterpret `sampleIndex`.
8. Do not treat GPIO27 as the sample clock.
9. Do not introduce 3D localization with only the current 3-node planar array.
10. Do not add field-hardware features in this cycle.
11. Do not invent BirdNET API signatures. Inspect installed package APIs or official docs if needed.
12. Do not guess biological taxonomy from a species-name string.
13. Do not silently map every BirdNET V3 class to BIRD.
14. Keep optional research dependencies optional where practical.
15. Run tests after each phase. Fix only demonstrated regressions.
16. Keep complete change notes.

## Existing known-good corrected files

This handoff contains:

- `known_good/classification/birdnet_backend.py`
- `known_good/classification/ensemble_backend.py`

These contain the already-fixed ClassificationResult contract:

`second_label is None` => `second_confidence is None`.

Use them as the current baseline if the repository still contains an older copy.

## Required completion report

After implementation, produce:

- files created;
- files modified;
- schema changes;
- dependency changes;
- tests added;
- tests passed/failed;
- benchmark commands;
- any intentionally deferred work.

Do not claim successful hardware behavior unless hardware was actually tested.
