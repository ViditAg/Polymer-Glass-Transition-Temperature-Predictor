# Week-by-Week Tasks

## Week 1 – Setup & Data
- [ ] Download Rasulev 2024 Tg dataset (~902 polymers).
- [ ] Explore dataset distribution (Tg histogram).
- [ ] Implement descriptor generation with RDKit/Mordred.
- [ ] Document dataset cleaning steps in `data/README.md`.

## Week 2 – Baseline Reproduction
- [ ] Train SVM with 15 selected descriptors.
- [ ] Validate R² ≈ 0.77 vs reported values.
- [ ] Save parity plot in `figs/`.

## Week 3 – Physics Constraints Prototype
- [ ] Define free-volume penalty function.
- [ ] Define WLF consistency penalty.
- [ ] Define backbone rigidity penalty.
- [ ] Add penalties to a simple regression model.

## Week 4 – Physics + UQ Integration
- [ ] Train ensembles with physics penalties.
- [ ] Implement deep ensembles (variance).
- [ ] Add conformal prediction wrapper.
- [ ] Implement physics-guided UQ (uncertainty boost for violations).

## Week 5 – Validation & Ablation
- [ ] Run ablation: baseline vs UQ vs physics vs combined.
- [ ] Save calibration plots + results table.

## Week 6 – Repo & Writing Prep
- [ ] Clean notebooks → scripts in `src/`.
- [ ] Finalize hyperparameters.
- [ ] Create manuscript outline.

## Week 7 – Drafting
- [ ] Write Methods, Results, and Discussion sections.
- [ ] Insert figures (≤4 total).
- [ ] Peer review draft.

## Week 8 – Submission
- [ ] Final proofread.
- [ ] Submit to MRS portal.
- [ ] Upload repo to GitHub.