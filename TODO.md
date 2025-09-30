# TODO List — Week by Week

---

## Week 1 — Setup & Baseline
- [ ] Create `data/` folder structure (`raw`, `processed`)
- [ ] Download and clean Rasulev 902-polymer dataset
- [ ] Write `data_prep.py` to standardize SMILES and preprocess descriptors
- [ ] Reproduce SVM baseline in `notebooks/exploratory.ipynb`
- [ ] Save baseline metrics (RMSE, R²) to `results/tables/`
- [ ] Set up repo: `requirements.txt`, `Dockerfile`, pre-commit linting
- [ ] Add unit tests for data preprocessing (`tests/test_data_prep.py`)

---

## Week 2 — P-GRP Representation
- [ ] Implement SMILES → P-GRP converter in `representation.py`
- [ ] Encode backbone vs side-chain, ring counts, sp2/sp3 flags
- [ ] Add support for tacticity markers (isotactic, syndiotactic, atactic)
- [ ] Write `tests/test_representation.py` to check graph construction
- [ ] Train first GNN on P-GRP features (no physics bias) in `train.py`
- [ ] Compare performance to descriptor-only MLP baseline
- [ ] Document representation choices in `README.md`

---

## Week 3 — Physics-Aware Inductive Bias
- [ ] Add physics head (free-volume proxy or rigidity score) in `models.py`
- [ ] Implement monotonic regularization loss in `train.py`
- [ ] Add auxiliary task for WLF parameter consistency
- [ ] Train GNN baseline vs physics-biased version
- [ ] Save comparison results (with/without physics) to `results/tables/`
- [ ] Add parity plots (predicted vs true Tg) to `results/figures/`
- [ ] Update `notebooks/ablation.ipynb` with results

---

## Week 4 — Uncertainty Quantification
- [ ] Implement ensemble-based UQ in `uncertainty.py`
- [ ] Train ensemble of 5 GNN models with different seeds
- [ ] Add conformal prediction calibration (split/CV+)
- [ ] Plot calibration curves in `evaluate.py`
- [ ] Save coverage tables (50%, 80%, 95%) to `results/tables/`
- [ ] Document UQ implementation in `README.md`

---

## Week 5 — Applicability Domain & External Validation
- [ ] Implement AD analysis (Mahalanobis distance, kNN) in `evaluate.py`
- [ ] Add visualization: scatter plot of in-domain vs out-of-domain points
- [ ] Validate models on external SciFinder dataset
- [ ] Compare coverage & error for in-domain vs out-of-domain
- [ ] Save AD plots + external set metrics to `results/figures/`
- [ ] Write `notebooks/external_validation.ipynb` to summarize results

---

## Week 6 — Ablation Study & Decision Demo
- [ ] Run full ablation: SVM, MLP, GNN, GNN+physics, GNN+UQ, GNN+physics+UQ
- [ ] Generate ablation bar charts (R², RMSE, calibration metrics)
- [ ] Implement decision demo: top-k virtual screening experiment
- [ ] Plot cost vs number of true hits (decision-centric figure)
- [ ] Document ablation results in `results/figures/ablation.png`
- [ ] Update `notebooks/ablation.ipynb` with decision analysis

---

## Week 7 — Streamlit App
- [ ] Build app in `streamlit_app.py`
    - Input: SMILES or repeat unit
    - Output: predicted Tg, UQ interval, AD flag, physics consistency score
- [ ] Add parity + interval plot inside app
- [ ] Add AD scatter plot panel
- [ ] Package app with Dockerfile for easy deployment
- [ ] Write app instructions in `README.md`
- [ ] Record short demo (gif/screencast) and save in `results/`

---

## Week 8 — Manuscript & Submission
- [ ] Prepare final figures (pipeline schematic, parity, calibration, ablation, decision demo) in `results/figures/`
- [ ] Compile final tables (metrics, coverage, ablation) in `results/tables/`
- [ ] Draft manuscript text (`manuscript/mrs_submission.md`)
- [ ] Cross-check citations and references
- [ ] Finalize README and LICENSE
- [ ] Add reproducibility script (`scripts/run_all.sh`)
- [ ] Submit to MRS Communications Special Issue
