# TODO List — 8-Week Research Sprint
*Focus: Physics-Aware Graph Neural Networks for Polymer Glass Transition Temperature Prediction*

---

## Week 1 — Setup & Baseline ✅
- [x] Create `data/` folder structure (`raw`, `processed`)
- [x] Download, clean and standardize Rasulev 902-polymer dataset (SMILES, Tg CSV)
- [x] Use author provided data to reproduce SVM baseline from author descriptors `notebooks/baseline_Tg_prediction.ipynb`
---

## Week 2 — Chemprop Baseline
- [x] Install Chemprop and prepare dataset in required format
- [x] Train baseline Chemprop MPNN model 
---

## Week 3 — Physics-Aware Modifications
- [ ] Implement physics-informed loss function in `src/physics_chemprop.py`
- [ ] Add monotonic regularization (rigidity → higher Tg)
- [ ] Integrate polymer physics descriptors (free volume, chain flexibility)
- [ ] Multi-task learning for WLF parameters (C1, C2)
- [ ] **Milestone**: Physics-aware model outperforms vanilla Chemprop

---

## Week 4 — Uncertainty Quantification
- [ ] Implement ensemble training (5+ models with different seeds)
- [ ] Add conformal prediction for calibrated confidence intervals
- [ ] Generate calibration curves and coverage analysis
- [ ] Create `src/uncertainty.py` with UQ utilities
- [ ] **Milestone**: Well-calibrated uncertainty estimates

---

## Week 5 — Applicability Domain & Model Analysis
- [ ] Implement multiple AD methods (Mahalanobis, kNN, isolation forest)
- [ ] Create domain boundary visualizations (PCA plots)
- [ ] Analyze error correlation with domain membership
- [ ] Test with synthetic challenging polymers
- [ ] **Milestone**: Robust method to identify unreliable predictions

---

## Week 6 — Comprehensive Evaluation
- [ ] Full ablation study: SVM → Chemprop → +Physics → +UQ → +AD
- [ ] Generate publication-quality figures (parity plots, calibration curves, ablation bars)
- [ ] Error analysis: where and why does the model fail?
- [ ] Create `notebooks/02_comprehensive_evaluation.ipynb`
- [ ] **Milestone**: Complete experimental validation of approach

---

## Week 7 — Manuscript Draft & Results Analysis
- [ ] Draft complete manuscript structure and introduction
- [ ] Write detailed methodology section
- [ ] Create all publication figures (`results/figures/`)
- [ ] Compile results tables with statistical significance tests
- [ ] Write results and discussion sections
- [ ] **Milestone**: Complete first manuscript draft

---

## Week 8 — Manuscript Finalization & Submission
- [ ] Revise manuscript based on internal review
- [ ] Finalize all figures with proper captions and formatting
- [ ] Complete supplementary materials and code documentation
- [ ] Add reproducibility script (`scripts/reproduce_results.sh`)
- [ ] Submit to MRS Communications Special Issue
- [ ] **Milestone**: Paper submitted for peer review

---

## Success Metrics
- **Technical**: Physics-aware model achieves RMSE < 40K (20% improvement over baseline)
- **Methodological**: Well-calibrated uncertainty with 90% coverage at 90% confidence
- **Scientific**: Clear demonstration that physics constraints improve generalization
- **Publication**: Accepted paper in MRS Communications or equivalent venue
