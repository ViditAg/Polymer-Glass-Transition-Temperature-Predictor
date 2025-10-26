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
- [x] Integrate polymer physics descriptors (free volume, chain flexibility)
- [x] Train Chemprop MPNN model with features 

---

## Week 4 — Uncertainty Quantification
- [ ] Apply Uncertainity quantification via Chemprop
- [ ] Hyperparameter tuning across all Chemprop models (baseline, +physics, +UQ) using grid/random search
- [ ] Apply optimal hyperparameters to all model variants for fair comparison
- [ ] Implement multiple AD methods (Mahalanobis and kNN)
- [ ] Create domain boundary visualizations (PCA plots)
- [ ] Analyze error correlation with domain membership
- [ ] Full ablation study: SVM → Chemprop → +Physics → +UQ → +AD
---

## Week 5 — Comprehensive Evaluation
- [ ] Generate publication-quality figures (parity plots, calibration curves, ablation bars)
- [ ] Draft complete manuscript structure and introduction
- [ ] Write detailed methodology section
- [ ] Create all publication figures (`results/figures/`)
- [ ] Compile results tables with statistical significance tests
- [ ] Write results and discussion sections
- [ ] **Milestone**: Complete first manuscript draft

---

## Success Metrics
- **Technical**: Physics-aware model achieves RMSE < 40K (20% improvement over baseline)
- **Methodological**: Well-calibrated uncertainty with 90% coverage at 90% confidence
- **Scientific**: Clear demonstration that physics constraints improve generalization
- **Publication**: Accepted paper in MRS Communications or equivalent venue
