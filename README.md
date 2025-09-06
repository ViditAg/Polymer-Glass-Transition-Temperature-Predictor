# Polymer Tg Prediction – Physics-Informed + UQ

This project reproduces and extends the **Nature Communications Chemistry (2024) Tg prediction baseline** by:
- Adding **physics-informed constraints** (free-volume, WLF, rigidity penalties).
- Implementing **uncertainty quantification** (ensembles + conformal prediction).
- Performing an **ablation study** to demonstrate improvements.

Target: **MRS Communications Special Issue (AI and Emerging Approaches, deadline Nov 15, 2025).**

---

## Repo Layout
- `data/` → polymer datasets
- `notebooks/` → exploratory and reproducibility notebooks
- `src/` → reusable scripts (models, descriptors, physics, UQ)
- `figs/` → saved plots for manuscript
- `manuscript/` → outline + draft materials

---

## Dependencies
- Python 3.12
- RDKit
- Mordred
- scikit-learn
- PyTorch
- Optuna
- Matplotlib / Seaborn

---

## References
- Rasulev et al., *Nature Comm. Chem.*, 2024
- ChemRxiv preprint on Tg UQ (2024)
- PENN (Physics-Enforced NN for polymers)
- LieConv equivariant Tg predictor