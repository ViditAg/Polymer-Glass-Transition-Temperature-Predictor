# Physics-Aware Polymer Tg Prediction (P-GRP)

This repository implements **Polymer-Graph Representations** with 
physics-aware inductive biases and uncertainty quantification for 
glass transition temperature (**Tg**) prediction.

## Features
- Polymer-specific graph representation
- Physics-informed GNN (auxiliary free-volume & monotonicity constraints)
- Ensemble + Conformal prediction for calibrated UQ
- Applicability Domain (AD) analysis
- Streamlit app for interactive Tg prediction

## Getting Started
```bash
git clone https://github.com/yourname/polymer-tg-prediction
cd polymer-tg-prediction
pip install -r requirements.txt
```

## Run Models
```bash
python src/train.py --model gnn --with_physics
python src/evaluate.py --model gnn --with_uncertainty
```

## Streamlit App
```bash
streamlit run src/streamlit_app.py
```

## Citation

TBD — after MRS submission.