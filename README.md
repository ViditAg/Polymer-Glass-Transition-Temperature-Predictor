# Polymer Glass Transition Temperature Predictor

## Overview
This project predicts polymer glass transition temperature (Tg) using Chemprop (MPNN) models. It includes data cleaning, feature engineering, model training, evaluation, and batch prediction for new datasets.

## Batch Prediction Usage

### Local Environment
1. Install dependencies:
	```bash
	python -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt
	```
2. Run prediction:
	```bash
	python predict_Tg.py --model data/processed/chemprop_models/with_features_rmse --input input.csv --output predictions.csv
	```

### Docker
1. Build image:
	```bash
	docker build -t tg-predictor .
	```
2. Run prediction:
	```bash
	docker run -v $(pwd):/app tg-predictor --model /app/data/processed/chemprop_models/with_features_rmse --input /app/input.csv --output /app/predictions.csv
	```

## Input Format
- CSV file with a column (default: `smiles`) containing SMILES strings.

## Output
- CSV file with columns: `smiles_cleaned`, `logTg_pred`, `uncertainty`, and `smiles_valid`.

## Troubleshooting
- Ensure model and input files exist and paths are correct.
- For Docker, use absolute paths and mount volumes as shown above.

## License
See LICENSE file for details.
