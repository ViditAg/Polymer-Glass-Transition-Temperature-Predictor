# Polymer Glass Transition Temperature Predictor

## Overview
This project predicts polymer glass transition temperature (Tg) using an ensemble of Chemprop (MPNN) models. Further, calculate 
epistemic uncertainty and Applicability domain metrics.

## Jupyter Notebooks for the ensemble-based prediction analysis
* Data cleaning: `notebooks/00_data_cleaning.ipynb`
* Base-line prediction using SVM regressor: `notebooks/08_baseline_Tg_prediction.ipynb`
* Chem-prop ensemble-based prediction: `notebooks/03_train_model_ensemble.ipynb`

## Setup and Installation

### Local Environment
**Note:** This project uses Python 3.12.3. Please ensure you have Python 3.12.3 installed for best compatibility.

1. Clone the repository:
	```bash
	git clone <repository-url>
	cd Polymer-Glass-Transition-Temperature-Predictor
	```

2. Install dependencies:
	```bash
	python -m venv venv
	source venv/bin/activate  # On Windows: venv\Scripts\activate
	pip install -r requirements.txt
	```

## Training the Ensemble

**Important:** Before running predictions, you must first train the ensemble models locally.

1. Train the ensemble (this will train 1000 models: 10 folds × 100 models per fold):
	```bash
	python train_ensemble.py --data data/processed/training_data.csv --output-dir models/ensemble_without_descriptors
	```

	This process may take several hours depending on your hardware. The script will:
	- Load the training data
	- Create 10-fold cross-validation splits
	- Train 100 models per fold with different random seeds
	- Save all models to `models/ensemble_without_descriptors/`
	- Save train/val split indices for reproducibility

2. Optional arguments:
	- `--data`: Path to training data CSV (default: `data/processed/training_data.csv`)
	- `--output-dir`: Directory to save trained models (default: `models/ensemble_without_descriptors`)
	- `-v` or `-vv`: Increase verbosity for debugging

## Batch Prediction Usage

After training the ensemble, you can use the prediction script:

```bash
python predict_Tg.py --input input.csv --output predictions.csv
```

Or with custom SMILES column name:
```bash
python predict_Tg.py -i input.csv -c SMILES_clean -o predictions.csv
```

Sample test file available at `data/processed/test_predict_Tg_input.csv`.

### Docker
**Note:** The Docker image is based on Python 3.12 and includes all required dependencies. You do not need to install Python or packages manually if using Docker.
1. Build image:
	```bash
	docker build -t tg-predictor .
	```
2. Run prediction:
	```bash
	docker run -v $(pwd):/app tg-predictor --input /app/input.csv --output /app/predictions.csv
	```

## Input Format
- CSV file with a column (default: `SMILES`) containing SMILES strings.

## Output
- CSV file with columns: `smiles_cleaned`, `logTg_pred` and `uncertainty`.

## Troubleshooting

### Training Issues
- **Out of memory**: Reduce `NUM_WORKERS` in `train_ensemble.py` or use a machine with more RAM
- **Training takes too long**: The ensemble training (1000 models) is computationally intensive. Consider reducing `N_ENSEMBLE` or `K_FOLDS` in `train_ensemble.py` for testing, though this will affect model performance

### Prediction Issues
- **No models found**: Ensure you have trained the ensemble first using `train_ensemble.py`
- **File not found**: Verify that input file paths are correct and files exist
- **SMILES column error**: Check that your input CSV has the correct SMILES column (default: `SMILES`, use `-c` flag to specify custom column name)

### Docker
- For Docker, use absolute paths and mount volumes as shown above

## License
See LICENSE file for details.
