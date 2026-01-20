"""
Python script to train an ensemble of Chemprop models for polymer Tg prediction.

This script trains an ensemble of 100 MPNN models (10 folds × 10 models per fold)
without using molecular descriptors. The trained models are saved and can be used
for prediction with predict_Tg.py.
"""

import argparse
import os
import sys
import logging
import traceback
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import KFold
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from chemprop import data, featurizers, models, nn, utils
from chemprop.nn.metrics import RMSE

# Reduce verbosity
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


# Configuration constants
SMILES_COLUMN = 'SMILES_clean'
TARGET_COLUMN = ['logTg']
N_ENSEMBLE = 100  # Number of models per fold
DATA_SPLIT_SEED = 42  # Fixed seed for train/val split
K_FOLDS = 10  # Number of folds for cross-validation
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 20
BATCH_SIZE = 64
NUM_WORKERS = 0
SEED_GENERATOR_SEED = 12345  # Seed for generating training seeds


def setup_logging(verbosity: int) -> None:
    """Setup logging configuration."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an ensemble of Chemprop models for polymer Tg prediction."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/training_data.csv",
        help="Path to training data CSV file. Default: data/processed/training_data.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/ensemble_without_descriptors",
        help="Directory to save trained models. Default: models/ensemble_without_descriptors"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity: -v (info), -vv (debug)."
    )
    return parser.parse_args()


def train_single_model_with_early_stopping(
    fold,
    train_idx,
    val_idx,
    all_mols,
    all_ys,
    configs,
    featurizer,
    logs_dir,
    train_seed,
):
    """
    Train a single MPNN model with early stopping.
    
    Args:
        fold: Fold number (0-indexed)
        train_idx: Training indices
        val_idx: Validation indices
        all_mols: List of all molecule objects
        all_ys: Array of all target values
        configs: Configuration dictionary
        featurizer: Molecule featurizer
        logs_dir: Directory for logging and checkpoints
        train_seed: Random seed for this training run
        
    Returns:
        Dictionary with training results including checkpoint path
    """
    # Set seed for reproducibility
    pl.seed_everything(train_seed, workers=True, verbose=False)

    # Prepare data without descriptors
    train_data = [
        data.MoleculeDatapoint(
            all_mols[train_idx[i]],
            all_ys[train_idx[i]]
        ) for i in range(len(train_idx))
    ]
    val_data = [
        data.MoleculeDatapoint(
            all_mols[val_idx[i]],
            all_ys[val_idx[i]],
        ) for i in range(len(val_idx))
    ]

    train_dset = data.MoleculeDataset(train_data, featurizer)
    val_dset = data.MoleculeDataset(val_data, featurizer)

    # Normalize targets
    fold_scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(fold_scaler)

    # Create data loaders
    train_loader = data.build_dataloader(
        train_dset,
        batch_size=configs["batch_size"],
        num_workers=configs["num_workers"],
        shuffle=True
    )
    val_loader = data.build_dataloader(
        val_dset,
        batch_size=configs["batch_size"],
        num_workers=configs["num_workers"],
        shuffle=False
    )

    # Build model
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(fold_scaler)
    ffn_input_dim = mp.output_dim  # No descriptors
    X_d_transform = None

    ffn = nn.RegressionFFN(
        input_dim=ffn_input_dim,
        output_transform=output_transform,
        criterion=configs["loss_criterion"],
    )
    mpnn = models.MPNN(
        mp,
        agg,
        ffn,
        batch_norm=configs["batch_norm"],
        X_d_transform=X_d_transform
    )
    
    # Setup logging and checkpointing
    model_log_dir = Path(logs_dir) / f"fold_{fold+1}" / f"seed_{train_seed}"
    model_log_dir.mkdir(parents=True, exist_ok=True)
    
    csv_logger = pl.loggers.CSVLogger(
        save_dir=model_log_dir,
        name=f"fold_{fold+1}_seed_{train_seed}"
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=model_log_dir,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=configs['early_stop_patience'],
        mode="min",
        verbose=False,
    )

    trainer = pl.Trainer(
        logger=csv_logger,
        callbacks=[early_stop_cb, checkpoint_cb],
        enable_checkpointing=True,
        enable_progress_bar=False,
        accelerator="auto",
        devices=1,
        max_epochs=configs["max_epochs"],
        deterministic=configs["deterministic"],
        enable_model_summary=False,
        log_every_n_steps=999999
    )

    trainer.fit(
        mpnn,
        train_loader,
        val_loader
    )

    # Get checkpoint path
    checkpoint_path = checkpoint_cb.best_model_path

    return {
        "fold": fold+1,
        "train_seed": train_seed,
        "checkpoint_path": str(checkpoint_path),
        "fold_scaling_factor": fold_scaler.scale_[0],
    }


def main() -> int:
    """Main training function."""
    args = parse_args()
    setup_logging(args.verbose)

    try:
        # Validate inputs
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"Training data file not found: {data_path}")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load training data
        logging.info(f"Loading training data from {data_path}...")
        df_train = pd.read_csv(data_path)
        
        if SMILES_COLUMN not in df_train.columns:
            raise ValueError(f"SMILES column '{SMILES_COLUMN}' not found in training data")
        if TARGET_COLUMN[0] not in df_train.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN[0]}' not found in training data")
        
        all_smis = df_train.loc[:, SMILES_COLUMN].values
        all_ys = df_train.loc[:, TARGET_COLUMN].values
        logging.info(f"Loaded {len(all_smis)} training samples")
        
        # Create RDKit molecule objects
        logging.info("Creating molecule objects from SMILES...")
        all_mols = [
            utils.make_mol(smi, keep_h=False, add_h=False) for smi in all_smis
        ]
        
        # Create K-Fold splits
        logging.info(f"Creating {K_FOLDS}-fold cross-validation splits...")
        kf = KFold(
            n_splits=K_FOLDS,
            shuffle=True,
            random_state=DATA_SPLIT_SEED
        )
        
        train_idx_dict = {}
        val_idx_dict = {}
        for fold, (train_idx, val_idx) in enumerate(kf.split(all_mols)):
            train_idx_dict[fold] = train_idx
            val_idx_dict[fold] = val_idx
            logging.info(f"Fold {fold+1}: {len(train_idx)} train, {len(val_idx)} val samples")
        
        # Save split indices for reproducibility
        models_dir = output_dir.parent
        models_dir.mkdir(exist_ok=True)
        with open(models_dir / "train_data_split_indexes.pkl", "wb") as f:
            pickle.dump(train_idx_dict, f)
        with open(models_dir / "validation_data_split_indexes.pkl", "wb") as f:
            pickle.dump(val_idx_dict, f)
        logging.info("Saved train/val split indices")
        
        # Generate random seeds for ensemble members
        train_seeds = np.random.default_rng(SEED_GENERATOR_SEED).integers(
            low=0, high=2**32, size=N_ENSEMBLE
        )
        
        # Training configuration
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        configs = {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "max_epochs": MAX_EPOCHS,
            "loss_criterion": RMSE(),
            "batch_norm": True,
            "deterministic": True,
            "early_stop_patience": EARLY_STOP_PATIENCE,
        }
        
        # Train ensemble
        logging.info(f"Training ensemble of {N_ENSEMBLE} models across {K_FOLDS} folds...")
        logging.info(f"Total models to train: {N_ENSEMBLE * K_FOLDS}")
        logging.info(f"Models will be saved to: {output_dir}")
        
        ensemble_results = []
        
        for fold_ in range(K_FOLDS):
            logging.info(f"Starting training for fold {fold_+1}/{K_FOLDS}")
            for i in tqdm(range(N_ENSEMBLE), desc=f"Fold {fold_+1}"):
                train_seed = int(train_seeds[i])
                result = train_single_model_with_early_stopping(
                    fold=fold_,
                    train_idx=train_idx_dict[fold_],
                    val_idx=val_idx_dict[fold_],
                    all_mols=all_mols,
                    all_ys=all_ys,
                    configs=configs,
                    featurizer=featurizer,
                    logs_dir=output_dir,
                    train_seed=train_seed,
                )
                ensemble_results.append(result)
        
        logging.info(f"✓ Successfully trained {len(ensemble_results)} models")
        logging.info(f"Models saved to: {output_dir}")
        logging.info("Training complete! You can now use predict_Tg.py for predictions.")
        
        return 0
        
    except Exception as e:
        logging.error(str(e))
        logging.debug("Traceback:\n%s", traceback.format_exc())
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
