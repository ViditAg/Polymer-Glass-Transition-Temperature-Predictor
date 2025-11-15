"""
Python script to predict Polymer Glass-transition temperature (Tg) from smiles notations
- Read in smiles notation csv file
- Clean up smiles notation
- prepare model input dataset
- load best performing Chemprop model
- Make prediction and it's uncertainity
- save an output table
"""

import argparse
import os
import sys
import logging
import traceback
import glob
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

DEFAULT_MODEL_PATH = "data/processed/chemprop_models/with_features_rmse"  # Best Chemprop ensemble directory


def setup_logging(verbosity: int) -> None:
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
    parser = argparse.ArgumentParser(
        description="Predict polymer log(Tg) and uncertainty from SMILES using a Chemprop model."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input CSV file path containing a SMILES column."
    )
    parser.add_argument(
        "-c", "--smiles-column",
        default="smiles",
        help="Name of the SMILES column in the input CSV. Default: smiles"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV file path. Default: <input_stem>_predictions.csv"
    )
    parser.add_argument(
        "-m", "--model",
        default=os.environ.get("CHEMPROP_TG_MODEL", DEFAULT_MODEL_PATH),
        help="Path to Chemprop checkpoint (.pt) or directory with ensemble checkpoints."
    )
    parser.add_argument(
        "--drop-invalid",
        action="store_true",
        help="Drop rows with invalid SMILES instead of keeping them with NaN predictions."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity: -v (info), -vv (debug)."
    )
    return parser.parse_args()


def _import_rdkit():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdMolStandardize as Std
        RDLogger.DisableLog("rdApp.*")
        return Chem, Std
    except Exception as e:
        raise RuntimeError(
            "RDKit is required for SMILES cleaning but was not found. Install rdkit-pypi."
        ) from e


def canonicalize_smiles(smiles: str, Chem, Std) -> Optional[str]:
    if smiles is None:
        return None
    s = str(smiles).strip()
    if not s:
        return None
    try:
        mol = Chem.MolFromSmiles(s, sanitize=True)
        if mol is None:
            return None

        # Largest fragment (remove salts)
        try:
            lfc = Std.LargestFragmentChooser(preferOrganic=True)
            mol = lfc.choose(mol)
        except Exception:
            pass

        # Uncharge
        try:
            uncharger = Std.Uncharger()
            mol = uncharger.uncharge(mol)
        except Exception:
            pass

        # Normalize (optional, safe)
        try:
            normalizer = Std.Normalizer()
            mol = normalizer.normalize(mol)
        except Exception:
            pass

        # Re-sanitize
        Chem.SanitizeMol(mol)
        can = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return can
    except Exception:
        return None


def clean_smiles_column(smiles: List[str]) -> Tuple[List[Optional[str]], List[bool]]:
    Chem, Std = _import_rdkit()
    cleaned: List[Optional[str]] = []
    valid_mask: List[bool] = []
    for s in smiles:
        cs = canonicalize_smiles(s, Chem, Std)
        cleaned.append(cs)
        valid_mask.append(cs is not None)
    return cleaned, valid_mask


def discover_checkpoints(model_path: str) -> List[str]:
    if not model_path:
        return []
    if os.path.isdir(model_path):
        ckpts = sorted(glob.glob(os.path.join(model_path, "*.pt")))
        return ckpts
    if os.path.isfile(model_path) and model_path.endswith(".pt"):
        return [model_path]
    return []


def predict_with_chemprop(smiles: List[str], checkpoints: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from chemprop.args import PredictArgs
        from chemprop.train import make_predictions
    except Exception as e:
        raise RuntimeError(
            "chemprop is required for prediction but was not found. Install chemprop."
        ) from e

    if not checkpoints:
        raise FileNotFoundError("No Chemprop checkpoints found. Provide --model as a .pt file or directory.")

    all_preds: List[np.ndarray] = []
    for ckpt in checkpoints:
        args = PredictArgs().parse_args([])
        # Set required fields
        args.checkpoint_path = ckpt
        # Optional performance knobs
        args.batch_size = 256
        args.num_workers = 0
        # Direct smiles input avoids writing temp files
        try:
            preds = make_predictions(args=args, smiles=smiles)
        except TypeError:
            # Older chemprop versions may require None for test_path and smiles as kw
            preds = make_predictions(args=args, smiles=smiles)
        # preds is List[List[float]] for single target
        arr = np.array(preds, dtype=float).reshape(len(smiles), -1)
        if arr.shape[1] != 1:
            # Take the first target if multiple provided
            arr = arr[:, [0]]
        all_preds.append(arr.squeeze(-1))

    preds_stack = np.stack(all_preds, axis=0)  # [n_models, n_samples]
    mean = preds_stack.mean(axis=0)
    std = preds_stack.std(axis=0) if preds_stack.shape[0] > 1 else np.full_like(mean, np.nan)
    return mean, std


def derive_output_path(input_path: str, output_arg: Optional[str]) -> str:
    if output_arg:
        return output_arg
    base, ext = os.path.splitext(os.path.basename(input_path))
    return os.path.join(os.path.dirname(input_path), f"{base}_predictions.csv")


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    try:
        if not os.path.isfile(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")

        df = pd.read_csv(args.input)
        if args.smiles_column not in df.columns:
            raise ValueError(f"SMILES column '{args.smiles_column}' not found in input CSV.")

        raw_smiles = df[args.smiles_column].astype(str).tolist()
        logging.info("Cleaning SMILES...")
        cleaned_smiles, valid_mask = clean_smiles_column(raw_smiles)

        # Build list for prediction and index mapping
        idx_map = [i for i, v in enumerate(valid_mask) if v]
        smiles_for_pred = [cleaned_smiles[i] for i in idx_map]

        checkpoints = discover_checkpoints(args.model)
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found at '{args.model}'. Provide a valid .pt file or directory."
            )

        preds_mean = np.full(len(df), np.nan, dtype=float)
        preds_std = np.full(len(df), np.nan, dtype=float)

        if len(smiles_for_pred) > 0:
            logging.info(f"Predicting with {len(checkpoints)} checkpoint(s) on {len(smiles_for_pred)} molecule(s)...")
            mean, std = predict_with_chemprop(smiles_for_pred, checkpoints)
            # Map back to original indices
            for j, i in enumerate(idx_map):
                preds_mean[i] = float(mean[j])
                preds_std[i] = float(std[j])
        else:
            logging.warning("No valid SMILES after cleaning. Predictions will be NaN.")

        out_df = df.copy()
        out_df["smiles_cleaned"] = cleaned_smiles
        out_df["logTg_pred"] = preds_mean
        out_df["uncertainty"] = preds_std
        out_df["smiles_valid"] = valid_mask

        if args.drop_invalid:
            before = len(out_df)
            out_df = out_df[out_df["smiles_valid"]].reset_index(drop=True)
            logging.info(f"Dropped {before - len(out_df)} invalid rows.")

        output_path = derive_output_path(args.input, args.output)
        out_df.to_csv(output_path, index=False)
        logging.info(f"Saved predictions to: {output_path}")
        return 0

    except Exception as e:
        logging.error(str(e))
        logging.debug("Traceback:\n%s", traceback.format_exc())
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
