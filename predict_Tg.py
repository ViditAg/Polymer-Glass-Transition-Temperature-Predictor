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

from glob import glob
from chemprop.models.utils import load_model
from lightning import pytorch as pl
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import Descriptors, rdMolDescriptors, MolFromSmiles
from chemprop import data, featurizers, utils
from chemprop import uncertainty

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Predict polymer log(Tg) and uncertainty from SMILES using a Chemprop model."
    )
    # Required arguments
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Input CSV file path containing a SMILES column."
    )
    parser.add_argument(
        "-c", "--smiles-column",
        default="SMILES",
        help="Name of the SMILES column in the input CSV. Default: SMILES"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV file path. Default: <input_stem>_predictions.csv"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity: -v (info), -vv (debug)."
    )
    return parser.parse_args()



def remove_salt_and_convert_to_cannonical_smiles(smiles: str) -> str:
    """
    Removes salts/fragments from a SMILES string and returns its canonical form.
    Returns None if SMILES is invalid.
    Args:
        smiles (str): Input SMILES string.
    Returns:
        str: Canonical SMILES string or None if invalid.
    """
    # Create molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)
    # Check if the molecule was successfully created
    if mol is None:
        print(f"Could not parse SMILES: {smiles}")
        return None
    # Remove salts/fragments
    mol_no_salt = SaltRemover().StripMol(mol)
    # Convert to canonical SMILES
    canonical_smiles = Chem.MolToSmiles(
        mol_no_salt,
        canonical=True
    )
    
    return canonical_smiles


def clean_smiles_column(
        smiles: List[str]
    ) -> List[str]:
    """
    Cleans a list of SMILES strings by removing salts and converting to canonical form.
    Args:
        smiles (List[str]): List of input SMILES strings.
    Returns:
        List[Optional[str]]: List of cleaned SMILES strings.
    """
    # initialize lists
    cleaned_smiles = []
    # process each SMILES
    for smi in smiles:
        # remove salts and convert to canonical form
        cleaned_smi = remove_salt_and_convert_to_cannonical_smiles(smi)
        if cleaned_smi is None:
            raise ValueError(f"Invalid SMILES after cleaning: {smi}")

        # append results
        cleaned_smiles.append(cleaned_smi)
    
    return cleaned_smiles


def compute_descriptors(smiles):
    """Compute molecular descriptors for a given SMILES string.

    Args:
        smiles (str): The SMILES representation of the molecule.

    Returns:
        descriptors (dict): A dictionary containing the computed molecular descriptors.
            keys include:
                - 'Moleculer Weight': Molecular Weight. 
                - 'Proxy for Free Volume': Proxy for free volume.
                - 'Rotatable Bonds': Number of Rotatable Bonds.
                - 'Aromatic Rings': Number of Aromatic Rings.
                - 'Topological Polar Surface Area': Topological Polar Surface Area.
                - 'H-Bond Donors': Number of Hydrogen Bond Donors.
                - 'H-Bond Acceptors': Number of Hydrogen Bond Acceptors
    """
    # Compute molecular descriptors
    mol = MolFromSmiles(smiles)
    # Check if the molecule was successfully created
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Compute molecular descriptors
    descriptors = {
        'Moleculer Weight': Descriptors.MolWt(mol),
        'Proxy for Free Volume': rdMolDescriptors.CalcExactMolWt(mol) / rdMolDescriptors.CalcNumHeavyAtoms(mol),
        'Rotatable Bonds':  Descriptors.NumRotatableBonds(mol),
        'Aromatic Rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'Topological Polar Surface Area': Descriptors.TPSA(mol),
        'H-Bond Donors': rdMolDescriptors.CalcNumHBD(mol),
        'H-Bond Acceptors': rdMolDescriptors.CalcNumHBA(mol)
    }

    return descriptors


def find_and_load_ensemble_models(
        model_path: str = DEFAULT_MODEL_PATH
    ) -> List:
    """
    Finds and loads Chemprop models from the specified path.
    Args:
        model_path (str): Path to a Chemprop checkpoint (.pt) or directory with ensemble checkpoints.
    Returns:
        List: List of loaded Chemprop models.
    """
    # find model checkpoints
    model_ensemble_paths = glob(os.path.join(model_path, "*.pickle"))
    # raise error if no checkpoints found
    if not model_ensemble_paths:
        raise FileNotFoundError(f"No model checkpoints found in {model_path}")

    # initialize list to hold models
    model_ensemble = []

    # load each model and append to the list
    for path in model_ensemble_paths:
        model_ensemble.append(load_model(path))
    
    return model_ensemble


def ensemble_predict(
    ensemble,
    test_smis,
    test_V_fs=None,
    num_workers=20
):
    """
    Predict using ensemble of Chemprop models with uncertainty estimation
    Parameters:
        ensemble : list of trained Chemprop models
        test_smis (np.ndarray): Array of SMILES strings for testing.
        test_V_fs (np.ndarray, optional): Array of additional feature descriptors for testing.
    Returns:
        mean_pred : np.ndarray
            Mean predictions from the ensemble.
        epistemic_uncertainty : np.ndarray
            Epistemic uncertainty (model uncertainty) estimates.
        aleatoric_uncertainty : np.ndarray
            Aleatoric uncertainty (data noise) estimates.
    """
    # create list of molecule objects from SMILES strings
    test_mols = [
        utils.make_mol(smi, keep_h=False, add_h=False) for smi in test_smis
    ]
    # if additional features are to be used
    if test_V_fs is not None:
        # create MoleculeDatapoint objects with additional features
        test_data = [
            data.MoleculeDatapoint(mol, x_d=X_d) for mol, X_d in zip(test_mols, test_V_fs)
        ]
    else:
        # create MoleculeDatapoint objects without additional features
        test_data = [data.MoleculeDatapoint(mol) for mol in test_mols]
    
    # initialize featurizer and dataset
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_dset = data.MoleculeDataset(test_data, featurizer)
    
    # create data loader
    test_loader = data.build_dataloader(
        test_dset,
        batch_size=len(test_smis),
        num_workers=num_workers,
        shuffle=False
    )
    
    
    # collect predictions from each model in the ensemble
    trainer = pl.Trainer(
            accelerator='auto',
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False
    )
    # initialize uncertainty estimator
    unc_estimator_ensemble = uncertainty.EnsembleEstimator()
    # get mean predictions and uncertainties
    mean_pred, epistemic_uncertainty = unc_estimator_ensemble(
        test_loader,
        ensemble,
        trainer
    )
    
    return mean_pred, epistemic_uncertainty
    


def derive_output_path(
        input_path: str,
        output_arg: Optional[str]
    ) -> str:
    """
    Derives the output file path based on input path and optional output argument.
    Args:
        input_path (str): Path to the input file.
        output_arg (Optional[str]): User-specified output file path.
    Returns:
        str: Derived output file path.
    """
    # if output_arg is provided, use it directly
    if output_arg: return output_arg
    
    # else, derive from input path
    base, _ = os.path.splitext(os.path.basename(input_path))
    # add suffix and return
    return os.path.join(
        os.path.dirname(input_path),
        f"{base}_predictions.csv"
    )


def main() -> int:
    # Parse arguments and setup logging
    args = parse_args()
    # Setup logging based on verbosity level
    setup_logging(args.verbose)

    try:
        # check if input file exists
        if not os.path.isfile(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        # read input csv
        df = pd.read_csv(args.input)
        
        # check if smiles column exists
        if args.smiles_column not in df.columns:
            raise ValueError(
                f"SMILES column '{args.smiles_column}' not found in input CSV."
            )

        # initial cleaning to identify valid SMILES
        raw_smiles = df[args.smiles_column].astype(str).tolist()
        logging.info(f"Check {len(raw_smiles)} SMILES from input.")        
        # validate smiles if its empty or non-string
        for smi in raw_smiles:
            if not isinstance(smi, str) or smi.strip() == "":
                raise ValueError(f"Empty or non-string SMILES found: {smi}")

        logging.info("Cleaning SMILES...")
        cleaned_smiles = clean_smiles_column(raw_smiles)
        
        # add molecular descriptors
        descriptor_list = []
        for smi in cleaned_smiles:
            descriptor_list.append(compute_descriptors(smi))

        # load ensemble models
        logging.info("Loading Chemprop ensemble models...")
        ensemble = find_and_load_ensemble_models()

        # make predictions with ensemble and calculate uncertainties
        logging.info("Making predictions with ensemble and calculating uncertainties...")
        preds, epistemic_uncertainty = ensemble_predict(
            ensemble,
            cleaned_smiles,
            test_V_fs=np.array([list(desc.values()) for desc in descriptor_list]),
            num_workers=20
        )
        mean_pred = np.array(preds).mean(axis=0)
        epistemic_unc = np.array(epistemic_uncertainty).reshape(-1, 1)
        out_df = df.copy()
        out_df["smiles_cleaned"] = cleaned_smiles
        out_df["logTg_pred"] = mean_pred
        out_df["uncertainty"] = epistemic_unc

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
