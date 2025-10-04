"""
Data preprocessing utilities for polymer property prediction.
Handles SMILES standardization and descriptor calculation.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.SaltRemover import SaltRemover
from mordred import Calculator, descriptors
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SMILESProcessor:
    """Handles SMILES string standardization and cleaning."""
    
    def __init__(self):
        self.salt_remover = SaltRemover()
    
    def canonicalize_smiles(self, smiles):
        """
        Convert SMILES to canonical form using RDKit.
        
        Example:
        Input: "CC(C)c1ccccc1" or "C(C)(C)c1ccccc1"
        Output: "CC(C)c1ccccc1" (canonical form)
        """
        try:
            # Parse SMILES string into molecule object
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Could not parse SMILES: {smiles}")
                return None
            
            # Convert back to canonical SMILES
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            return canonical_smiles
        
        except Exception as e:
            logger.error(f"Error canonicalizing SMILES {smiles}: {e}")
            return None
    
    def remove_salts(self, smiles):
        """
        Remove salts and counterions from SMILES.
        
        Example:
        Input: "CCO.Cl" (ethanol with chloride salt)
        Output: "CCO" (just ethanol)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Remove salts
            mol_no_salt = self.salt_remover.StripMol(mol)
            return Chem.MolToSmiles(mol_no_salt, canonical=True)
        
        except Exception as e:
            logger.error(f"Error removing salts from {smiles}: {e}")
            return None
    
    def standardize_smiles(self, smiles):
        """
        Full SMILES standardization pipeline.
        
        Steps:
        1. Remove salts
        2. Canonicalize
        3. Handle stereochemistry (optional)
        """
        if pd.isna(smiles) or smiles == "":
            return None
        
        # Step 1: Remove salts
        clean_smiles = self.remove_salts(smiles)
        if clean_smiles is None:
            return None
        
        # Step 2: Canonicalize
        canonical_smiles = self.canonicalize_smiles(clean_smiles)
        
        return canonical_smiles


class DescriptorCalculator:
    """Calculates molecular descriptors for polymers."""
    
    def __init__(self):
        # Initialize Mordred calculator with all 2D descriptors
        self.mordred_calc = Calculator(descriptors, ignore_3D=True)
    
    def calculate_rdkit_descriptors(self, smiles):
        """
        Calculate basic RDKit descriptors.
        
        Returns common descriptors like:
        - Molecular weight
        - LogP (lipophilicity)
        - Number of rings
        - Hydrogen bond donors/acceptors
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        desc_dict = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'HBD': Lipinski.NumHDonors(mol),  # Hydrogen bond donors
            'HBA': Lipinski.NumHAcceptors(mol),  # Hydrogen bond acceptors
            'TPSA': Descriptors.TPSA(mol),  # Topological polar surface area
            'NumRings': rdMolDescriptors.CalcNumRings(mol),
            'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'FractionCsp3': rdMolDescriptors.CalcFractionCsp3(mol),
        }
        
        return desc_dict
    
    def calculate_mordred_descriptors(self, smiles):
        """
        Calculate comprehensive 2D descriptors using Mordred.
        
        Mordred provides 1800+ descriptors including:
        - Constitutional descriptors
        - Topological descriptors
        - Molecular graph descriptors
        - Pharmacophore descriptors
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        # Calculate all descriptors
        desc_values = self.mordred_calc(mol)
        
        # Convert to dictionary, handling missing values
        desc_dict = {}
        for desc, value in zip(self.mordred_calc.descriptors, desc_values):
            desc_name = str(desc)
            # Handle missing/invalid values
            if pd.isna(value) or np.isinf(value):
                desc_dict[desc_name] = 0.0
            else:
                desc_dict[desc_name] = float(value)
        
        return desc_dict
    
    def calculate_polymer_specific_descriptors(self, smiles):
        """
        Calculate polymer-specific descriptors.
        
        These are custom descriptors relevant for polymers:
        - Repeat unit molecular weight
        - Flexibility indicators
        - Glass transition related features
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        # Example polymer-specific descriptors
        desc_dict = {
            'RepeatUnitMW': Descriptors.MolWt(mol),  # For repeat unit
            'FlexibilityIndex': rdMolDescriptors.CalcNumRotatableBonds(mol) / max(1, mol.GetNumAtoms()),
            'AromaticFraction': rdMolDescriptors.CalcNumAromaticRings(mol) / max(1, rdMolDescriptors.CalcNumRings(mol)),
            'Sp3Fraction': rdMolDescriptors.CalcFractionCsp3(mol),
        }
        
        return desc_dict


def process_polymer_dataset(df, smiles_column='SMILES', target_column='Tg'):
    """
    Complete preprocessing pipeline for polymer dataset.
    
    Args:
        df: DataFrame with polymer data
        smiles_column: name of SMILES column
        target_column: name of target property column (e.g., 'Tg')
    
    Returns:
        processed_df: DataFrame with standardized SMILES and descriptors
    """
    logger.info("Starting polymer dataset preprocessing...")
    
    # Initialize processors
    smiles_processor = SMILESProcessor()
    desc_calculator = DescriptorCalculator()
    
    # Step 1: Standardize SMILES
    logger.info("Standardizing SMILES...")
    df['SMILES_standardized'] = df[smiles_column].apply(smiles_processor.standardize_smiles)
    
    # Remove invalid SMILES
    valid_mask = df['SMILES_standardized'].notna()
    df_clean = df[valid_mask].copy()
    logger.info(f"Removed {(~valid_mask).sum()} invalid SMILES. {len(df_clean)} molecules remaining.")
    
    # Step 2: Calculate descriptors
    logger.info("Calculating RDKit descriptors...")
    rdkit_descriptors = []
    for smiles in df_clean['SMILES_standardized']:
        desc = desc_calculator.calculate_rdkit_descriptors(smiles)
        rdkit_descriptors.append(desc)
    
    rdkit_df = pd.DataFrame(rdkit_descriptors)
    
    logger.info("Calculating Mordred descriptors...")
    mordred_descriptors = []
    for smiles in df_clean['SMILES_standardized']:
        desc = desc_calculator.calculate_mordred_descriptors(smiles)
        mordred_descriptors.append(desc)
    
    mordred_df = pd.DataFrame(mordred_descriptors)
    
    logger.info("Calculating polymer-specific descriptors...")
    polymer_descriptors = []
    for smiles in df_clean['SMILES_standardized']:
        desc = desc_calculator.calculate_polymer_specific_descriptors(smiles)
        polymer_descriptors.append(desc)
    
    polymer_df = pd.DataFrame(polymer_descriptors)
    
    # Step 3: Combine all data
    result_df = pd.concat([
        df_clean.reset_index(drop=True),
        rdkit_df,
        mordred_df,
        polymer_df
    ], axis=1)
    
    logger.info(f"Preprocessing complete. Final dataset shape: {result_df.shape}")
    return result_df


# Example usage
if __name__ == "__main__":
    # Example dataset
    sample_data = {
        'SMILES': [
            'CC(C)(C)c1ccc(cc1)C(C)(C)C',  # Polystyrene-like
            'CCCCCCCC',  # Polyethylene-like  
            'CC(C)CC',  # Simple polymer repeat unit
            'CCO.Cl',  # With salt (will be cleaned)
        ],
        'Tg': [373, 200, 250, 300]  # Glass transition temperatures
    }
    
    df = pd.DataFrame(sample_data)
    
    # Process the dataset
    processed_df = process_polymer_dataset(df)
    
    print("Original data:")
    print(df)
    print("\nProcessed data (first few columns):")
    print(processed_df[['SMILES', 'SMILES_standardized', 'Tg', 'MW', 'LogP', 'NumRings']].head())
