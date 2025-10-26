

# ------- Extracting physical property descriptors for molecules------------

from rdkit.Chem import Descriptors, rdMolDescriptors, MolFromSmiles




def compute_descriptors(smiles):
    """Compute molecular descriptors for a given SMILES string.

    Args:
        smiles (str): The SMILES representation of the molecule.

    Returns:
        descriptors (dict): A dictionary containing the computed molecular descriptors.
            keys include:
                - 'MW': Molecular Weight. 
                - 'FreeVolumeProxy': Proxy for free volume.
                - 'RotatableBonds': Number of Rotatable Bonds.
                - 'AromaticRings': Number of Aromatic Rings.
                - 'TPSA': Topological Polar Surface Area.
                - 'HBondDonors': Number of Hydrogen Bond Donors.
                - 'HBondAcceptors': Number of Hydrogen Bond Acceptors
    """
    # Compute molecular descriptors
    mol = MolFromSmiles(smiles)
    # Check if the molecule was successfully created
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Compute molecular descriptors
    descriptors = {
        'MW': Descriptors.MolWt(mol),
        'FreeVolumeProxy': rdMolDescriptors.CalcExactMolWt(mol) / rdMolDescriptors.CalcNumHeavyAtoms(mol),
        'RotatableBonds':  Descriptors.NumRotatableBonds(mol),
        'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBondDonors': rdMolDescriptors.CalcNumHBD(mol),
        'HBondAcceptors': rdMolDescriptors.CalcNumHBA(mol)
    }

    return descriptors


def train_model_with_kfold(config, full_dset, num_workers, scaler, k_folds=5):
    """
    Train model with k-fold cross-validation
    """
    # Extract hyperparameters from config
    depth = int(config["depth"])
    ffn_hidden_dim = int(config["ffn_hidden_dim"])
    ffn_num_layers = int(config["ffn_num_layers"])
    message_hidden_dim = int(config["message_hidden_dim"])
    learning_rate = config["learning_rate"]
    batch_size = int(config["batch_size"])
    max_epochs = int(config["max_epochs"])
    
    # Get data indices for k-fold
    n_samples = len(full_dset)
    indices = np.arange(n_samples)
    
    # Initialize k-fold cross-validator
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_val_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"Training fold {fold + 1}/{k_folds}")
        
        # Create train/val subsets for this fold
        train_subset = [full_dset[i] for i in train_idx]
        val_subset = [full_dset[i] for i in val_idx]
        
        # Create datasets
        train_dset_fold = data.MoleculeDataset(train_subset, full_dset.featurizer)
        val_dset_fold = data.MoleculeDataset(val_subset, full_dset.featurizer)
        
        # Apply same scaling as full dataset
        train_dset_fold.Y = scaler.transform(train_dset_fold.Y)
        val_dset_fold.Y = scaler.transform(val_dset_fold.Y)
        
        # Copy scaling for X_d if it exists
        if hasattr(full_dset, 'X_d_scaler') and full_dset.X_d_scaler is not None:
            if train_dset_fold.X_d is not None:
                train_dset_fold.X_d = full_dset.X_d_scaler.transform(train_dset_fold.X_d)
            if val_dset_fold.X_d is not None:
                val_dset_fold.X_d = full_dset.X_d_scaler.transform(val_dset_fold.X_d)
        
        # Create data loaders with hyperparameter batch_size
        train_loader = data.build_dataloader(train_dset_fold, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        val_loader = data.build_dataloader(val_dset_fold, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        
        # Build model
        mp = nn.BondMessagePassing(d_h=message_hidden_dim, depth=depth)
        agg = nn.MeanAggregation()
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
        ffn = nn.RegressionFFN(
            output_transform=output_transform, 
            input_dim=message_hidden_dim, 
            hidden_dim=ffn_hidden_dim, 
            n_layers=ffn_num_layers
        )
        batch_norm = True
        metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]
        model = models.MPNN(mp, agg, ffn, batch_norm, metric_list)
        
        # Configure optimizer with hyperparameter learning_rate
        model.configure_optimizers = lambda: torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Create trainer with hyperparameter max_epochs
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=max_epochs,
            enable_progress_bar=False,  # Reduce output for k-fold
            logger=False,  # Disable logging for cleaner output
            # Ray integration
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
        )
        
        trainer = prepare_trainer(trainer)
        trainer.fit(model, train_loader, val_loader)
        
        # Get validation loss for this fold
        val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
        fold_val_losses.append(val_loss.item() if hasattr(val_loss, 'item') else val_loss)
    
    # Return mean validation loss across all folds
    mean_val_loss = np.mean(fold_val_losses)
    std_val_loss = np.std(fold_val_losses)
    
    print(f"K-fold CV results: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
    
    return mean_val_loss


# OPTION 2: Single Train/Val Split (Faster)
def train_model_single_split(config, train_dset, val_dset, num_workers, scaler):
    """Updated train_model function with new hyperparameters"""
    # Extract hyperparameters
    depth = int(config["depth"])
    ffn_hidden_dim = int(config["ffn_hidden_dim"])
    ffn_num_layers = int(config["ffn_num_layers"])
    message_hidden_dim = int(config["message_hidden_dim"])
    learning_rate = config["learning_rate"]
    batch_size = int(config["batch_size"])
    max_epochs = int(config["max_epochs"])

    # Create data loaders with hyperparameter batch_size
    train_loader = data.build_dataloader(train_dset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = data.build_dataloader(val_dset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Build model
    mp = nn.BondMessagePassing(d_h=message_hidden_dim, depth=depth)
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform, input_dim=message_hidden_dim, hidden_dim=ffn_hidden_dim, n_layers=ffn_num_layers)
    batch_norm = True
    metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]
    model = models.MPNN(mp, agg, ffn, batch_norm, metric_list)
    
    # Configure optimizer with hyperparameter learning_rate
    model.configure_optimizers = lambda: torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create trainer with hyperparameter max_epochs
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        # Ray integration
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
    )

    trainer = prepare_trainer(trainer)
    trainer.fit(model, train_loader, val_loader)