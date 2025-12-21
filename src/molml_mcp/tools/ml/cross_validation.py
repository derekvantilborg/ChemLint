
# k-fold (random)
# stratified k-fold
# leave-p-out
# monte carlo (random train/val splits)
# group-based splitting (scaffold or cluster)


def cv_splits_kfold(k: int, smiles: list, val_size: float, random_state: int, shuffle: bool = True) -> list[dict]:
    """
    Split data into k folds for cross-validation.
    
    Args:
        k: Number of folds
        smiles: List of SMILES strings
        val_size: Fraction of data to use for validation (not used in k-fold, kept for API consistency)
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle data before splitting
    
    Returns:
        List of dicts with keys 'train_smiles' and 'val_smiles'
        [{'train_smiles': [...], 'val_smiles': [...]}, ...]
    """
    from sklearn.model_selection import KFold
    import numpy as np
    
    smiles_array = np.array(smiles)
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state if shuffle else None)
    
    splits = []
    for train_idx, val_idx in kf.split(smiles_array):
        splits.append({
            'train_smiles': smiles_array[train_idx].tolist(),
            'val_smiles': smiles_array[val_idx].tolist()
        })
    
    return splits

def cv_splits_stratifiedkfold(k: int, smiles: list, y: list, val_size: float, random_state: int, shuffle: bool = True) -> list[dict]:
    """
    Split data into k stratified folds for cross-validation.
    
    Stratified splitting ensures each fold maintains the same class distribution as the original dataset.
    Important for imbalanced classification problems.
    
    Args:
        k: Number of folds
        smiles: List of SMILES strings
        y: List of labels (for stratification)
        val_size: Fraction of data to use for validation (not used in k-fold, kept for API consistency)
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle data before splitting
    
    Returns:
        List of dicts with keys 'train_smiles' and 'val_smiles'
        [{'train_smiles': [...], 'val_smiles': [...]}, ...]
    """
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    
    smiles_array = np.array(smiles)
    y_array = np.array(y)
    
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state if shuffle else None)
    
    splits = []
    for train_idx, val_idx in skf.split(smiles_array, y_array):
        splits.append({
            'train_smiles': smiles_array[train_idx].tolist(),
            'val_smiles': smiles_array[val_idx].tolist()
        })
    
    return splits

def cv_splits_leavepout(p: int, smiles: list, val_size: float, random_state: int, max_splits: int = None) -> list[dict]:
    """
    Split data using Leave-P-Out cross-validation.
    
    In Leave-P-Out CV, p samples are left out for validation in each fold, and the model is trained
    on the remaining samples. This generates C(n, p) splits where n is the total number of samples.
    
    WARNING: Can generate a very large number of splits! For n=100 and p=2, this creates 4,950 splits.
    Use max_splits to limit the number of splits for computational efficiency.
    
    Args:
        p: Number of samples to leave out in each fold
        smiles: List of SMILES strings
        val_size: Fraction of data to use for validation (not used, kept for API consistency)
        random_state: Random seed for reproducibility (used if max_splits is set)
        max_splits: Maximum number of splits to generate (if None, generates all possible splits)
    
    Returns:
        List of dicts with keys 'train_smiles' and 'val_smiles'
        [{'train_smiles': [...], 'val_smiles': [...]}, ...]
    """
    from sklearn.model_selection import LeavePOut
    import numpy as np
    
    smiles_array = np.array(smiles)
    lpo = LeavePOut(p=p)
    
    splits = []
    for i, (train_idx, val_idx) in enumerate(lpo.split(smiles_array)):
        if max_splits is not None and i >= max_splits:
            break
        splits.append({
            'train_smiles': smiles_array[train_idx].tolist(),
            'val_smiles': smiles_array[val_idx].tolist()
        })
    
    # If max_splits is set and we want random sampling, shuffle the splits
    if max_splits is not None and random_state is not None and len(splits) > max_splits:
        np.random.seed(random_state)
        indices = np.random.choice(len(splits), max_splits, replace=False)
        splits = [splits[i] for i in sorted(indices)]
    
    return splits

def cv_splits_montecarlo(n_splits: int, smiles: list, val_size: float, random_state: int) -> list[dict]:
    """
    Split data using Monte Carlo cross-validation (repeated random sub-sampling).
    
    Monte Carlo CV randomly splits the data into training and validation sets n_splits times.
    Unlike k-fold, samples may appear in validation multiple times or not at all across splits.
    Useful when you want to control the exact validation size and don't need exhaustive coverage.
    
    Args:
        n_splits: Number of random train/val splits to generate
        smiles: List of SMILES strings
        val_size: Fraction of data to use for validation (e.g., 0.2 for 20%)
        random_state: Random seed for reproducibility
    
    Returns:
        List of dicts with keys 'train_smiles' and 'val_smiles'
        [{'train_smiles': [...], 'val_smiles': [...]}, ...]
    """
    from sklearn.model_selection import ShuffleSplit
    import numpy as np
    
    smiles_array = np.array(smiles)
    ss = ShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=random_state)
    
    splits = []
    for train_idx, val_idx in ss.split(smiles_array):
        splits.append({
            'train_smiles': smiles_array[train_idx].tolist(),
            'val_smiles': smiles_array[val_idx].tolist()
        })
    
    return splits

def cv_splits_cluster(k: int, smiles: list, clusters: list, val_size: float, random_state: int, shuffle: bool = True) -> list[dict]:
    """
    Split data into k folds based on pre-defined cluster assignments.
    
    Group-based splitting ensures that all molecules in the same cluster are kept together
    in either training or validation. This is critical for evaluating model generalization
    to new chemical scaffolds or structural clusters.
    
    Uses sklearn's GroupKFold under the hood, which ensures no cluster appears in both
    train and validation within the same fold.
    
    Args:
        k: Number of folds
        smiles: List of SMILES strings
        clusters: List of cluster assignments (one per SMILES). Can be integers, strings, or any hashable type.
        val_size: Fraction of data to use for validation (not used in k-fold, kept for API consistency)
        random_state: Random seed for reproducibility (used when shuffle=True)
        shuffle: Whether to shuffle the groups before splitting
    
    Returns:
        List of dicts with keys 'train_smiles' and 'val_smiles'
        [{'train_smiles': [...], 'val_smiles': [...]}, ...]
    
    Note:
        - Fold sizes may be unequal if clusters have different sizes
        - All molecules in the same cluster will be in the same fold
        - Use for scaffold-based or structural cluster-based CV
    """
    from sklearn.model_selection import GroupKFold
    import numpy as np
    
    if len(smiles) != len(clusters):
        raise ValueError(f"Length mismatch: {len(smiles)} SMILES but {len(clusters)} cluster assignments")
    
    smiles_array = np.array(smiles)
    clusters_array = np.array(clusters)
    
    # GroupKFold doesn't support shuffle directly, so we shuffle the groups manually if needed
    if shuffle:
        unique_clusters = np.unique(clusters_array)
        np.random.seed(random_state)
        shuffled_clusters = np.random.permutation(unique_clusters)
        
        # Create a mapping from old cluster ID to new shuffled order
        cluster_map = {old: new for new, old in enumerate(shuffled_clusters)}
        clusters_array = np.array([cluster_map[c] for c in clusters_array])
    
    gkf = GroupKFold(n_splits=k)
    
    splits = []
    for train_idx, val_idx in gkf.split(smiles_array, groups=clusters_array):
        splits.append({
            'train_smiles': smiles_array[train_idx].tolist(),
            'val_smiles': smiles_array[val_idx].tolist()
        })
    
    return splits


def cv_splits_scaffold(k: int, smiles: list, val_size: float, random_state: int, scaffold_type: str = 'bemis_murcko', shuffle: bool = True) -> list[dict]:
    """
    Split data into k folds based on Bemis-Murcko scaffolds.
    
    Automatically extracts scaffolds from SMILES, assigns each unique scaffold a cluster ID,
    and uses cluster-based splitting. Molecules without a scaffold are assigned to a separate
    'no_scaffold' cluster. This ensures models are evaluated on their ability to generalize
    to new chemical scaffolds.
    
    Args:
        k: Number of folds
        smiles: List of SMILES strings
        val_size: Fraction of data to use for validation (not used in k-fold, kept for API consistency)
        random_state: Random seed for reproducibility
        scaffold_type: Type of scaffold to extract ('bemis_murcko', 'generic', 'cyclic_skeleton')
        shuffle: Whether to shuffle the scaffold groups before splitting
    
    Returns:
        List of dicts with keys 'train_smiles' and 'val_smiles'
        [{'train_smiles': [...], 'val_smiles': [...]}, ...]
    
    Note:
        - Molecules with the same scaffold will always be in the same fold
        - Molecules without a scaffold are grouped together in a 'no_scaffold' cluster
        - Uses cv_splits_cluster internally after scaffold extraction
    """
    from molml_mcp.tools.core_mol.scaffolds import _get_scaffold
    
    # Extract scaffolds for all SMILES
    scaffold_list = []
    for smi in smiles:
        scaffold_smi, comment = _get_scaffold(smi, scaffold_type=scaffold_type)
        scaffold_list.append(scaffold_smi)
    
    # Create cluster assignments: map scaffold SMILES to cluster IDs
    # Molecules with None/no scaffold get their own cluster
    unique_scaffolds = {}
    cluster_id = 0
    
    # Reserve cluster 0 for molecules without scaffolds
    no_scaffold_cluster = 0
    cluster_id = 1
    
    clusters = []
    for scaffold_smi in scaffold_list:
        if scaffold_smi is None or scaffold_smi == '':
            # Assign to 'no_scaffold' cluster
            clusters.append(no_scaffold_cluster)
        else:
            # Assign to scaffold-specific cluster
            if scaffold_smi not in unique_scaffolds:
                unique_scaffolds[scaffold_smi] = cluster_id
                cluster_id += 1
            clusters.append(unique_scaffolds[scaffold_smi])
    
    # Check if we have enough unique scaffolds for k folds
    n_unique_clusters = len(set(clusters))
    if k > n_unique_clusters:
        raise ValueError(
            f"Cannot split into {k} folds with only {n_unique_clusters} unique scaffolds. "
            f"Reduce k to {n_unique_clusters} or fewer."
        )
    
    # Use cluster-based splitting
    return cv_splits_cluster(
        k=k,
        smiles=smiles,
        clusters=clusters,
        val_size=val_size,
        random_state=random_state,
        shuffle=shuffle
    )


def _cross_validate_and_eval(model, X, Y, cv_strategy: str, n_folds: int, random_state: int, task_type: str, metric: str) -> float:
    # internal function for cross validation in for hyperparam tuning. Returns the average metric across folds
    
    # create the CV splits

    # perform model training and evaluation for each fold using _train_ml_model() and _eval_single_ml_model()

    # return average metric across folds
    pass



