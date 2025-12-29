"""
Activity cliff detection for identifying structurally similar molecules with large activity differences.

Activity cliffs are pairs of molecules that are highly similar in structure but have significantly
different biological activities. These are important for SAR analysis and understanding the 
molecular basis of activity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from molml_mcp.infrastructure.resources import _load_resource, _store_resource


def _compute_fold_difference_matrix(activity_values: np.ndarray) -> np.ndarray:
    """
    Compute pairwise fold-difference matrix from activity values.
    
    ⚠️ IMPORTANT: Expects LINEAR SCALE activity values (IC50_nM, Ki_nM, EC50_nM, etc.).
    
    Fold-difference is calculated as the simple ratio: max(val_i, val_j) / min(val_i, val_j)
    
    This ensures fold-difference is always >= 1.0, representing how many times more
    potent one molecule is compared to another.
    
    Examples with IC50 in nM:
    - IC50=100nM vs IC50=10nM: fold = 100/10 = 10-fold
    - IC50=1000nM vs IC50=10nM: fold = 1000/10 = 100-fold
    - IC50=50nM vs IC50=25nM: fold = 50/25 = 2-fold
    
    ⚠️ DO NOT use log-scale data (pIC50, pKi):
    - pIC50=7.0 vs pIC50=8.0 would give 8.0/7.0 = 1.14-fold (WRONG!)
    - The actual fold-difference is 10-fold
    - Convert first: IC50_nM = 10^(9 - pIC50) * 1e9
    
    Parameters
    ----------
    activity_values : np.ndarray
        1D array of activity values for each molecule in LINEAR SCALE.
        Examples: IC50 in nM, Ki in nM, EC50 in μM, etc.
        
    Returns
    -------
    np.ndarray
        Symmetric matrix of shape (n_molecules, n_molecules) with fold-differences.
        Diagonal values are 1.0 (molecule compared to itself).
        
    Example
    -------
    >>> activities = np.array([100.0, 10.0, 50.0])  # IC50 in nM
    >>> fold_matrix = _compute_fold_difference_matrix(activities)
    >>> fold_matrix[0, 1]  # 100 / 10 = 10.0
    10.0
    """
    n_molecules = len(activity_values)
    fold_diff_matrix = np.ones((n_molecules, n_molecules))
    
    for i in range(n_molecules):
        for j in range(i + 1, n_molecules):
            # Simple ratio: bigger / smaller
            val_i = activity_values[i]
            val_j = activity_values[j]
            
            if val_i == 0 or val_j == 0:
                # Avoid division by zero - set to very large fold-difference
                fold_diff = np.inf
            else:
                fold_diff = max(val_i, val_j) / min(val_i, val_j)
            
            fold_diff_matrix[i, j] = fold_diff
            fold_diff_matrix[j, i] = fold_diff  # Symmetric
    
    return fold_diff_matrix


def annotate_activity_cliff_molecules(
    dataset_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    activity_column: str,
    similarity_matrix_filename: str,
    output_filename: str,
    explanation: str = "Annotated molecules with activity cliff information",
    similarity_threshold: float = 0.9,
    fold_difference_threshold: float = 10.0,
) -> Dict:
    """
    Annotate each molecule in the dataset with activity cliff information.
    
    Instead of returning all pairwise cliffs (which can be 100+ combinations), this function
    adds columns to the original dataset indicating:
    - Whether each molecule participates in any activity cliff
    - The strongest cliff partner (similar molecule with largest activity difference)
    
    This is much more practical for large datasets while preserving key insights.
    
    **What is the "strongest" cliff partner?**
    Among all molecules that meet the similarity threshold, the strongest partner is the one
    with the LARGEST FOLD-DIFFERENCE in activity. This identifies the most dramatic activity
    cliff for each molecule.
    
    ⚠️ CRITICAL: Activity values MUST be in LINEAR SCALE (IC50_nM, Ki_nM, EC50_μM, etc.)
                DO NOT use log-scale data (pIC50, pKi, pEC50)!
                
    If you have log-scale data (pIC50, pKi, etc.):
    Convert first using:
    - IC50_nM = 10^(9 - pIC50)  [converts pIC50 to IC50 in nM]
    - Ki_nM = 10^(9 - pKi)      [converts pKi to Ki in nM]
    - EC50_nM = 10^(9 - pEC50)  [converts pEC50 to EC50 in nM]
    
    Parameters
    ----------
    dataset_filename : str
        Input dataset filename from manifest (must contain SMILES and activity columns).
    project_manifest_path : str
        Path to the project manifest.json file.
    smiles_column : str
        Name of the column containing SMILES strings.
    activity_column : str
        Name of the column containing activity values in LINEAR SCALE.
        ✅ CORRECT: IC50_nM, Ki_nM, EC50_μM, Kd_nM
        ❌ WRONG: pIC50, pKi, pEC50, pKd (log scale)
    similarity_matrix_filename : str
        Filename of the precomputed similarity matrix (from compute_similarity_matrix).
    output_filename : str
        Name for the output dataset (will be versioned with unique ID).
    explanation : str
        Human-readable description of this operation.
    similarity_threshold : float, default=0.9
        Minimum Tanimoto similarity for molecules to be considered structurally similar.
        Typical range: 0.85-0.95
    fold_difference_threshold : float, default=10.0
        Minimum fold-difference in activity to be considered an activity cliff.
        For LINEAR SCALE: fold = max(IC50_i, IC50_j) / min(IC50_i, IC50_j)
        Typical range: 10-100
        
    Returns
    -------
    dict
        Contains:
        - output_filename: Versioned filename of annotated dataset
        - n_molecules: Total molecules in dataset
        - n_cliff_molecules: Number of molecules participating in cliffs
        - n_non_cliff_molecules: Number of molecules not in any cliff
        - columns: List of all columns (including new annotation columns)
        - preview: First 5 rows
        - summary: Human-readable summary
        
    New Columns Added
    -----------------
    - is_activity_cliff_molecule : str ('yes' or 'no')
        'yes' if molecule participates in at least one activity cliff, 'no' otherwise
    - n_activity_cliff_partners : int
        Number of cliff partners this molecule has (0 if no cliffs)
    - strongest_cliff_partner_idx : int or NaN
        Index of the molecule that forms the strongest cliff (largest fold-difference)
        among all similar molecules (similarity > threshold)
    - strongest_cliff_partner_smiles : str or NaN
        SMILES of the strongest cliff partner
        
    Example
    -------
    >>> result = annotate_activity_cliff_molecules(
    ...     'dataset.csv',
    ...     '/path/to/manifest.json',
    ...     'SMILES',
    ...     'IC50_nM',  # LINEAR SCALE!
    ...     'similarity_matrix.joblib',
    ...     'annotated_dataset',
    ...     similarity_threshold=0.9,
    ...     fold_difference_threshold=10.0
    ... )
    >>> print(result['summary'])
    Found 45 activity cliff molecules out of 150 total (30.0%).
    These molecules participate in 127 activity cliff pairs.
    """
    # Print warning about linear scale requirement
    print(f"⚠️  IMPORTANT: Activity values must be in LINEAR SCALE (IC50_nM, Ki_nM, etc.)")
    print(f"             DO NOT use log-scale (pIC50, pKi, etc.)!")
    print(f"             Analyzing column '{activity_column}' - please verify this is LINEAR SCALE.\n")
    
    # Load dataset
    df = _load_resource(project_manifest_path, dataset_filename)
    n_molecules = len(df)
    
    # Validate columns
    if smiles_column not in df.columns:
        raise ValueError(f"SMILES column '{smiles_column}' not found. Available: {df.columns.tolist()}")
    if activity_column not in df.columns:
        raise ValueError(f"Activity column '{activity_column}' not found. Available: {df.columns.tolist()}")
    
    # Load similarity matrix
    sim_matrix = _load_resource(project_manifest_path, similarity_matrix_filename)
    
    # Validate matrix dimensions
    if sim_matrix.shape[0] != n_molecules or sim_matrix.shape[1] != n_molecules:
        raise ValueError(
            f"Similarity matrix shape {sim_matrix.shape} does not match dataset size {n_molecules}. "
            f"Ensure the matrix was computed for this exact dataset."
        )
    
    # Get activity values and handle NaN
    activity_values = df[activity_column].values
    valid_mask = ~np.isnan(activity_values)
    n_valid = valid_mask.sum()
    n_nan = (~valid_mask).sum()
    
    if n_nan > 0:
        print(f"⚠️  Warning: {n_nan} molecules have NaN activity values and will be marked as non-cliff molecules.\n")
    
    # Compute fold-difference matrix
    fold_diff_matrix = _compute_fold_difference_matrix(activity_values)
    
    # Initialize annotation columns
    df['is_activity_cliff_molecule'] = 'no'
    df['n_activity_cliff_partners'] = 0
    df['strongest_cliff_partner_idx'] = np.nan
    df['strongest_cliff_partner_smiles'] = np.nan
    
    # Find cliffs for each molecule
    total_cliff_pairs = 0
    
    for i in range(n_molecules):
        # Skip if this molecule has NaN activity
        if not valid_mask[i]:
            continue
            
        # Find all molecules that form cliffs with molecule i
        # Requirements: similarity > threshold, fold-diff > threshold, valid activity, not self
        cliff_mask = (
            (sim_matrix[i, :] > similarity_threshold) &
            (fold_diff_matrix[i, :] > fold_difference_threshold) &
            valid_mask &
            (np.arange(n_molecules) != i)
        )
        
        cliff_indices = np.where(cliff_mask)[0]
        n_cliffs = len(cliff_indices)
        
        if n_cliffs > 0:
            df.at[i, 'is_activity_cliff_molecule'] = 'yes'
            df.at[i, 'n_activity_cliff_partners'] = n_cliffs
            total_cliff_pairs += n_cliffs
            
            # Find the strongest cliff partner (largest fold-difference among cliffs)
            cliff_fold_diffs = fold_diff_matrix[i, cliff_indices]
            strongest_idx_in_cliffs = np.argmax(cliff_fold_diffs)
            strongest_partner_idx = cliff_indices[strongest_idx_in_cliffs]
            
            df.at[i, 'strongest_cliff_partner_idx'] = int(strongest_partner_idx)
            df.at[i, 'strongest_cliff_partner_smiles'] = df.iloc[strongest_partner_idx][smiles_column]
    
    # Count each pair once (currently counted twice)
    total_cliff_pairs = total_cliff_pairs // 2
    
    # Count cliff molecules
    n_cliff_molecules = (df['is_activity_cliff_molecule'] == 'yes').sum()
    n_non_cliff_molecules = n_molecules - n_cliff_molecules
    
    # Store annotated dataset
    output_id = _store_resource(
        df,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    # Create summary message
    if n_cliff_molecules == 0:
        summary_msg = (
            f"No activity cliff molecules found with similarity > {similarity_threshold} "
            f"and fold-difference > {fold_difference_threshold}. "
            f"Consider lowering thresholds or checking data quality."
        )
    else:
        pct_cliff = 100.0 * n_cliff_molecules / n_molecules
        summary_msg = (
            f"Found {n_cliff_molecules} activity cliff molecules out of {n_molecules} total ({pct_cliff:.1f}%). "
            f"These molecules participate in {total_cliff_pairs} activity cliff pairs. "
            f"Activity column: {activity_column}."
        )
    
    return {
        "output_filename": output_id,
        "n_molecules": n_molecules,
        "n_molecules_with_nan_activity": int(n_nan),
        "n_cliff_molecules": int(n_cliff_molecules),
        "n_non_cliff_molecules": int(n_non_cliff_molecules),
        "n_total_cliff_pairs": int(total_cliff_pairs),
        "similarity_threshold": similarity_threshold,
        "fold_difference_threshold": fold_difference_threshold,
        "activity_column": activity_column,
        "columns": df.columns.tolist(),
        "summary": summary_msg
    }


def get_all_activity_cliff_tools():
    """
    Returns a list of MCP-exposed activity cliff functions for server registration.
    """
    return [
        annotate_activity_cliff_molecules,
    ]
