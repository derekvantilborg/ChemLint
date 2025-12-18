"""
Dataset filtering tools for property-based selection.
"""
from rdkit.Chem import Descriptors, Lipinski
from rdkit import Chem
from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from molml_mcp.tools.core_mol.pains import _check_smiles_for_pains


def filter_by_property_range(
    input_filename: str,
    project_manifest_path: str,
    property_ranges: dict[str, tuple[float, float]],
    output_filename: str,
    explanation: str
) -> dict:
    """
    Filter dataset by property ranges. Molecules must pass ALL criteria (AND logic).
    Ranges are inclusive: min ≤ value ≤ max.
    
    Parameters
    ----------
    input_filename : str
        Filename of the input dataset.
    project_manifest_path : str
        Path to the project's manifest.json file.
    property_ranges : dict[str, tuple[float, float]]
        Dictionary mapping column names to (min, max) tuples.
        Example: {'MolWt': (200, 500), 'TPSA': (0, 140)}
    output_filename : str
        Base name for the output filtered dataset.
    explanation : str
        Description of this filtering operation.
    
    Returns
    -------
    dict
        Contains output_filename, n_input, n_output, n_removed, percent_retained,
        filters_applied (per-filter statistics), columns, preview, and note.
    
    Examples
    --------
    Lipinski's Rule of Five:
    
        result = filter_by_property_range(
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            property_ranges={
                'MolWt': (0, 500),
                'MolLogP': (-5, 5),
                'NumHDonors': (0, 5),
                'NumHAcceptors': (0, 10)
            },
            output_filename='lipinski_filtered',
            explanation='Lipinski filtering'
        )
    
    Multiple property windows:
    
        result = filter_by_property_range(
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            property_ranges={
                'MolWt': (200, 500),
                'TPSA': (0, 140),
                'MolLogP': (-2, 5)
            },
            output_filename='filtered_druglike',
            explanation='Drug-like filtering'
        )
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_input = len(df)
    
    # Validate inputs
    if not property_ranges:
        raise ValueError("property_ranges cannot be empty. Provide at least one property range filter.")
    
    # Check all property columns exist
    missing_columns = [col for col in property_ranges.keys() if col not in df.columns]
    if missing_columns:
        available_columns = df.columns.tolist()
        raise ValueError(
            f"Property columns not found in dataset: {missing_columns}. "
            f"Available columns: {available_columns}"
        )
    
    # Validate ranges
    for prop, (min_val, max_val) in property_ranges.items():
        if min_val > max_val:
            raise ValueError(
                f"Invalid range for property '{prop}': min ({min_val}) > max ({max_val}). "
                f"Range must have min ≤ max."
            )
    
    # Apply filters
    df_filtered = df.copy()
    filters_applied = []
    
    for prop, (min_val, max_val) in property_ranges.items():
        # Create filter mask
        mask = (df_filtered[prop] >= min_val) & (df_filtered[prop] <= max_val)
        
        # Count before filtering
        n_before = len(df_filtered)
        
        # Apply filter
        df_filtered = df_filtered[mask]
        
        # Track what was applied
        n_after = len(df_filtered)
        n_removed_this_filter = n_before - n_after
        filters_applied.append({
            'property': prop,
            'min': float(min_val),
            'max': float(max_val),
            'n_removed': n_removed_this_filter
        })
    
    n_output = len(df_filtered)
    n_removed = n_input - n_output
    percent_retained = (n_output / n_input * 100) if n_input > 0 else 0.0
    
    # Store output
    output_filename = _store_resource(
        df_filtered,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_filename,
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "percent_retained": percent_retained,
        "filters_applied": filters_applied,
        "columns": df_filtered.columns.tolist(),
        "note": (
            f"Filtered dataset from {n_input} to {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Applied {len(property_ranges)} property range filters. "
            f"Removed {n_removed} molecules not passing all criteria."
        )
    }


def filter_by_lipinski_ro5(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    explanation: str
) -> dict:
    """
    Filter dataset by Lipinski's Rule of Five criteria.
    
    Calculates required properties on-the-fly from SMILES and filters molecules that pass ALL criteria:
    - Molecular weight ≤ 500 Da
    - LogP ≤ 5
    - H-bond donors ≤ 5
    - H-bond acceptors ≤ 10
    
    Parameters
    ----------
    input_filename : str
        Filename of the input dataset.
    project_manifest_path : str
        Path to the project's manifest.json file.
    smiles_column : str
        Name of the column containing SMILES strings.
    output_filename : str
        Base name for the output filtered dataset.
    explanation : str
        Description of this filtering operation.
    
    Returns
    -------
    dict
        Contains output_filename, n_input, n_output, n_removed, percent_retained,
        filters_applied (per-filter statistics), lipinski_properties_added (list of properties),
        columns, preview, and note.
    
    Examples
    --------
    Apply Lipinski filtering:
    
        result = filter_by_lipinski_ro5(
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            smiles_column='smiles',
            output_filename='lipinski_compliant',
            explanation='Lipinski Rule of Five filtering'
        )
        
        print(f"Retained {result['percent_retained']:.1f}% of molecules")
        print(f"Properties added: {result['lipinski_properties_added']}")
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_input = len(df)
    
    # Validate SMILES column exists
    if smiles_column not in df.columns:
        available_columns = df.columns.tolist()
        raise ValueError(
            f"SMILES column '{smiles_column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    # Calculate Lipinski properties on-the-fly
    df_with_props = df.copy()
    
    mw_list = []
    logp_list = []
    hbd_list = []
    hba_list = []
    valid_mask = []
    
    for smiles in df_with_props[smiles_column]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mw_list.append(Descriptors.MolWt(mol))
            logp_list.append(Descriptors.MolLogP(mol))
            hbd_list.append(Lipinski.NumHDonors(mol))
            hba_list.append(Lipinski.NumHAcceptors(mol))
            valid_mask.append(True)
        else:
            # Invalid SMILES - will be filtered out
            mw_list.append(None)
            logp_list.append(None)
            hbd_list.append(None)
            hba_list.append(None)
            valid_mask.append(False)
    
    # Add properties as columns
    df_with_props['MolWt'] = mw_list
    df_with_props['MolLogP'] = logp_list
    df_with_props['NumHDonors'] = hbd_list
    df_with_props['NumHAcceptors'] = hba_list
    
    # Remove invalid SMILES first
    df_with_props = df_with_props[valid_mask].copy()
    n_invalid = n_input - len(df_with_props)
    
    # Apply Lipinski Rule of Five filters
    property_ranges = {
        'MolWt': (0, 500),
        'MolLogP': (-float('inf'), 5),
        'NumHDonors': (0, 5),
        'NumHAcceptors': (0, 10)
    }
    
    df_filtered = df_with_props.copy()
    filters_applied = []
    
    for prop, (min_val, max_val) in property_ranges.items():
        mask = (df_filtered[prop] >= min_val) & (df_filtered[prop] <= max_val)
        n_before = len(df_filtered)
        df_filtered = df_filtered[mask]
        n_after = len(df_filtered)
        n_removed_this_filter = n_before - n_after
        filters_applied.append({
            'property': prop,
            'min': float(min_val) if min_val != -float('inf') else 'no_limit',
            'max': float(max_val),
            'n_removed': n_removed_this_filter
        })
    
    n_output = len(df_filtered)
    n_removed = n_input - n_output
    percent_retained = (n_output / n_input * 100) if n_input > 0 else 0.0
    
    # Store output
    output_filename = _store_resource(
        df_filtered,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    lipinski_properties = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors']
    
    return {
        "output_filename": output_filename,
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "n_invalid_smiles": n_invalid,
        "percent_retained": percent_retained,
        "filters_applied": filters_applied,
        "lipinski_properties_added": lipinski_properties,
        "columns": df_filtered.columns.tolist(),
        "note": (
            f"Lipinski Rule of Five filtering: {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Removed {n_invalid} invalid SMILES. "
            f"Added properties: {', '.join(lipinski_properties)}. "
            f"Criteria: MW≤500, LogP≤5, HBD≤5, HBA≤10."
        )
    }


def filter_by_veber_rules(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    explanation: str
) -> dict:
    """
    Filter dataset by Veber's rules for oral bioavailability.
    
    Calculates required properties on-the-fly from SMILES and filters molecules that pass ALL criteria:
    - Topological polar surface area (TPSA) ≤ 140 Ų
    - Number of rotatable bonds ≤ 10
    
    Parameters
    ----------
    input_filename : str
        Filename of the input dataset.
    project_manifest_path : str
        Path to the project's manifest.json file.
    smiles_column : str
        Name of the column containing SMILES strings.
    output_filename : str
        Base name for the output filtered dataset.
    explanation : str
        Description of this filtering operation.
    
    Returns
    -------
    dict
        Contains output_filename, n_input, n_output, n_removed, percent_retained,
        filters_applied (per-filter statistics), veber_properties_added (list of properties),
        columns, preview, and note.
    
    Examples
    --------
    Apply Veber filtering:
    
        result = filter_by_veber_rules(
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            smiles_column='smiles',
            output_filename='veber_compliant',
            explanation='Veber rules filtering'
        )
        
        print(f"Retained {result['percent_retained']:.1f}% of molecules")
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_input = len(df)
    
    # Validate SMILES column exists
    if smiles_column not in df.columns:
        available_columns = df.columns.tolist()
        raise ValueError(
            f"SMILES column '{smiles_column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    # Calculate Veber properties on-the-fly
    df_with_props = df.copy()
    
    tpsa_list = []
    rotatable_bonds_list = []
    valid_mask = []
    
    for smiles in df_with_props[smiles_column]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            tpsa_list.append(Descriptors.TPSA(mol))
            rotatable_bonds_list.append(Descriptors.NumRotatableBonds(mol))
            valid_mask.append(True)
        else:
            # Invalid SMILES - will be filtered out
            tpsa_list.append(None)
            rotatable_bonds_list.append(None)
            valid_mask.append(False)
    
    # Add properties as columns
    df_with_props['TPSA'] = tpsa_list
    df_with_props['NumRotatableBonds'] = rotatable_bonds_list
    
    # Remove invalid SMILES first
    df_with_props = df_with_props[valid_mask].copy()
    n_invalid = n_input - len(df_with_props)
    
    # Apply Veber rules filters
    property_ranges = {
        'TPSA': (0, 140),
        'NumRotatableBonds': (0, 10)
    }
    
    df_filtered = df_with_props.copy()
    filters_applied = []
    
    for prop, (min_val, max_val) in property_ranges.items():
        mask = (df_filtered[prop] >= min_val) & (df_filtered[prop] <= max_val)
        n_before = len(df_filtered)
        df_filtered = df_filtered[mask]
        n_after = len(df_filtered)
        n_removed_this_filter = n_before - n_after
        filters_applied.append({
            'property': prop,
            'min': float(min_val),
            'max': float(max_val),
            'n_removed': n_removed_this_filter
        })
    
    n_output = len(df_filtered)
    n_removed = n_input - n_output
    percent_retained = (n_output / n_input * 100) if n_input > 0 else 0.0
    
    # Store output
    output_filename = _store_resource(
        df_filtered,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    veber_properties = ['TPSA', 'NumRotatableBonds']
    
    return {
        "output_filename": output_filename,
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "n_invalid_smiles": n_invalid,
        "percent_retained": percent_retained,
        "filters_applied": filters_applied,
        "veber_properties_added": veber_properties,
        "columns": df_filtered.columns.tolist(),
        "preview": df_filtered.head(5).to_dict('records'),
        "note": (
            f"Veber rules filtering: {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Removed {n_invalid} invalid SMILES. "
            f"Added properties: {', '.join(veber_properties)}. "
            f"Criteria: TPSA≤140, RotatableBonds≤10."
        )
    }


def filter_by_pains(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    explanation: str = "Remove PAINS flagged molecules",
    action: str = 'drop'
) -> dict:
    """
    Filter dataset by removing or keeping PAINS-flagged molecules.
    
    Screens molecules for PAINS (Pan-Assay INterference compoundS) patterns using RDKit.
    PAINS are substructures that cause false positives in screening through non-specific
    binding, aggregation, or assay interference.
    
    Parameters
    ----------
    input_filename : str
        Filename of the input dataset.
    project_manifest_path : str
        Path to the project's manifest.json file.
    smiles_column : str
        Name of the column containing SMILES strings.
    output_filename : str
        Base name for the output filtered dataset.
    explanation : str, default='Remove PAINS flagged molecules'
        Description of this filtering operation.
    action : str, default='drop'
        Filter action: 'drop' removes PAINS hits, 'keep' retains only PAINS hits.
    
    Returns
    -------
    dict
        Contains output_filename, n_input, n_output, n_removed, n_pains_flagged,
        n_invalid_smiles, percent_retained, action, columns, preview, and note.
    
    Examples
    --------
    Remove PAINS-flagged molecules (default):
    
        result = filter_by_pains(
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            smiles_column='smiles',
            output_filename='pains_filtered',
            action='drop'
        )
        
        print(f"Removed {result['n_pains_flagged']} PAINS-flagged molecules")
    
    Keep only PAINS-flagged molecules for analysis:
    
        result = filter_by_pains(
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            smiles_column='smiles',
            output_filename='pains_only',
            explanation='Extract PAINS hits for analysis',
            action='keep'
        )
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    n_input = len(df)
    
    # Validate SMILES column exists
    if smiles_column not in df.columns:
        available_columns = df.columns.tolist()
        raise ValueError(
            f"SMILES column '{smiles_column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    # Validate action
    if action not in ['drop', 'keep']:
        raise ValueError(
            f"Invalid action '{action}'. Must be 'drop' (remove PAINS) or 'keep' (retain only PAINS)."
        )
    
    # Check all molecules for PAINS
    df_with_pains = df.copy()
    pains_results = []
    
    for smiles in df_with_pains[smiles_column]:
        pains_result = _check_smiles_for_pains(smiles)
        pains_results.append(pains_result)
    
    # Add PAINS check results as column
    df_with_pains['pains_check'] = pains_results
    
    # Count different categories
    n_passed = sum(1 for r in pains_results if r == 'Passed')
    n_pains = sum(1 for r in pains_results if r.startswith('PAINS:'))
    n_failed = sum(1 for r in pains_results if r.startswith('Failed:'))
    
    # Apply filter based on action
    if action == 'drop':
        # Keep only molecules that passed PAINS check
        df_filtered = df_with_pains[df_with_pains['pains_check'] == 'Passed'].copy()
    else:  # action == 'keep'
        # Keep only PAINS-flagged molecules
        df_filtered = df_with_pains[df_with_pains['pains_check'].str.startswith('PAINS:')].copy()
    
    n_output = len(df_filtered)
    n_removed = n_input - n_output
    percent_retained = (n_output / n_input * 100) if n_input > 0 else 0.0
    
    # Store output
    output_filename = _store_resource(
        df_filtered,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    if action == 'drop':
        note_text = (
            f"PAINS filtering: {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Removed {n_pains} PAINS-flagged and {n_failed} invalid molecules. "
            f"Kept {n_passed} clean molecules."
        )
    else:
        note_text = (
            f"PAINS extraction: {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Kept {n_pains} PAINS-flagged molecules for analysis. "
            f"Removed {n_passed} clean and {n_failed} invalid molecules."
        )
    
    return {
        "output_filename": output_filename,
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "n_pains_flagged": n_pains,
        "n_invalid_smiles": n_failed,
        "percent_retained": percent_retained,
        "action": action,
        "columns": df_filtered.columns.tolist(),
        "note": note_text
    }
