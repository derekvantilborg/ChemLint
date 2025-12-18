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
        "warning": "⚠️ This is a crude filtering tool. Results should be manually validated and used with caution.",
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
        "warning": "⚠️ Lipinski's Rule of Five is a crude guideline, not a strict rule. Many successful drugs violate these criteria. Use with caution and validate results.",
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
        "warning": "⚠️ Veber's rules are crude predictors of oral bioavailability. Many factors beyond TPSA and rotatable bonds affect absorption. Use with caution.",
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
        "warning": "⚠️ PAINS filters are controversial and may remove valid compounds. Context-dependent - a PAINS hit in one assay may be fine in another. Use with caution.",
        "note": note_text
    }


def filter_by_lead_likeness(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    explanation: str = "Filter by lead-likeness criteria",
    strict: bool = True
) -> dict:
    """
    Filter dataset by lead-likeness rules for hit-to-lead optimization.
    
    Calculates required properties on-the-fly from SMILES and filters molecules that pass:
    - Molecular weight: 200-350 Da (strict) or 150-400 Da (lenient)
    - LogP ≤ 3.5 (strict) or ≤ 4.0 (lenient)
    - Rotatable bonds ≤ 7 (strict) or ≤ 10 (lenient)
    - Number of rings ≥ 1 (at least one ring system required)
    
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
    explanation : str, default='Filter by lead-likeness criteria'
        Description of this filtering operation.
    strict : bool, default=True
        If True, use strict lead-likeness criteria. If False, use lenient criteria.
    
    Returns
    -------
    dict
        Contains output_filename, n_input, n_output, n_removed, n_invalid_smiles,
        percent_retained, criteria_applied, lead_properties_added, columns, preview, and note.
    
    Examples
    --------
    Apply strict lead-likeness filtering:
    
        result = filter_by_lead_likeness(
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            smiles_column='smiles',
            output_filename='lead_like',
            strict=True
        )
        
        print(f"Retained {result['percent_retained']:.1f}% of molecules")
    
    Apply lenient lead-likeness filtering:
    
        result = filter_by_lead_likeness(
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            smiles_column='smiles',
            output_filename='lead_like_lenient',
            strict=False
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
    
    # Calculate lead-likeness properties on-the-fly
    df_with_props = df.copy()
    
    mw_list = []
    logp_list = []
    rot_bonds_list = []
    ring_count_list = []
    aromatic_rings_list = []
    total_rings_list = []
    valid_mask = []
    
    for smiles in df_with_props[smiles_column]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            ring_count = Lipinski.RingCount(mol)
            aromatic_rings = Lipinski.NumAromaticRings(mol)
            total_rings = ring_count  # Total ring count
            
            mw_list.append(mw)
            logp_list.append(logp)
            rot_bonds_list.append(rot_bonds)
            ring_count_list.append(ring_count)
            aromatic_rings_list.append(aromatic_rings)
            total_rings_list.append(total_rings)
            valid_mask.append(True)
        else:
            # Invalid SMILES - will be filtered out
            mw_list.append(None)
            logp_list.append(None)
            rot_bonds_list.append(None)
            ring_count_list.append(None)
            aromatic_rings_list.append(None)
            total_rings_list.append(None)
            valid_mask.append(False)
    
    # Add properties as columns
    df_with_props['MolWt'] = mw_list
    df_with_props['MolLogP'] = logp_list
    df_with_props['NumRotatableBonds'] = rot_bonds_list
    df_with_props['RingCount'] = ring_count_list
    df_with_props['NumAromaticRings'] = aromatic_rings_list
    df_with_props['TotalRings'] = total_rings_list
    
    # Remove invalid SMILES first
    df_with_props = df_with_props[valid_mask].copy()
    n_invalid = n_input - len(df_with_props)
    
    # Define lead-likeness criteria based on strict/lenient mode
    if strict:
        property_ranges = {
            'MolWt': (200, 350),
            'MolLogP': (-float('inf'), 3.5),
            'NumRotatableBonds': (0, 7),
            'TotalRings': (1, float('inf'))  # At least 1 ring
        }
        criteria_mode = "strict"
    else:
        property_ranges = {
            'MolWt': (150, 400),
            'MolLogP': (-float('inf'), 4.0),
            'NumRotatableBonds': (0, 10),
            'TotalRings': (1, float('inf'))  # At least 1 ring
        }
        criteria_mode = "lenient"
    
    # Apply filters
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
            'max': float(max_val) if max_val != float('inf') else 'no_limit',
            'n_removed': n_removed_this_filter
        })
    
    n_output = len(df_filtered)
    n_removed = n_input - n_output
    percent_retained = (n_output / n_input * 100) if n_input > 0 else 0.0
    
    # Store output
    output_filename_stored = _store_resource(
        df_filtered,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    lead_properties = ['MolWt', 'MolLogP', 'NumRotatableBonds', 'RingCount', 'NumAromaticRings', 'TotalRings']
    
    if strict:
        criteria_text = "MW:200-350, LogP≤3.5, RotBonds≤7, Rings≥1"
    else:
        criteria_text = "MW:150-400, LogP≤4.0, RotBonds≤10, Rings≥1"
    
    return {
        "output_filename": output_filename_stored,
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "n_invalid_smiles": n_invalid,
        "percent_retained": percent_retained,
        "criteria_mode": criteria_mode,
        "filters_applied": filters_applied,
        "lead_properties_added": lead_properties,
        "columns": df_filtered.columns.tolist(),
        "warning": "⚠️ Lead-likeness rules are crude guidelines for hit-to-lead optimization. Optimal ranges vary by target class and project goals. Use with caution.",
        "note": (
            f"Lead-likeness filtering ({criteria_mode}): {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Removed {n_invalid} invalid SMILES. "
            f"Added properties: {', '.join(lead_properties)}. "
            f"Criteria: {criteria_text}."
        )
    }


def filter_by_rule_of_three(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    explanation: str = "Filter by Rule of Three",
    strict: bool = True
) -> dict:
    """
    Filter dataset by Rule of Three for fragment-based drug discovery.
    
    Calculates required properties on-the-fly from SMILES and filters molecules that pass:
    - Molecular weight ≤ 300 Da (strict) or ≤ 350 Da (lenient)
    - LogP ≤ 3 (strict) or ≤ 3.5 (lenient)
    - H-bond donors ≤ 3 (strict) or ≤ 4 (lenient)
    - H-bond acceptors ≤ 3 (strict) or ≤ 6 (lenient)
    - Rotatable bonds ≤ 3 (strict) or ≤ 5 (lenient)
    - TPSA ≤ 60 Ų (strict) or ≤ 90 Ų (lenient)
    
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
    explanation : str, default='Filter by Rule of Three'
        Description of this filtering operation.
    strict : bool, default=True
        If True, use strict Rule of Three criteria. If False, use lenient criteria.
    
    Returns
    -------
    dict
        Contains output_filename, n_input, n_output, n_removed, n_invalid_smiles,
        percent_retained, criteria_applied, ro3_properties_added, columns, preview, and note.
    
    Examples
    --------
    Apply strict Rule of Three filtering:
    
        result = filter_by_rule_of_three(
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            smiles_column='smiles',
            output_filename='ro3_fragments',
            strict=True
        )
        
        print(f"Retained {result['percent_retained']:.1f}% fragment-like molecules")
    
    Apply lenient Rule of Three filtering:
    
        result = filter_by_rule_of_three(
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            smiles_column='smiles',
            output_filename='ro3_lenient',
            strict=False
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
    
    # Calculate Rule of Three properties on-the-fly
    df_with_props = df.copy()
    
    mw_list = []
    logp_list = []
    hbd_list = []
    hba_list = []
    rot_bonds_list = []
    tpsa_list = []
    valid_mask = []
    
    for smiles in df_with_props[smiles_column]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mw_list.append(Descriptors.MolWt(mol))
            logp_list.append(Descriptors.MolLogP(mol))
            hbd_list.append(Lipinski.NumHDonors(mol))
            hba_list.append(Lipinski.NumHAcceptors(mol))
            rot_bonds_list.append(Descriptors.NumRotatableBonds(mol))
            tpsa_list.append(Descriptors.TPSA(mol))
            valid_mask.append(True)
        else:
            # Invalid SMILES - will be filtered out
            mw_list.append(None)
            logp_list.append(None)
            hbd_list.append(None)
            hba_list.append(None)
            rot_bonds_list.append(None)
            tpsa_list.append(None)
            valid_mask.append(False)
    
    # Add properties as columns
    df_with_props['MolWt'] = mw_list
    df_with_props['MolLogP'] = logp_list
    df_with_props['NumHDonors'] = hbd_list
    df_with_props['NumHAcceptors'] = hba_list
    df_with_props['NumRotatableBonds'] = rot_bonds_list
    df_with_props['TPSA'] = tpsa_list
    
    # Remove invalid SMILES first
    df_with_props = df_with_props[valid_mask].copy()
    n_invalid = n_input - len(df_with_props)
    
    # Define Rule of Three criteria based on strict/lenient mode
    if strict:
        property_ranges = {
            'MolWt': (0, 300),
            'MolLogP': (-float('inf'), 3),
            'NumHDonors': (0, 3),
            'NumHAcceptors': (0, 3),
            'NumRotatableBonds': (0, 3),
            'TPSA': (0, 60)
        }
        criteria_mode = "strict"
    else:
        property_ranges = {
            'MolWt': (0, 350),
            'MolLogP': (-float('inf'), 3.5),
            'NumHDonors': (0, 4),
            'NumHAcceptors': (0, 6),
            'NumRotatableBonds': (0, 5),
            'TPSA': (0, 90)
        }
        criteria_mode = "lenient"
    
    # Apply filters
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
    output_filename_stored = _store_resource(
        df_filtered,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    ro3_properties = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'TPSA']
    
    if strict:
        criteria_text = "MW≤300, LogP≤3, HBD≤3, HBA≤3, RotBonds≤3, TPSA≤60"
    else:
        criteria_text = "MW≤350, LogP≤3.5, HBD≤4, HBA≤6, RotBonds≤5, TPSA≤90"
    
    return {
        "output_filename": output_filename_stored,
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "n_invalid_smiles": n_invalid,
        "percent_retained": percent_retained,
        "criteria_mode": criteria_mode,
        "filters_applied": filters_applied,
        "ro3_properties_added": ro3_properties,
        "columns": df_filtered.columns.tolist(),
        "warning": "⚠️ Rule of Three is a crude guideline for fragment libraries. Fragment quality depends heavily on binding mode and target. Use with caution.",
        "note": (
            f"Rule of Three filtering ({criteria_mode}): {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Removed {n_invalid} invalid SMILES. "
            f"Added properties: {', '.join(ro3_properties)}. "
            f"Criteria: {criteria_text}."
        )
    }


def filter_by_qed(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    output_filename: str,
    explanation: str = "Filter by QED score",
    min_qed: float = 0.5
) -> dict:
    """
    Filter dataset by QED (Quantitative Estimate of Drug-likeness) score.
    
    Calculates QED descriptor on-the-fly from SMILES and filters molecules with QED ≥ min_qed.
    QED is a composite score (0-1) combining MW, LogP, HBA, HBD, PSA, rotatable bonds,
    aromatic rings, and structural alerts. Higher scores indicate more drug-like molecules.
    
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
    explanation : str, default='Filter by QED score'
        Description of this filtering operation.
    min_qed : float, default=0.5
        Minimum QED score threshold (0-1). Typical values:
        - 0.5: Moderate drug-likeness
        - 0.6: Good drug-likeness
        - 0.7: High drug-likeness
    
    Returns
    -------
    dict
        Contains output_filename, n_input, n_output, n_removed, n_invalid_smiles,
        percent_retained, min_qed_threshold, mean_qed, median_qed, columns, preview, and note.
    
    Examples
    --------
    Filter by moderate QED threshold:
    
        result = filter_by_qed(
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            smiles_column='smiles',
            output_filename='qed_filtered',
            min_qed=0.5
        )
        
        print(f"Retained {result['percent_retained']:.1f}% with QED ≥ 0.5")
    
    Filter by high drug-likeness threshold:
    
        result = filter_by_qed(
            input_filename='molecules_AB12CD34.csv',
            project_manifest_path='/path/to/manifest.json',
            smiles_column='smiles',
            output_filename='high_qed',
            min_qed=0.7
        )
    """
    from rdkit.Chem import QED
    
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
    
    # Validate min_qed threshold
    if not 0 <= min_qed <= 1:
        raise ValueError(
            f"min_qed must be between 0 and 1, got {min_qed}."
        )
    
    # Calculate QED scores on-the-fly
    df_with_qed = df.copy()
    
    qed_scores = []
    valid_mask = []
    
    for smiles in df_with_qed[smiles_column]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                qed_score = QED.qed(mol)
                qed_scores.append(qed_score)
                valid_mask.append(True)
            except:
                # QED calculation failed
                qed_scores.append(None)
                valid_mask.append(False)
        else:
            # Invalid SMILES
            qed_scores.append(None)
            valid_mask.append(False)
    
    # Add QED as column
    df_with_qed['QED'] = qed_scores
    
    # Remove invalid SMILES first
    df_with_qed = df_with_qed[valid_mask].copy()
    n_invalid = n_input - len(df_with_qed)
    
    # Apply QED filter
    df_filtered = df_with_qed[df_with_qed['QED'] >= min_qed].copy()
    
    n_output = len(df_filtered)
    n_removed = n_input - n_output
    percent_retained = (n_output / n_input * 100) if n_input > 0 else 0.0
    
    # Calculate QED statistics for filtered set
    if n_output > 0:
        mean_qed = float(df_filtered['QED'].mean())
        median_qed = float(df_filtered['QED'].median())
    else:
        mean_qed = 0.0
        median_qed = 0.0
    
    # Store output
    output_filename_stored = _store_resource(
        df_filtered,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_filename_stored,
        "n_input": n_input,
        "n_output": n_output,
        "n_removed": n_removed,
        "n_invalid_smiles": n_invalid,
        "percent_retained": percent_retained,
        "min_qed_threshold": min_qed,
        "mean_qed": mean_qed,
        "median_qed": median_qed,
        "columns": df_filtered.columns.tolist(),
        "warning": "⚠️ QED is a crude composite drug-likeness score. High QED doesn't guarantee success, and many approved drugs have low QED. Use with caution.",
        "note": (
            f"QED filtering: {n_input} → {n_output} molecules ({percent_retained:.1f}% retained). "
            f"Removed {n_invalid} invalid SMILES. "
            f"Threshold: QED ≥ {min_qed}. "
            f"Filtered set: mean QED = {mean_qed:.3f}, median QED = {median_qed:.3f}."
        )
    }
