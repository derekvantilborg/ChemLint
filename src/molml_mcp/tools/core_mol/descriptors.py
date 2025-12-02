from rdkit.Chem import Descriptors, MolFromSmiles
from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from molml_mcp.infrastructure.logging import loggable



DESCRIPTOR_REGISTRY: dict[str, callable] = {
    name: fn for name, fn in Descriptors._descList
}

def list_rdkit_descriptors() -> list[dict]:
    """
    List all available RDKit 2D molecular descriptors.
    
    Returns a list of descriptor metadata including names and descriptions.
    Use this to discover which descriptors are available before computing them
    with calculate_descriptors().
    
    Returns:
        List of dicts with keys:
            - name: Descriptor function name (e.g. "MolWt", "TPSA")
            - explanation: Brief description from the descriptor's docstring
    
    Example return:
        [
            {"name": "MolWt", "explanation": "Molecular weight"},
            {"name": "TPSA", "explanation": "Topological polar surface area"},
            ...
        ]
    """
    out = []
    for name, fn in Descriptors._descList:
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        out.append({
            "descriptor name": name,
            "explanation": doc,
        })
    return out


@loggable
def calculate_descriptors(resource_id: str, smiles_column: str, descriptor_names: list[str]) -> dict:
    """
    Calculate RDKit molecular descriptors for molecules in a dataset.
    
    This function computes the specified RDKit 2D descriptors for all molecules 
    in the given SMILES column and adds each descriptor as a new column to the dataset.
    Invalid SMILES or calculation failures result in None values for that descriptor.
    
    Parameters
    ----------
    resource_id : str
        Identifier for the tabular dataset resource.
    smiles_column : str
        Name of the column containing SMILES strings.
    descriptor_names : list[str]
        List of RDKit descriptor names to calculate (e.g., ["MolWt", "TPSA", "LogP"]).
        Use list_rdkit_descriptors() to see all available descriptor names.
    
    Returns
    -------
    dict
        {
            "resource_id": str,              # identifier for the new dataset
            "n_rows": int,                   # number of rows
            "columns": list[str],            # all column names including new descriptors
            "descriptors_added": list[str],  # names of descriptors successfully added
            "n_failed": int,                 # number of molecules that failed calculation
            "preview": list[dict],           # first 5 rows as records
        }
    
    Raises
    ------
    ValueError
        If the SMILES column is not found or if invalid descriptor names are provided.
    
    Examples
    --------
    # Calculate molecular weight and TPSA
    calculate_descriptors(rid, "smiles", ["MolWt", "TPSA"])
    
    # Calculate multiple descriptors
    calculate_descriptors(rid, "canonical_smiles", ["MolWt", "TPSA", "MolLogP", "NumHDonors"])
    """
    import pandas as pd
    
    df = _load_resource(resource_id)
    
    # Validate inputs
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in dataset.")
    
    # Validate descriptor names
    invalid_descriptors = [name for name in descriptor_names if name not in DESCRIPTOR_REGISTRY]
    if invalid_descriptors:
        available = list(DESCRIPTOR_REGISTRY.keys())
        raise ValueError(
            f"Invalid descriptor names: {invalid_descriptors}. "
            f"Use list_rdkit_descriptors() to see all available descriptors. "
            f"Examples: {available}"
        )
    
    # Calculate descriptors for each molecule
    n_failed = {name: 0 for name in descriptor_names}
    descriptor_data = {name: [] for name in descriptor_names}
    
    for smiles in df[smiles_column]:
        mol = MolFromSmiles(smiles) if pd.notna(smiles) else None
        
        for desc_name in descriptor_names:
            try:
                if mol is not None:
                    value = DESCRIPTOR_REGISTRY[desc_name](mol)
                else:
                    value = None
                    n_failed[desc_name] += 1
            except Exception:
                value = None
                n_failed[desc_name]  += 1
            
            descriptor_data[desc_name].append(value)
    
    # Add descriptor columns to dataframe
    for desc_name, values in descriptor_data.items():
        df[desc_name] = values
    
    # Store the updated dataset
    new_resource_id = _store_resource(df, 'csv')
    
    return {
        "resource_id": new_resource_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "descriptors_added": descriptor_names,
        "n_failed": n_failed,
        "comment": "If molecules fails a descriptor calculation, a None value is assigned for that descriptor.",
        "preview": df.head(5).to_dict(orient="records"),
    }

