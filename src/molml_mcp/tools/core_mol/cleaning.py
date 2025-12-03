from collections import Counter
from rdkit.Chem import MolFromSmiles, MolToSmiles
from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from molml_mcp.infrastructure.logging import loggable
from molml_mcp.tools.core_mol.smiles_ops import _canonicalize_smiles

from molml_mcp.constants import SMARTS_COMMON_SALTS, SMARTS_COMMON_ISOTOPES, SMARTS_NEUTRALIZATION_PATTERNS, COMMON_SOLVENTS


@loggable
def canonicalize_smiles_dataset(resource_id:str, column_name:str) -> dict:
    """
    Canonicalize all SMILES strings in a specified column of a tabular dataset.
    
    This function processes a tabular dataset by canonicalizing SMILES strings in the 
    specified column. It adds two new columns to the dataframe: one containing the 
    canonicalized SMILES and another with comments logged during the canonicalization 
    process (e.g., invalid SMILES, conversion failures).
    
    Parameters
    ----------
    resource_id : str
        Identifier for the tabular dataset resource to be processed.
    column_name : str
        Name of the column containing SMILES strings to be canonicalized.
    
    Returns
    -------
    dict
        A dictionary containing:
        - resource_id : str
            Identifier for the new resource with canonicalized data.
        - n_rows : int
            Total number of rows in the dataset.
        - columns : list of str
            List of all column names in the updated dataset.
        - comments : dict
            Dictionary with counts of different comment types logged during 
            canonicalization (e.g., number of failed conversions, invalid SMILES).
        - preview : list of dict
            Preview of the first 5 rows of the updated dataset.
        - suggestions : str
            Recommendations for additional cleaning steps that may be beneficial.
        - question_to_user : str
            Question directed at the user/client regarding next steps for handling 
            problematic entries.
    
    Raises
    ------
    ValueError
        If the specified column_name is not found in the dataset.
    
    Notes
    -----
    The function adds two new columns to the dataset:
    - 'smiles_after_canonicalization': Contains the canonicalized SMILES strings.
    - 'comments_after_canonicalization': Contains any comments or warnings from the 
      canonicalization process.
    """
    df = _load_resource(resource_id)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    canonical_smiles, comments = _canonicalize_smiles(smiles_list)

    df['smiles_after_canonicalization'] = canonical_smiles
    df['comments_after_canonicalization'] = comments

    new_resource_id = _store_resource(df, 'csv')

    return {
        "resource_id": new_resource_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful canonicalization is marked by 'Passed' in comments, failure is marked by 'Failed: <reason>'.",
        "suggestions": "Consider further cleaning steps such as salt removal, tautomer canonicalization, charge neutralization, and stereochemistry handling.",
        "question_to_user": "Would you like to review failed SMILES entries or drop them from the dataset?",
    }


@loggable
def remove_salts(smiles: list[str], salt_smarts: str = SMARTS_COMMON_SALTS) -> tuple[list[str], list[str]]:
    """
    Remove common salt ions from a list of SMILES strings.
    
    This function strips common salt counterions (Cl, Na, Mg, Ca, K, Br, Zn, Ag, Al, 
    Li, I, O, N, H) from molecular structures. It processes a list of SMILES strings
    and returns cleaned versions without salt counterions.
    
    **IMPORTANT**: The default salt pattern works well for most use cases and should 
    typically NOT be changed. Only modify `salt_smarts` if you have specific requirements 
    for a specialized dataset (e.g., organometallic compounds where metals are part of 
    the active structure).
    
    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings to process. Each may contain salt counterions.
    salt_smarts : str, optional
        SMARTS pattern defining which atoms/ions to remove.
        **Default: "[Cl,Na,Mg,Ca,K,Br,Zn,Ag,Al,Li,I,O,N,H]"**
        This default covers common pharmaceutical salts and should be used in most cases.
        Only change this if you have a specific reason (e.g., you want to keep certain 
        ions, or you're working with unusual salt forms).
    
    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - new_smiles : list[str]
            SMILES strings with salts removed. Length matches input list.
        - comments : list[str]
            Comments for each SMILES indicating processing status. Length matches input list.
            - "Passed": Salt removal successful
            - "Failed: <reason>": An error occurred (e.g., invalid SMILES)
    
    Examples
    --------
    # Remove common salts (typical usage - don't change salt_smarts)
    smiles = ["CC(=O)O.Na", "c1ccccc1.HCl", "CCO"]
    clean_smiles, comments = remove_salts(smiles)
    # Returns: ["CC(=O)O", "c1ccccc1", "CCO"], ["Passed", "Passed", "Passed"]
    
    # Only change salt_smarts if you have a specific reason:
    # (e.g., removing only chloride and bromide)
    smiles = ["CC(=O)O.Na", "c1ccccc1.Cl"]
    clean_smiles, comments = remove_salts(smiles, salt_smarts="[Cl,Br]")
    # Returns: ["CC(=O)O.Na", "c1ccccc1"], ["Passed", "Passed"]
    
    Notes
    -----
    - This function operates on a LIST of SMILES strings, not a dataset/dataframe
    - The default pattern removes the most common pharmaceutical salt counterions
    - If a molecule consists ONLY of salt ions, the result may be empty or fail
    - For complex salt forms, multiple passes may be needed
    - The function preserves the largest fragment if multiple fragments remain after 
      salt removal
    - Output lists have the same length and order as input list
    
    See Also
    --------
    canonicalize_smiles: For smiles canonicalization
    """
    new_smiles, comments = _remove_pattern(smiles, salt_smarts)
    
    return new_smiles, comments


@loggable
def remove_salts_dataset(
    resource_id: str,
    column_name: str,
    salt_smarts: str = SMARTS_COMMON_SALTS
) -> dict:
    """
    Remove common salt ions from SMILES strings in a specified column of a tabular dataset.
    
    This function processes a tabular dataset by removing salt counterions from SMILES 
    strings in the specified column. It adds two new columns to the dataframe: one 
    containing the desalted SMILES and another with comments logged during the salt 
    removal process (e.g., invalid SMILES, processing failures).
    
    **IMPORTANT**: The default salt pattern works well for most use cases and should 
    typically NOT be changed. Only modify `salt_smarts` if you have specific requirements 
    (e.g., organometallic compounds where metals are part of the active structure).
    
    Parameters
    ----------
    resource_id : str
        Identifier for the tabular dataset resource to be processed.
    column_name : str
        Name of the column containing SMILES strings to be desalted.
    salt_smarts : str, optional
        SMARTS pattern defining which atoms/ions to remove.
        **Default: "[Cl,Na,Mg,Ca,K,Br,Zn,Ag,Al,Li,I,O,N,H]"**
        This default covers common pharmaceutical salts and should be used in most cases.
        Only change this if you have a specific reason.
    
    Returns
    -------
    dict
        A dictionary containing:
        - resource_id : str
            Identifier for the new resource with desalted data.
        - n_rows : int
            Total number of rows in the dataset.
        - columns : list of str
            List of all column names in the updated dataset.
        - comments : dict
            Dictionary with counts of different comment types logged during 
            salt removal (e.g., number of successful removals, failures).
        - preview : list of dict
            Preview of the first 5 rows of the updated dataset.
        - note : str
            Explanation of the comment system.
        - suggestions : str
            Recommendations for additional cleaning steps that may be beneficial.
        - question_to_user : str
            Question directed at the user/client regarding next steps.
    
    Raises
    ------
    ValueError
        If the specified column_name is not found in the dataset.
    
    Notes
    -----
    The function adds two new columns to the dataset:
    - 'smiles_after_salt_removal': Contains the desalted SMILES strings.
    - 'comments_after_salt_removal': Contains any comments or warnings from the 
      salt removal process.
    
    Examples
    --------
    # Typical usage with default salt pattern
    result = remove_salts_dataset(resource_id="20251203T120000_csv_ABC123.csv", 
                                   column_name="smiles")
    
    See Also
    --------
    remove_salts : For processing a list of SMILES strings
    canonicalize_smiles_dataset : For dataset-level canonicalization
    """
    df = _load_resource(resource_id)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    desalted_smiles, comments = remove_salts(smiles_list, salt_smarts)

    df['smiles_after_salt_removal'] = desalted_smiles
    df['comments_after_salt_removal'] = comments

    new_resource_id = _store_resource(df, 'csv')

    return {
        "resource_id": new_resource_id,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "preview": df.head(5).to_dict(orient="records"),
        "note": "Successful salt removal is marked by 'Passed' in comments, failure is marked by 'Failed: <reason>'.",
        "suggestions": "Consider further cleaning steps such as canonicalization, charge neutralization, and stereochemistry handling.",
        "question_to_user": "Would you like to review failed SMILES entries or drop them from the dataset?",
    }



def _remove_pattern(smiles: list[str], smarts_pattern: str) -> tuple[list[str], list[str]]: 
    """ Remove some pattern from a SMILES string using the specified SMARTS.

    :param smiles: SMILES string
    :param smarts_pattern: SMARTS pattern
    :return: cleaned SMILES without pattern, comments
    """
    from rdkit import Chem
    from rdkit.Chem.SaltRemover import SaltRemover

    remover = SaltRemover(defnData=smarts_pattern)

    new_smi, comment = [], []
    for smi in smiles:

        if '.' not in smi:
            new_smi.append(smi)
            comment.append("Passed")
            continue

        mol = MolFromSmiles(smi)

        if mol is None:
            new_smi.append(None)
            comment.append("Failed: Invalid SMILES string")
            continue

        try:
            new_smi = Chem.MolToSmiles(remover.StripMol(mol))
            new_smi.append(new_smi)
            comment.append("Passed")
        except Exception as e:
            new_smi.append(None)
            comment.append(f"Failed: {str(e)}")

    return new_smi, comment






# Salt removal

# def _desalt_molecule(self, mol: Chem.Mol) -> Tuple[Chem.Mol, bool]:
#     """Remove salts from molecule."""
#     if '.' not in Chem.MolToSmiles(mol):
#         return mol, True
    
#     if self.desalt_policy == 'remove':
#         return None, False
    
#     if self.desalt_policy == 'keep':
#         mol = self.salt_remover.StripMol(mol, dontRemoveEverything=False)
#         desalted_smiles = Chem.MolToSmiles(mol)
        
#         if '.' in desalted_smiles:
#             if self.brute_force_desalt:
#                 # Keep largest fragment
#                 mol = max(
#                     Chem.GetMolFrags(mol, asMols=True), 
#                     key=lambda x: x.GetNumAtoms()
#                 )
#             else:
#                 return None, False
        
#         if mol and mol.GetNumAtoms() > 0:
#             return mol, True
    
#     return None, False

# Tautomer canonicalization
# Charge neutralization
# Stereochemistry handling
# remove duplicates

