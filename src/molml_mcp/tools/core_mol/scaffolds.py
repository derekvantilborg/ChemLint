from collections import Counter
from molml_mcp.infrastructure.resources import _load_resource, _store_resource
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd


def _is_invalid_smiles(smi) -> bool:
    """Check if SMILES is None, NaN, or otherwise invalid."""
    if smi is None:
        return True
    # Check for pandas NA, numpy NaN, or float NaN
    if pd.isna(smi):
        return True
    # Check if it's not a string
    if not isinstance(smi, str):
        return True
    return False


def _get_scaffold(smiles: str, scaffold_type: str = 'bemis_murcko') -> tuple[str | None, str]:
    """ Get the molecular scaffold from a SMILES string. Supports three different scaffold types:
            `bemis_murcko`: RDKit implementation of the bemis-murcko scaffold; a scaffold of rings and linkers, retains
            some sidechains and ring-bonded substituents.
            `generic`: Bemis-Murcko scaffold where all atoms are carbons & bonds are single, i.e., a molecular skeleton.
            `cyclic_skeleton`: A molecular skeleton w/o any sidechains, only preserves ring structures and linkers.

    Examples:
        original molecule: 'CCCN(Cc1ccccn1)C(=O)c1cc(C)cc(OCCCON=C(N)N)c1'
        Bemis-Murcko scaffold: 'O=C(NCc1ccccn1)c1ccccc1'
        Generic RDKit: 'CC(CCC1CCCCC1)C1CCCCC1'
        Cyclic skeleton: 'C1CCC(CCCC2CCCCC2)CC1'

    :param smiles: SMILES string
    :param scaffold_type: 'bemis_murcko' (default), 'generic', 'cyclic_skeleton'
    :return: Tuple of (scaffold SMILES or None, comment)
    """
    all_scaffs = ['bemis_murcko', 'generic', 'cyclic_skeleton']
    if scaffold_type not in all_scaffs:
        return None, f"Failed: scaffold_type='{scaffold_type}' is not supported. Pick from: {all_scaffs}"

    # Handle None or NaN input (failed previous step or missing data)
    if _is_invalid_smiles(smiles):
        return None, "Skipped: Invalid SMILES string"

    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Failed: Invalid SMILES string"

    try:
        # designed to match atoms that are doubly bonded to another atom.
        PATT = Chem.MolFromSmarts("[$([D1]=[*])]")
        # replacement SMARTS (matches any atom)
        REPL = Chem.MolFromSmarts("[*]")

        Chem.RemoveStereochemistry(mol)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)

        if scaffold_type == 'bemis_murcko':
            pass  # scaffold already set

        elif scaffold_type == 'generic':
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)

        elif scaffold_type == 'cyclic_skeleton':
            scaffold = AllChem.ReplaceSubstructs(scaffold, PATT, REPL, replaceAll=True)[0]
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
            scaffold = MurckoScaffold.GetScaffoldForMol(scaffold)

        # Convert to SMILES and validate
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return None, "Failed: No scaffold found (molecule may lack ring systems)"
        
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        
        # Validate output SMILES
        if Chem.MolFromSmiles(scaffold_smiles) is None:
            return None, "Failed: Generated invalid scaffold SMILES"
        
        return scaffold_smiles, "Passed"
    
    except Exception as e:
        return None, f"Failed: {str(e)}"


def calculate_scaffolds(smiles: list[str], scaffold_type: str = 'bemis_murcko') -> tuple[list[str], list[str]]:
    """
    Calculate molecular scaffolds for a list of SMILES strings.
    
    This function extracts molecular scaffolds from a list of SMILES strings using 
    one of three scaffold types: Bemis-Murcko, generic (skeleton), or cyclic skeleton.
    Scaffolds represent the core ring systems and linkers of molecules, useful for 
    clustering, diversity analysis, and scaffold hopping in drug discovery.
    
    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings to process.
    scaffold_type : str, default='bemis_murcko'
        Type of scaffold to extract. Options:
        - 'bemis_murcko': Ring systems and linkers with some substituents retained
        - 'generic': Molecular skeleton (all atoms as carbon, all bonds single)
        - 'cyclic_skeleton': Skeleton without sidechains, only rings and linkers
    
    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - scaffolds : list[str]
            Scaffold SMILES strings. Length matches input list.
            Failed extractions or molecules without scaffolds return None.
        - comments : list[str]
            Comments for each SMILES indicating processing status. Length matches input list.
            - "Passed": Scaffold extraction successful
            - "Failed: <reason>": An error occurred or no scaffold exists
            - "Skipped: Invalid SMILES string": Input was invalid/NaN
    
    Examples
    --------
    # Extract Bemis-Murcko scaffolds
    smiles = ["c1ccccc1CCO", "CCO", "c1ccc(cc1)C(=O)O"]
    scaffolds, comments = calculate_scaffolds(smiles)
    # Returns scaffolds for molecules with rings, None for aliphatic molecules
    
    # Extract generic scaffolds
    scaffolds, comments = calculate_scaffolds(smiles, scaffold_type='generic')
    
    Notes
    -----
    - This function operates on a LIST of SMILES strings, not a dataset/dataframe
    - Molecules without ring systems will return None (no scaffold)
    - Scaffolds are automatically canonicalized by RDKit
    - All stereochemistry is removed from scaffolds
    - Output lists have the same length and order as input list
    
    See Also
    --------
    calculate_scaffolds_dataset : For dataset-level scaffold calculation
    _get_scaffold : Low-level helper function for single SMILES
    """
    scaffold_list, comment_list = [], []
    for smi in smiles:
        scaffold, comment = _get_scaffold(smi, scaffold_type)
        scaffold_list.append(scaffold)
        comment_list.append(comment)
    
    return scaffold_list, comment_list


def calculate_scaffolds_dataset(
    input_filename: str,
    column_name: str,
    project_manifest_path: str,
    output_filename: str,
    scaffold_type: str = 'bemis_murcko',
    explanation: str = "Calculate molecular scaffolds"
) -> dict:
    """
    Calculate molecular scaffolds for all SMILES strings in a dataset column.
    
    This function processes a tabular dataset by extracting molecular scaffolds from 
    SMILES strings in the specified column. It adds two new columns to the dataframe: 
    one containing the scaffold SMILES and another with comments logged during the 
    scaffold extraction process.
    
    Scaffolds represent the core ring systems and linkers of molecules, making them 
    valuable for:
    - Scaffold diversity analysis
    - Clustering by structural similarity
    - Scaffold hopping in medicinal chemistry
    - Series analysis in drug discovery
    
    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset (e.g., 'dataset_raw_A3F2B1D4').
    column_name : str
        Name of the column containing SMILES strings to process.
    project_manifest_path : str
        Path to the project manifest file for tracking this resource.
    output_filename : str
        Base filename for the stored resource without extension (e.g., 'dataset_scaffolds').
    scaffold_type : str, default='bemis_murcko'
        Type of scaffold to extract. Options:
        - 'bemis_murcko': Ring systems and linkers with some substituents retained
        - 'generic': Molecular skeleton (all atoms as carbon, all bonds single)
        - 'cyclic_skeleton': Skeleton without sidechains, only rings and linkers
    explanation : str
        Brief description of the scaffold calculation performed.
    
    Returns
    -------
    dict
        A dictionary containing:
        - output_filename : str
            Full filename with unique ID for the new resource with scaffold data.
        - n_rows : int
            Total number of rows in the dataset.
        - columns : list of str
            List of all column names in the updated dataset.
        - comments : dict
            Dictionary with counts of different comment types logged during 
            scaffold extraction (e.g., number of failed extractions, molecules without scaffolds).
        - n_scaffolds_found : int
            Number of molecules with successfully extracted scaffolds.
        - n_no_scaffold : int
            Number of molecules without scaffolds (typically aliphatic molecules).
        - scaffold_type : str
            Type of scaffold that was calculated.
        - preview : list of dict
            Preview of the first 5 rows of the updated dataset.
        - note : str
            Information about column naming and success/failure interpretation.
    
    Raises
    ------
    ValueError
        If the specified column_name is not found in the dataset.
    
    Examples
    --------
    # Calculate Bemis-Murcko scaffolds for a dataset
    result = calculate_scaffolds_dataset(
        input_filename='cleaned_molecules_A3F2B1D4.csv',
        column_name='smiles',
        project_manifest_path='/path/to/manifest.json',
        output_filename='molecules_with_scaffolds',
        scaffold_type='bemis_murcko'
    )
    
    # Calculate generic scaffolds (molecular skeletons)
    result = calculate_scaffolds_dataset(
        input_filename='cleaned_molecules_A3F2B1D4.csv',
        column_name='smiles',
        project_manifest_path='/path/to/manifest.json',
        output_filename='molecules_with_skeletons',
        scaffold_type='generic'
    )
    
    Notes
    -----
    The function adds two new columns to the dataset:
    - 'scaffold_{scaffold_type}': Contains the scaffold SMILES strings or None.
    - 'scaffold_comments': Contains any comments or warnings from the extraction process.
    
    Molecules without ring systems (e.g., aliphatic chains) will have None in the 
    scaffold column with comment "Failed: No scaffold found (molecule may lack ring systems)".
    
    See Also
    --------
    calculate_scaffolds : For processing a list of SMILES strings
    _get_scaffold : Low-level helper function for single SMILES
    """
    df = _load_resource(project_manifest_path, input_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    smiles_list = df[column_name].tolist()
    scaffolds, comments = calculate_scaffolds(smiles_list, scaffold_type)

    df[f'scaffold_{scaffold_type}'] = scaffolds
    df['scaffold_comments'] = comments

    output_filename = _store_resource(df, project_manifest_path, output_filename, explanation, 'csv')

    # Count successes and failures
    n_scaffolds_found = sum(1 for s in scaffolds if s is not None)
    n_no_scaffold = sum(1 for c in comments if 'No scaffold found' in c)

    return {
        "output_filename": output_filename,
        "n_rows": len(df),
        "columns": list(df.columns),
        "comments": dict(Counter(comments)),
        "n_scaffolds_found": n_scaffolds_found,
        "n_no_scaffold": n_no_scaffold,
        "scaffold_type": scaffold_type,
        "preview": df.head(5).to_dict(orient="records"),
        "note": f"Scaffold column: 'scaffold_{scaffold_type}'. Successful extraction is marked by 'Passed' in scaffold_comments, failure is marked by 'Failed: <reason>'.",
    }


def get_all_scaffold_tools():
    """Return a list of all molecular scaffold tools."""
    return [
        calculate_scaffolds,
        calculate_scaffolds_dataset,
    ]
