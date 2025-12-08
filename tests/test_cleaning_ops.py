"""
Test script for molecular cleaning functions with new resource management system.

Tests canonicalize_smiles_dataset, remove_salts_dataset, remove_common_solvents_dataset,
and other cleaning operations to verify they work with the new manifest-based system.
"""

import pandas as pd
import json
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from molml_mcp.tools.core_mol.cleaning import (
    canonicalize_smiles_dataset,
    remove_salts_dataset,
    remove_common_solvents_dataset,
    defragment_smiles_dataset,
    neutralize_smiles_dataset
)
from molml_mcp.tools.core.dataset_ops import store_csv_as_dataset


def create_test_project():
    """Create a temporary test project with manifest."""
    test_dir = Path(tempfile.mkdtemp(prefix="test_cleaning_ops_"))
    manifest_path = test_dir / "project_manifest.json"
    
    # Create initial manifest
    manifest = {
        "project_name": "Test Molecular Cleaning Project",
        "created_at": "2025-12-08T00:00:00.000000",
        "resources": []
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Created test project at: {test_dir}")
    return str(manifest_path), test_dir


def test_canonicalize_smiles_dataset(manifest_path):
    """Test SMILES canonicalization with dataset."""
    print("\n=== Testing canonicalize_smiles_dataset ===")
    
    # Use existing test data
    test_csv = Path(__file__).parent / "data" / "cleaning_test.csv"
    
    # First store the CSV
    result = store_csv_as_dataset(
        file_path=str(test_csv),
        project_manifest_path=manifest_path,
        filename="raw_molecules",
        explanation="Raw molecular data for cleaning tests"
    )
    
    input_filename = result['output_filename']
    print(f"✓ Loaded test data: {input_filename} ({result['n_rows']} rows)")
    
    # Canonicalize SMILES
    result = canonicalize_smiles_dataset(
        input_filename=input_filename,
        column_name="smiles",
        project_manifest_path=manifest_path,
        output_filename="step1_canonicalized",
        explanation="SMILES after canonicalization"
    )
    
    print(f"✓ Canonicalized SMILES")
    print(f"  Output: {result['output_filename']}")
    print(f"  Rows: {result['n_rows']}")
    print(f"  New columns: {[c for c in result['columns'] if 'canonicalization' in c]}")
    print(f"  Comment counts: {result['comments']}")
    print(f"  First row preview: {result['preview'][0] if result['preview'] else 'None'}")
    
    return result['output_filename']


def test_remove_salts_dataset(manifest_path, input_filename):
    """Test salt removal from molecules."""
    print("\n=== Testing remove_salts_dataset ===")
    
    result = remove_salts_dataset(
        input_filename=input_filename,
        column_name="smiles_after_canonicalization",
        project_manifest_path=manifest_path,
        output_filename="step2_desalted",
        explanation="SMILES after salt removal"
    )
    
    print(f"✓ Removed salts")
    print(f"  Output: {result['output_filename']}")
    print(f"  Rows: {result['n_rows']}")
    print(f"  New columns: {[c for c in result['columns'] if 'salt' in c]}")
    print(f"  Comment counts: {result['comments']}")
    
    # Check if any salts were actually removed
    if 'Passed' in result['comments']:
        print(f"  Processed {result['comments']['Passed']} molecules successfully")
    
    return result['output_filename']


def test_remove_solvents_dataset(manifest_path, input_filename):
    """Test solvent removal from molecules."""
    print("\n=== Testing remove_common_solvents_dataset ===")
    
    result = remove_common_solvents_dataset(
        input_filename=input_filename,
        column_name="smiles_after_salt_removal",
        project_manifest_path=manifest_path,
        output_filename="step3_no_solvents",
        explanation="SMILES after solvent removal"
    )
    
    print(f"✓ Removed common solvents")
    print(f"  Output: {result['output_filename']}")
    print(f"  Rows: {result['n_rows']}")
    print(f"  New columns: {[c for c in result['columns'] if 'solvent' in c]}")
    print(f"  Comment counts: {result['comments']}")
    
    return result['output_filename']


def test_defragment_smiles_dataset(manifest_path, input_filename):
    """Test defragmentation (keeping largest fragment)."""
    print("\n=== Testing defragment_smiles_dataset ===")
    
    result = defragment_smiles_dataset(
        input_filename=input_filename,
        column_name="smiles_after_solvent_removal",
        project_manifest_path=manifest_path,
        output_filename="step4_defragmented",
        explanation="SMILES after defragmentation (largest fragment only)"
    )
    
    print(f"✓ Defragmented molecules")
    print(f"  Output: {result['output_filename']}")
    print(f"  Rows: {result['n_rows']}")
    print(f"  New columns: {[c for c in result['columns'] if 'defragment' in c]}")
    print(f"  Comment counts: {result['comments']}")
    
    return result['output_filename']


def test_neutralize_charges_dataset(manifest_path, input_filename):
    """Test charge neutralization."""
    print("\n=== Testing neutralize_smiles_dataset ===")
    
    result = neutralize_smiles_dataset(
        input_filename=input_filename,
        column_name="smiles_after_defragmentation",
        project_manifest_path=manifest_path,
        output_filename="step5_neutralized",
        explanation="SMILES after charge neutralization"
    )
    
    print(f"✓ Neutralized charges")
    print(f"  Output: {result['output_filename']}")
    print(f"  Rows: {result['n_rows']}")
    print(f"  New columns: {[c for c in result['columns'] if 'neutraliz' in c]}")
    print(f"  Comment counts: {result['comments']}")
    
    return result['output_filename']


def verify_data_integrity(manifest_path, final_filename):
    """Verify the final cleaned data makes sense."""
    print("\n=== Verifying Data Integrity ===")
    
    from molml_mcp.infrastructure.resources import _load_resource
    
    df = _load_resource(manifest_path, final_filename)
    
    print(f"✓ Loaded final dataset: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    
    # Check for expected columns from each step
    expected_columns = [
        'smiles',
        'smiles_after_canonicalization',
        'smiles_after_salt_removal',
        'smiles_after_solvent_removal',
        'smiles_after_defragmentation',
        'smiles_after_neutralization'
    ]
    
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        print(f"  ⚠ Missing expected columns: {missing}")
    else:
        print(f"  ✓ All expected columns present")
    
    # Show a sample row to verify the cleaning pipeline
    if len(df) > 0:
        print(f"\n  Sample transformation:")
        print(f"    Original:       {df['smiles'].iloc[0]}")
        print(f"    Canonicalized:  {df['smiles_after_canonicalization'].iloc[0]}")
        print(f"    Desalted:       {df['smiles_after_salt_removal'].iloc[0]}")
        print(f"    Defragmented:   {df['smiles_after_defragmentation'].iloc[0]}")
        print(f"    Neutralized:    {df['smiles_after_neutralization'].iloc[0]}")


def inspect_manifest(manifest_path):
    """Inspect the final manifest to see all tracked resources."""
    print("\n=== Final Manifest Inspection ===")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"✓ Project: {manifest['project_name']}")
    print(f"  Total resources tracked: {len(manifest['resources'])}")
    print("\n  Cleaning Pipeline Resources:")
    for i, res in enumerate(manifest['resources'], 1):
        print(f"    {i}. {res['filename']}")
        print(f"       - Type: {res['type_tag']}")
        print(f"       - Explanation: {res['explaination']}")
        print(f"       - Created by: {res['parent_function_name']}")


def main():
    """Run all molecular cleaning tests."""
    print("=" * 70)
    print("TESTING MOLECULAR CLEANING WITH NEW RESOURCE MANAGEMENT")
    print("=" * 70)
    
    # Create test project
    manifest_path, test_dir = create_test_project()
    
    try:
        # Test 1: Canonicalize SMILES
        filename1 = test_canonicalize_smiles_dataset(manifest_path)
        
        # Test 2: Remove salts
        filename2 = test_remove_salts_dataset(manifest_path, filename1)
        
        # Test 3: Remove solvents
        filename3 = test_remove_solvents_dataset(manifest_path, filename2)
        
        # Test 4: Defragment (keep largest fragment)
        filename4 = test_defragment_smiles_dataset(manifest_path, filename3)
        
        # Test 5: Neutralize charges
        filename5 = test_neutralize_charges_dataset(manifest_path, filename4)
        
        # Verify final data
        verify_data_integrity(manifest_path, filename5)
        
        # Inspect manifest
        inspect_manifest(manifest_path)
        
        print("\n" + "=" * 70)
        print("✓ ALL CLEANING TESTS PASSED!")
        print("=" * 70)
        
        # Clean up temporary directory
        print(f"\nCleaning up test project at: {test_dir}")
        shutil.rmtree(test_dir)
        print("✓ Cleanup complete")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nTest project location: {test_dir}")
        print("(Directory kept for debugging)")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
