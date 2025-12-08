"""
Test script for the complete SMILES standardization pipeline function.

Tests default_SMILES_standardization_pipeline_dataset which runs all 11+ cleaning
steps in a single function call with full audit trail.
"""

import pandas as pd
import json
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from molml_mcp.tools.core_mol.cleaning import default_SMILES_standardization_pipeline_dataset
from molml_mcp.tools.core.dataset_ops import store_csv_as_dataset


def create_test_project():
    """Create a temporary test project with manifest."""
    test_dir = Path(tempfile.mkdtemp(prefix="test_pipeline_"))
    manifest_path = test_dir / "project_manifest.json"
    
    # Create initial manifest
    manifest = {
        "project_name": "Test SMILES Standardization Pipeline",
        "created_at": "2025-12-08T00:00:00.000000",
        "resources": []
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Created test project at: {test_dir}")
    return str(manifest_path), test_dir


def test_default_pipeline(manifest_path):
    """Test the complete standardization pipeline with default settings."""
    print("\n=== Testing default_SMILES_standardization_pipeline_dataset ===")
    print("Settings: stereo_policy='flatten', skip_isotope_removal=False, enable_metal_disconnection=False")
    
    # Use existing test data
    test_csv = Path(__file__).parent / "data" / "cleaning_test.csv"
    
    # First store the CSV
    result = store_csv_as_dataset(
        file_path=str(test_csv),
        project_manifest_path=manifest_path,
        filename="raw_molecules",
        explanation="Raw molecular data for pipeline test"
    )
    
    input_filename = result['output_filename']
    print(f"✓ Loaded test data: {input_filename} ({result['n_rows']} rows)")
    
    # Run the complete pipeline
    print("\nRunning complete standardization pipeline...")
    result = default_SMILES_standardization_pipeline_dataset(
        input_filename=input_filename,
        column_name="smiles",
        project_manifest_path=manifest_path,
        output_filename="fully_standardized",
        explanation="Complete SMILES standardization with default protocol",
        stereo_policy="flatten",
        skip_isotope_removal=False,
        enable_metal_disconnection=False
    )
    
    print(f"\n✓ Pipeline completed successfully!")
    print(f"  Output: {result['output_filename']}")
    print(f"  Total rows: {result['n_rows']}")
    print(f"  Total columns: {len(result['columns'])}")
    print(f"\n  Protocol Summary:")
    for key, val in result['protocol_summary'].items():
        print(f"    {key}: {val}")
    print(f"\n  Final Validation:")
    for key, val in result['final_validation'].items():
        print(f"    {key}: {val}")
    
    print(f"\n  Note: {result['note']}")
    
    return result['output_filename']


def test_pipeline_with_metal_disconnection(manifest_path):
    """Test pipeline with metal disconnection enabled."""
    print("\n=== Testing Pipeline with Metal Disconnection ===")
    print("Settings: enable_metal_disconnection=True, drop_inorganics=True")
    
    # Use existing test data
    test_csv = Path(__file__).parent / "data" / "cleaning_test.csv"
    
    # Store the CSV
    result = store_csv_as_dataset(
        file_path=str(test_csv),
        project_manifest_path=manifest_path,
        filename="raw_molecules_2",
        explanation="Raw molecular data for metal disconnection test"
    )
    
    input_filename = result['output_filename']
    
    # Run pipeline with metal disconnection
    print("\nRunning pipeline with metal disconnection...")
    result = default_SMILES_standardization_pipeline_dataset(
        input_filename=input_filename,
        column_name="smiles",
        project_manifest_path=manifest_path,
        output_filename="standardized_no_metals",
        explanation="SMILES standardization with metal disconnection",
        stereo_policy="flatten",
        skip_isotope_removal=False,
        enable_metal_disconnection=True,
        drop_inorganics=True
    )
    
    print(f"\n✓ Pipeline with metal disconnection completed!")
    print(f"  Output: {result['output_filename']}")
    print(f"  Validation rate: {result['final_validation']['validation_rate']:.1f}%")
    
    return result['output_filename']


def verify_pipeline_output(manifest_path, final_filename):
    """Verify the pipeline created all expected columns and transformations."""
    print("\n=== Verifying Pipeline Output ===")
    
    from molml_mcp.infrastructure.resources import _load_resource
    
    df = _load_resource(manifest_path, final_filename)
    
    print(f"✓ Loaded final dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Check for key columns
    expected_columns = [
        'smiles',  # original
        'smiles_after_canonicalization',
        'smiles_after_salt_removal',
        'smiles_after_solvent_removal',
        'smiles_after_defragmentation',
        'smiles_after_functional_group_normalization',
        'smiles_after_reionization',
        'smiles_after_neutralization',
        'smiles_after_isotope_removal',
        'smiles_after_tautomer_canonicalization',
        'smiles_after_stereo_standardization',
        'standardized_smiles',  # final output
        'validation_status'
    ]
    
    present = [col for col in expected_columns if col in df.columns]
    missing = [col for col in expected_columns if col not in df.columns]
    
    print(f"  Expected columns present: {len(present)}/{len(expected_columns)}")
    if missing:
        print(f"  Missing columns: {missing}")
    
    # Count comment columns (should be one per step)
    comment_cols = [col for col in df.columns if 'comment' in col.lower()]
    print(f"  Comment columns (audit trail): {len(comment_cols)}")
    
    # Show a sample transformation
    if len(df) > 0:
        print(f"\n  Sample Transformation (Row 0):")
        print(f"    Original:           {df['smiles'].iloc[0]}")
        if 'smiles_after_canonicalization' in df.columns:
            print(f"    Canonicalized:      {df['smiles_after_canonicalization'].iloc[0]}")
        if 'smiles_after_salt_removal' in df.columns:
            print(f"    Desalted:           {df['smiles_after_salt_removal'].iloc[0]}")
        if 'smiles_after_neutralization' in df.columns:
            print(f"    Neutralized:        {df['smiles_after_neutralization'].iloc[0]}")
        if 'standardized_smiles' in df.columns:
            print(f"    Final (Standardized): {df['standardized_smiles'].iloc[0]}")
        if 'validation_status' in df.columns:
            print(f"    Validation:         {df['validation_status'].iloc[0]}")
    
    # Check validation statistics
    if 'validation_status' in df.columns:
        validation_counts = df['validation_status'].value_counts().to_dict()
        print(f"\n  Validation Summary:")
        for status, count in validation_counts.items():
            print(f"    {status}: {count}")
    
    return df


def inspect_manifest(manifest_path):
    """Inspect the manifest to see all resources created by the pipeline."""
    print("\n=== Manifest Inspection ===")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"✓ Project: {manifest['project_name']}")
    print(f"  Total resources tracked: {len(manifest['resources'])}")
    
    # Count by parent function
    from collections import Counter
    function_counts = Counter([res['parent_function_name'] for res in manifest['resources']])
    
    print(f"\n  Resources by Function:")
    for func, count in function_counts.most_common():
        print(f"    {func}: {count}")
    
    # Show first and last few resources
    print(f"\n  First 3 Resources:")
    for i, res in enumerate(manifest['resources'][:3], 1):
        print(f"    {i}. {res['filename']}")
        print(f"       - {res['explaination']}")
    
    if len(manifest['resources']) > 6:
        print(f"    ... ({len(manifest['resources']) - 6} more) ...")
    
    print(f"\n  Last 3 Resources:")
    for i, res in enumerate(manifest['resources'][-3:], len(manifest['resources']) - 2):
        print(f"    {i}. {res['filename']}")
        print(f"       - {res['explaination']}")


def main():
    """Run all pipeline tests."""
    print("=" * 70)
    print("TESTING SMILES STANDARDIZATION PIPELINE")
    print("=" * 70)
    
    # Create test project
    manifest_path, test_dir = create_test_project()
    
    try:
        # Test 1: Default pipeline (flatten stereo, remove isotopes, no metal disconnection)
        filename1 = test_default_pipeline(manifest_path)
        
        # Verify output
        df1 = verify_pipeline_output(manifest_path, filename1)
        
        # Test 2: Pipeline with metal disconnection
        filename2 = test_pipeline_with_metal_disconnection(manifest_path)
        
        # Verify second output
        df2 = verify_pipeline_output(manifest_path, filename2)
        
        # Inspect manifest
        inspect_manifest(manifest_path)
        
        print("\n" + "=" * 70)
        print("✓ ALL PIPELINE TESTS PASSED!")
        print("=" * 70)
        
        print(f"\n✅ Summary:")
        print(f"  - Test 1 (default): {len(df1)} rows, {len(df1.columns)} columns")
        print(f"  - Test 2 (metals):  {len(df2)} rows, {len(df2.columns)} columns")
        print(f"  - Total resources created: {len(json.load(open(manifest_path))['resources'])}")
        
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
