"""
Test script for dataset_ops functions with new resource management system.

This tests the new _store_resource and _load_resource implementation that uses
project_manifest_path and filenames instead of resource_ids.
"""

import pandas as pd
import json
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from molml_mcp.tools.core.dataset_ops import (
    store_csv_as_dataset,
    store_csv_as_dataset_from_text,
    get_dataset_head,
    get_dataset_full,
    get_dataset_summary,
    inspect_dataset_rows,
    drop_from_dataset,
    keep_from_dataset,
    deduplicate_molecules_dataset,
    drop_duplicate_rows,
    drop_empty_rows
)


def create_test_project():
    """Create a temporary test project with manifest."""
    test_dir = Path(tempfile.mkdtemp(prefix="test_dataset_ops_"))
    manifest_path = test_dir / "project_manifest.json"
    
    # Create initial manifest
    manifest = {
        "project_name": "Test Dataset Ops Project",
        "created_at": "2025-12-08T00:00:00.000000",
        "resources": []
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Created test project at: {test_dir}")
    return str(manifest_path), test_dir


def test_store_csv_as_dataset(manifest_path):
    """Test storing a CSV file as a dataset."""
    print("\n=== Testing store_csv_as_dataset ===")
    
    # Use existing test data
    test_csv = Path(__file__).parent / "data" / "cleaning_test.csv"
    
    result = store_csv_as_dataset(
        file_path=str(test_csv),
        project_manifest_path=manifest_path,
        filename="test_initial_load",
        explanation="Initial test dataset load"
    )
    
    print(f"✓ Stored dataset: {result['output_filename']}")
    print(f"  Rows: {result['n_rows']}")
    print(f"  Columns: {result['columns']}")
    print(f"  Preview (first 2 rows): {result['preview'][:2]}")
    
    return result['output_filename']


def test_store_csv_from_text(manifest_path):
    """Test storing CSV from text content."""
    print("\n=== Testing store_csv_as_dataset_from_text ===")
    
    csv_content = """smiles,label,name
CC(=O)O,1,acetic_acid
CCO,0,ethanol
c1ccccc1,1,benzene"""
    
    result = store_csv_as_dataset_from_text(
        csv_content=csv_content,
        project_manifest_path=manifest_path,
        filename="test_from_text",
        explanation="Dataset created from text content"
    )
    
    print(f"✓ Stored dataset from text: {result['output_filename']}")
    print(f"  Rows: {result['n_rows']}")
    print(f"  Columns: {result['columns']}")
    
    return result['output_filename']


def test_get_dataset_head(manifest_path, filename):
    """Test getting first n rows of dataset."""
    print("\n=== Testing get_dataset_head ===")
    
    result = get_dataset_head(
        project_manifest_path=manifest_path,
        input_filename=filename,
        n_rows=5
    )
    
    print(f"✓ Retrieved head of dataset")
    print(f"  Returned {result['n_rows_returned']} of {result['n_rows_total']} total rows")
    print(f"  First row: {result['rows'][0] if result['rows'] else 'None'}")
    
    return result


def test_get_dataset_full(manifest_path, filename):
    """Test getting entire dataset."""
    print("\n=== Testing get_dataset_full ===")
    
    result = get_dataset_full(
        project_manifest_path=manifest_path,
        input_filename=filename,
        max_rows=1000
    )
    
    print(f"✓ Retrieved full dataset")
    print(f"  Total rows: {result['n_rows_total']}")
    print(f"  Returned: {result['n_rows_returned']}")
    print(f"  Truncated: {result['truncated']}")
    
    return result


def test_get_dataset_summary(manifest_path, filename):
    """Test dataset summary statistics."""
    print("\n=== Testing get_dataset_summary ===")
    
    result = get_dataset_summary(
        project_manifest_path=manifest_path,
        input_filename=filename,
        columns=["label"]  # Test with specific column
    )
    
    print(f"✓ Retrieved dataset summary")
    print(f"  Total columns in dataset: {result['n_columns']}")
    print(f"  Columns summarized: {result['n_columns_summarized']}")
    print(f"  Label column summary: {result['column_summaries'].get('label', 'Not found')}")
    
    return result


def test_inspect_dataset_rows(manifest_path, filename):
    """Test inspecting specific rows with filter conditions."""
    print("\n=== Testing inspect_dataset_rows ===")
    
    # Test with filter condition
    result = inspect_dataset_rows(
        project_manifest_path=manifest_path,
        input_filename=filename,
        filter_condition="label == 1",
        max_rows=5
    )
    
    print(f"✓ Filtered rows with condition 'label == 1'")
    print(f"  Matched {result['n_rows_returned']} rows")
    print(f"  First match: {result['rows'][0] if result['rows'] else 'None'}")
    
    # Test with row indices
    result2 = inspect_dataset_rows(
        project_manifest_path=manifest_path,
        input_filename=filename,
        row_indices=[0, 1, 2]
    )
    
    print(f"✓ Retrieved specific row indices [0, 1, 2]")
    print(f"  Rows returned: {result2['n_rows_returned']}")
    
    return result


def test_drop_from_dataset(manifest_path, filename):
    """Test dropping rows based on condition."""
    print("\n=== Testing drop_from_dataset ===")
    
    result = drop_from_dataset(
        input_filename=filename,
        column_name="label",
        condition="0",  # Drop rows where label == 0
        project_manifest_path=manifest_path,
        output_filename="test_after_drop",
        explanation="Dropped rows with label 0"
    )
    
    print(f"✓ Dropped rows where label == 0")
    print(f"  New dataset: {result['output_filename']}")
    print(f"  Remaining rows: {result['n_rows']}")
    
    return result['output_filename']


def test_keep_from_dataset(manifest_path, filename):
    """Test keeping only rows matching condition."""
    print("\n=== Testing keep_from_dataset ===")
    
    result = keep_from_dataset(
        input_filename=filename,
        column_name="label",
        condition="1",  # Keep only rows where label == 1
        project_manifest_path=manifest_path,
        output_filename="test_after_keep",
        explanation="Kept only rows with label 1"
    )
    
    print(f"✓ Kept only rows where label == 1")
    print(f"  New dataset: {result['output_filename']}")
    print(f"  Remaining rows: {result['n_rows']}")
    
    return result['output_filename']


def test_deduplicate_molecules(manifest_path):
    """Test deduplication by molecule identifier."""
    print("\n=== Testing deduplicate_molecules_dataset ===")
    
    # Create a dataset with duplicates
    csv_content = """smiles,mol_id,label
CC(=O)O,mol_1,1
CCO,mol_2,0
CC(=O)O,mol_1,1
c1ccccc1,mol_3,1
CCO,mol_2,0"""
    
    # Store it first
    result = store_csv_as_dataset_from_text(
        csv_content=csv_content,
        project_manifest_path=manifest_path,
        filename="test_with_duplicates",
        explanation="Dataset with duplicate molecules"
    )
    
    input_filename = result['output_filename']
    
    # Deduplicate
    result = deduplicate_molecules_dataset(
        input_filename=input_filename,
        molecule_id_column="mol_id",
        project_manifest_path=manifest_path,
        output_filename="test_deduplicated",
        explanation="Removed duplicate molecules"
    )
    
    print(f"✓ Deduplicated dataset")
    print(f"  Rows before: {result['n_rows_before']}")
    print(f"  Rows after: {result['n_rows_after']}")
    print(f"  New dataset: {result['output_filename']}")
    
    return result['output_filename']


def test_drop_duplicate_rows(manifest_path):
    """Test dropping completely duplicate rows."""
    print("\n=== Testing drop_duplicate_rows ===")
    
    # Create a dataset with duplicate rows
    csv_content = """smiles,label
CC(=O)O,1
CCO,0
CC(=O)O,1
c1ccccc1,1"""
    
    # Store it first
    result = store_csv_as_dataset_from_text(
        csv_content=csv_content,
        project_manifest_path=manifest_path,
        filename="test_with_dup_rows",
        explanation="Dataset with duplicate rows"
    )
    
    input_filename = result['output_filename']
    
    # Drop duplicates
    result = drop_duplicate_rows(
        input_filename=input_filename,
        subset_columns=None,  # All columns
        project_manifest_path=manifest_path,
        output_filename="test_no_dup_rows",
        explanation="Removed duplicate rows"
    )
    
    print(f"✓ Dropped duplicate rows")
    print(f"  Rows before: {result['n_rows_before']}")
    print(f"  Rows after: {result['n_rows_after']}")
    
    return result['output_filename']


def test_drop_empty_rows(manifest_path):
    """Test dropping completely empty rows."""
    print("\n=== Testing drop_empty_rows ===")
    
    # Create a dataset with empty rows
    df = pd.DataFrame({
        'smiles': ['CC(=O)O', None, 'CCO', None],
        'label': [1, None, 0, None],
        'name': ['acetic_acid', None, 'ethanol', None]
    })
    
    # Store it first
    import tempfile
    temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_csv.name, index=False)
    temp_csv.close()
    
    result = store_csv_as_dataset(
        file_path=temp_csv.name,
        project_manifest_path=manifest_path,
        filename="test_with_empty_rows",
        explanation="Dataset with empty rows"
    )
    
    input_filename = result['output_filename']
    
    # Clean up temp file
    Path(temp_csv.name).unlink()
    
    # Drop empty rows
    result = drop_empty_rows(
        input_filename=input_filename,
        project_manifest_path=manifest_path,
        output_filename="test_no_empty_rows",
        explanation="Removed empty rows"
    )
    
    print(f"✓ Dropped empty rows")
    print(f"  Rows before: {result['n_rows_before']}")
    print(f"  Rows after: {result['n_rows_after']}")
    
    return result['output_filename']


def inspect_manifest(manifest_path):
    """Inspect the final manifest to see all tracked resources."""
    print("\n=== Final Manifest Inspection ===")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"✓ Project: {manifest['project_name']}")
    print(f"  Total resources tracked: {len(manifest['resources'])}")
    print("\n  Resources:")
    for i, res in enumerate(manifest['resources'], 1):
        print(f"    {i}. {res['filename']} ({res['type_tag']})")
        print(f"       - {res['explaination']}")
        print(f"       - Created by: {res['parent_function_name']}")
        print(f"       - Timestamp: {res['timestamp']}")


def main():
    """Run all dataset_ops tests."""
    print("=" * 70)
    print("TESTING DATASET_OPS WITH NEW RESOURCE MANAGEMENT")
    print("=" * 70)
    
    # Create test project
    manifest_path, test_dir = create_test_project()
    
    try:
        # Test 1: Store CSV as dataset
        filename1 = test_store_csv_as_dataset(manifest_path)
        
        # Test 2: Store CSV from text
        filename2 = test_store_csv_from_text(manifest_path)
        
        # Test 3: Get dataset head
        test_get_dataset_head(manifest_path, filename1)
        
        # Test 4: Get full dataset
        test_get_dataset_full(manifest_path, filename2)
        
        # Test 5: Get dataset summary
        test_get_dataset_summary(manifest_path, filename1)
        
        # Test 6: Inspect dataset rows
        test_inspect_dataset_rows(manifest_path, filename1)
        
        # Test 7: Drop from dataset
        filename3 = test_drop_from_dataset(manifest_path, filename1)
        
        # Test 8: Keep from dataset
        filename4 = test_keep_from_dataset(manifest_path, filename1)
        
        # Test 9: Deduplicate molecules
        filename5 = test_deduplicate_molecules(manifest_path)
        
        # Test 10: Drop duplicate rows
        filename6 = test_drop_duplicate_rows(manifest_path)
        
        # Test 11: Drop empty rows
        filename7 = test_drop_empty_rows(manifest_path)
        
        # Inspect final manifest
        inspect_manifest(manifest_path)
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
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
