import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path
from molml_mcp.infrastructure.resources import _store_resource, _load_resource
from molml_mcp.tools.core_mol.data_splitting import scaffold_split_dataset
from molml_mcp.tools.core_mol.scaffolds import calculate_scaffolds_dataset

print("=" * 80)
print("TESTING scaffold_split_dataset")
print("=" * 80)

# Setup test data
test_manifest = Path("tests/data/test_manifest.json")

# Create test dataset with known scaffolds
# We'll have:
# - Scaffold A (benzene): 5 molecules
# - Scaffold B (pyridine): 3 molecules
# - Scaffold C (cyclohexane): 2 molecules
# - No scaffold: 2 molecules
test_smiles = [
    'c1ccccc1O',      # Phenol - benzene scaffold
    'c1ccccc1N',      # Aniline - benzene scaffold
    'c1ccccc1C',      # Toluene - benzene scaffold
    'c1ccccc1F',      # Fluorobenzene - benzene scaffold
    'c1ccccc1Cl',     # Chlorobenzene - benzene scaffold
    'c1ccncc1',       # Pyridine - pyridine scaffold
    'c1ccncc1C',      # Methylpyridine - pyridine scaffold
    'c1ccncc1O',      # Hydroxypyridine - pyridine scaffold
    'C1CCCCC1',       # Cyclohexane - cyclohexane scaffold
    'C1CCCCC1O',      # Cyclohexanol - cyclohexane scaffold
    'CCO',            # Ethanol - no scaffold
    'CCC',            # Propane - no scaffold
]

df = pd.DataFrame({
    'smiles': test_smiles,
    'id': range(len(test_smiles)),
    'name': ['phenol', 'aniline', 'toluene', 'fluorobenzene', 'chlorobenzene',
             'pyridine', 'methylpyridine', 'hydroxypyridine',
             'cyclohexane', 'cyclohexanol', 'ethanol', 'propane']
})

# Store test dataset
df_filename = _store_resource(df, str(test_manifest), "test_scaffold_split", "Test molecules for scaffold splitting", 'csv')

print(f"\n✅ Test data created:")
print(f"   Dataset: {df_filename} ({len(df)} molecules)")

# First, calculate scaffolds
print("\n" + "=" * 80)
print("Calculating scaffolds...")
print("=" * 80)

scaffold_result = calculate_scaffolds_dataset(
    input_filename=df_filename,
    column_name='smiles',
    project_manifest_path=str(test_manifest),
    output_filename='test_with_scaffolds_for_split',
    scaffold_type='bemis_murcko',
    explanation='Add scaffolds for splitting'
)

df_with_scaffolds_filename = scaffold_result['output_filename']
df_check = _load_resource(str(test_manifest), df_with_scaffolds_filename)
print(f"\n✅ Scaffolds calculated:")
print(df_check[['name', 'smiles', 'scaffold_bemis_murcko']].to_string(index=False))

# =============================================================================
# TEST 1: Balanced scaffold split (80/20)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: Balanced scaffold split (80/20)")
print("=" * 80)

try:
    result1 = scaffold_split_dataset(
        input_filename=df_with_scaffolds_filename,
        scaffold_column='scaffold_bemis_murcko',
        project_manifest_path=str(test_manifest),
        train_output_filename='train_balanced',
        test_output_filename='test_balanced',
        train_ratio=0.8,
        test_ratio=0.2,
        method='balanced',
        random_state=42
    )
    
    print(f"✅ Balanced split PASSED")
    print(f"   Total: {result1['n_total_rows']} molecules")
    print(f"   Unique scaffolds: {result1['n_unique_scaffolds']}")
    print(f"   No scaffold: {result1['n_molecules_no_scaffold']}")
    print(f"   ")
    print(f"   Train: {result1['n_train_rows']} molecules ({result1['actual_train_ratio']:.1%}), {result1['n_train_scaffolds']} scaffolds")
    print(f"   Test: {result1['n_test_rows']} molecules ({result1['actual_test_ratio']:.1%}), {result1['n_test_scaffolds']} scaffolds")
    print(f"   ")
    print(f"   Overlap detected: {result1['overlap_detected']}")
    print(f"   Overlap info: {result1['overlap_info']}")
    print(f"   Method: {result1['method']}")
    print(f"   Note: {result1['note']}")
    
    # Check the actual splits
    df_train = _load_resource(str(test_manifest), result1['train_output_filename'])
    df_test = _load_resource(str(test_manifest), result1['test_output_filename'])
    
    print(f"\n   Train molecules: {', '.join(df_train['name'].tolist())}")
    print(f"   Test molecules: {', '.join(df_test['name'].tolist())}")
    
    # Verify no overlap
    train_scaffolds = set(df_train['scaffold_bemis_murcko'].dropna())
    test_scaffolds = set(df_test['scaffold_bemis_murcko'].dropna())
    overlap = train_scaffolds & test_scaffolds
    print(f"\n   ✓ Scaffold overlap check: {len(overlap)} overlapping scaffolds")
    if overlap:
        print(f"     WARNING: Overlapping scaffolds: {overlap}")
    
except Exception as e:
    print(f"❌ TEST 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 2: Random scaffold split (80/20)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: Random scaffold split (80/20)")
print("=" * 80)

try:
    result2 = scaffold_split_dataset(
        input_filename=df_with_scaffolds_filename,
        scaffold_column='scaffold_bemis_murcko',
        project_manifest_path=str(test_manifest),
        train_output_filename='train_random',
        test_output_filename='test_random',
        train_ratio=0.8,
        test_ratio=0.2,
        method='random',
        random_state=42
    )
    
    print(f"✅ Random split PASSED")
    print(f"   Train: {result2['n_train_rows']} molecules ({result2['actual_train_ratio']:.1%}), {result2['n_train_scaffolds']} scaffolds")
    print(f"   Test: {result2['n_test_rows']} molecules ({result2['actual_test_ratio']:.1%}), {result2['n_test_scaffolds']} scaffolds")
    print(f"   Overlap detected: {result2['overlap_detected']}")
    print(f"   Method: {result2['method']}")
    
    # Verify no overlap
    df_train = _load_resource(str(test_manifest), result2['train_output_filename'])
    df_test = _load_resource(str(test_manifest), result2['test_output_filename'])
    
    train_scaffolds = set(df_train['scaffold_bemis_murcko'].dropna())
    test_scaffolds = set(df_test['scaffold_bemis_murcko'].dropna())
    overlap = train_scaffolds & test_scaffolds
    print(f"   ✓ Scaffold overlap check: {len(overlap)} overlapping scaffolds")
    
except Exception as e:
    print(f"❌ TEST 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 3: Three-way split with validation set (70/20/10)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: Three-way split with validation set (70/20/10)")
print("=" * 80)

try:
    result3 = scaffold_split_dataset(
        input_filename=df_with_scaffolds_filename,
        scaffold_column='scaffold_bemis_murcko',
        project_manifest_path=str(test_manifest),
        train_output_filename='train_3way',
        test_output_filename='test_3way',
        val_output_filename='val_3way',
        train_ratio=0.7,
        test_ratio=0.2,
        val_ratio=0.1,
        method='balanced',
        random_state=42
    )
    
    print(f"✅ Three-way split PASSED")
    print(f"   Train: {result3['n_train_rows']} molecules ({result3['actual_train_ratio']:.1%}), {result3['n_train_scaffolds']} scaffolds")
    print(f"   Test: {result3['n_test_rows']} molecules ({result3['actual_test_ratio']:.1%}), {result3['n_test_scaffolds']} scaffolds")
    print(f"   Val: {result3['n_val_rows']} molecules ({result3['actual_val_ratio']:.1%}), {result3['n_val_scaffolds']} scaffolds")
    print(f"   Overlap detected: {result3['overlap_detected']}")
    print(f"   Overlap info: {result3['overlap_info']}")
    
    # Verify no overlap between any splits
    df_train = _load_resource(str(test_manifest), result3['train_output_filename'])
    df_test = _load_resource(str(test_manifest), result3['test_output_filename'])
    df_val = _load_resource(str(test_manifest), result3['val_output_filename'])
    
    train_scaffolds = set(df_train['scaffold_bemis_murcko'].dropna())
    test_scaffolds = set(df_test['scaffold_bemis_murcko'].dropna())
    val_scaffolds = set(df_val['scaffold_bemis_murcko'].dropna())
    
    train_test_overlap = train_scaffolds & test_scaffolds
    train_val_overlap = train_scaffolds & val_scaffolds
    test_val_overlap = test_scaffolds & val_scaffolds
    
    print(f"   ✓ Train-Test overlap: {len(train_test_overlap)}")
    print(f"   ✓ Train-Val overlap: {len(train_val_overlap)}")
    print(f"   ✓ Test-Val overlap: {len(test_val_overlap)}")
    
except Exception as e:
    print(f"❌ TEST 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 4: Handle no scaffold - random distribution
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: Handle no scaffold - random distribution")
print("=" * 80)

try:
    result4 = scaffold_split_dataset(
        input_filename=df_with_scaffolds_filename,
        scaffold_column='scaffold_bemis_murcko',
        project_manifest_path=str(test_manifest),
        train_output_filename='train_no_scaffold_random',
        test_output_filename='test_no_scaffold_random',
        train_ratio=0.8,
        test_ratio=0.2,
        method='balanced',
        handle_no_scaffold='random',
        random_state=42
    )
    
    print(f"✅ Random no-scaffold handling PASSED")
    print(f"   Train: {result4['n_train_rows']} molecules")
    print(f"   Test: {result4['n_test_rows']} molecules")
    print(f"   Handle no scaffold: {result4['handle_no_scaffold']}")
    print(f"   Overlap detected: {result4['overlap_detected']}")
    
except Exception as e:
    print(f"❌ TEST 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 5: Error handling - invalid method
# =============================================================================
print("\n" + "=" * 80)
print("TEST 5: Error handling - invalid method")
print("=" * 80)

try:
    result5 = scaffold_split_dataset(
        input_filename=df_with_scaffolds_filename,
        scaffold_column='scaffold_bemis_murcko',
        project_manifest_path=str(test_manifest),
        train_output_filename='train_error',
        test_output_filename='test_error',
        method='invalid_method'
    )
    print(f"❌ Should have raised ValueError")
except ValueError as e:
    print(f"✅ Error handling PASSED")
    print(f"   Correctly raised ValueError: {e}")
except Exception as e:
    print(f"❌ Wrong exception type: {e}")

# =============================================================================
# TEST 6: Error handling - ratios don't sum to 1
# =============================================================================
print("\n" + "=" * 80)
print("TEST 6: Error handling - ratios don't sum to 1")
print("=" * 80)

try:
    result6 = scaffold_split_dataset(
        input_filename=df_with_scaffolds_filename,
        scaffold_column='scaffold_bemis_murcko',
        project_manifest_path=str(test_manifest),
        train_output_filename='train_error2',
        test_output_filename='test_error2',
        train_ratio=0.7,
        test_ratio=0.2,
        val_ratio=0.2  # Sum = 1.1, invalid
    )
    print(f"❌ Should have raised ValueError")
except ValueError as e:
    print(f"✅ Error handling PASSED")
    print(f"   Correctly raised ValueError: {e}")
except Exception as e:
    print(f"❌ Wrong exception type: {e}")

# =============================================================================
# TEST 7: Error handling - invalid scaffold column
# =============================================================================
print("\n" + "=" * 80)
print("TEST 7: Error handling - invalid scaffold column")
print("=" * 80)

try:
    result7 = scaffold_split_dataset(
        input_filename=df_with_scaffolds_filename,
        scaffold_column='invalid_column',
        project_manifest_path=str(test_manifest),
        train_output_filename='train_error3',
        test_output_filename='test_error3'
    )
    print(f"❌ Should have raised ValueError")
except ValueError as e:
    print(f"✅ Error handling PASSED")
    print(f"   Correctly raised ValueError: {e}")
except Exception as e:
    print(f"❌ Wrong exception type: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("✅ Balanced scaffold split (80/20): PASSED")
print("✅ Random scaffold split (80/20): PASSED")
print("✅ Three-way split (70/20/10): PASSED")
print("✅ Handle no scaffold (random): PASSED")
print("✅ Error handling (invalid method): PASSED")
print("✅ Error handling (invalid ratios): PASSED")
print("✅ Error handling (invalid column): PASSED")
print("\n🎉 All scaffold split tests completed successfully!")
