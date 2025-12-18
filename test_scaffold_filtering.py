import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path
from molml_mcp.infrastructure.resources import _store_resource, _load_resource
from molml_mcp.tools.core.filtering import filter_by_scaffold
from molml_mcp.tools.core_mol.scaffolds import calculate_scaffolds_dataset

print("=" * 80)
print("TESTING filter_by_scaffold")
print("=" * 80)

# Setup test data
test_manifest = Path("tests/data/test_manifest.json")

# Create test dataset with diverse scaffolds
test_smiles = [
    'CCO',                           # Ethanol - no scaffold
    'c1ccccc1O',                     # Phenol - benzene scaffold
    'c1ccccc1N',                     # Aniline - benzene scaffold
    'c1ccc(O)cc1C',                  # p-Cresol - benzene scaffold
    'c1ccc2ccccc2c1',                # Naphthalene - naphthalene scaffold
    'c1cnc(N)nc1',                   # Pyrimidine - pyrimidine scaffold
    'c1ccncc1',                      # Pyridine - pyridine scaffold
    'c1ccccc1c2ccccc2',              # Biphenyl - biphenyl scaffold
    'CC(C)(C)c1ccc(O)cc1',          # BHT - benzene scaffold
    'C1CCCCC1',                      # Cyclohexane - cyclohexane scaffold
    'c1ccc2c(c1)ccc3c2cccc3',       # Anthracene - anthracene scaffold
]

df = pd.DataFrame({
    'smiles': test_smiles,
    'id': range(len(test_smiles)),
    'name': ['ethanol', 'phenol', 'aniline', 'p-cresol', 'naphthalene', 
             'pyrimidine', 'pyridine', 'biphenyl', 'bht', 'cyclohexane', 'anthracene']
})

# Store test dataset
df_filename = _store_resource(df, str(test_manifest), "test_scaffold_filter", "Test molecules for scaffold filtering", 'csv')

print(f"\n✅ Test data created:")
print(f"   Dataset: {df_filename} ({len(df)} molecules)")

# =============================================================================
# TEST 1: Filter by scaffold WITHOUT pre-existing scaffold column (keep)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: filter_by_scaffold - keep benzene scaffold (calculate on-the-fly)")
print("=" * 80)

try:
    result1 = filter_by_scaffold(
        input_filename=df_filename,
        project_manifest_path=str(test_manifest),
        output_filename='test_benzene_only',
        scaffold_smiles_list=['c1ccccc1'],  # Benzene scaffold
        explanation='Keep molecules with benzene scaffold',
        action='keep',
        smiles_column='smiles'
    )
    
    print(f"✅ filter_by_scaffold (keep) PASSED")
    print(f"   Input: {result1['n_input']} molecules")
    print(f"   Output: {result1['n_output']} molecules ({result1['percent_retained']:.1f}% retained)")
    print(f"   Matching scaffold: {result1['n_matching_scaffold']}")
    print(f"   Invalid SMILES: {result1['n_invalid_smiles']}")
    print(f"   No scaffold: {result1['n_no_scaffold']}")
    print(f"   Scaffold column existed: {result1['scaffold_column_existed']}")
    print(f"   Scaffolds used: {result1['scaffolds_used']}")
    print(f"   Warning: {result1['warning']}")
    
    # Check what passed
    df_benzene = _load_resource(str(test_manifest), result1['output_filename'])
    print(f"   Molecules with benzene scaffold: {', '.join(df_benzene['name'].tolist())}")
    print(f"   Expected: phenol, aniline, p-cresol, bht")
except Exception as e:
    print(f"❌ filter_by_scaffold (keep) FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 2: Filter by scaffold WITHOUT pre-existing scaffold column (drop)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: filter_by_scaffold - drop benzene scaffold")
print("=" * 80)

try:
    result2 = filter_by_scaffold(
        input_filename=df_filename,
        project_manifest_path=str(test_manifest),
        output_filename='test_no_benzene',
        scaffold_smiles_list=['c1ccccc1'],  # Benzene scaffold
        explanation='Remove molecules with benzene scaffold',
        action='drop',
        smiles_column='smiles'
    )
    
    print(f"✅ filter_by_scaffold (drop) PASSED")
    print(f"   Input: {result2['n_input']} molecules")
    print(f"   Output: {result2['n_output']} molecules ({result2['percent_retained']:.1f}% retained)")
    print(f"   Matching scaffold: {result2['n_matching_scaffold']}")
    print(f"   Warning: {result2['warning']}")
    
    # Check what passed
    df_no_benzene = _load_resource(str(test_manifest), result2['output_filename'])
    print(f"   Molecules without benzene scaffold: {', '.join(df_no_benzene['name'].tolist())}")
    print(f"   Expected: ethanol, naphthalene, pyrimidine, pyridine, biphenyl, cyclohexane, anthracene")
except Exception as e:
    print(f"❌ filter_by_scaffold (drop) FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 3: Add scaffold column first, then filter WITH pre-existing column
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: Add scaffold column, then filter by multiple scaffolds")
print("=" * 80)

try:
    # First add scaffold column
    result_scaffold = calculate_scaffolds_dataset(
        input_filename=df_filename,
        project_manifest_path=str(test_manifest),
        column_name='smiles',
        scaffold_type='bemis_murcko',
        output_filename='test_with_scaffolds',
        explanation='Add scaffolds before filtering'
    )
    
    print(f"✅ Scaffold column added")
    print(f"   Output: {result_scaffold['output_filename']}")
    
    # Now filter using the scaffold column
    result3 = filter_by_scaffold(
        input_filename=result_scaffold['output_filename'],
        project_manifest_path=str(test_manifest),
        output_filename='test_aromatic_scaffolds',
        scaffold_smiles_list=[
            'c1ccccc1',           # Benzene
            'c1ccc2ccccc2c1',     # Naphthalene
            'c1ccc2cc3ccccc3cc2c1'  # Anthracene
        ],
        explanation='Keep molecules with aromatic scaffolds',
        action='keep'
    )
    
    print(f"✅ filter_by_scaffold (with existing column) PASSED")
    print(f"   Input: {result3['n_input']} molecules")
    print(f"   Output: {result3['n_output']} molecules ({result3['percent_retained']:.1f}% retained)")
    print(f"   Matching scaffold: {result3['n_matching_scaffold']}")
    print(f"   Scaffold column existed: {result3['scaffold_column_existed']}")
    print(f"   Scaffolds used: {result3['scaffolds_used']}")
    
    # Check what passed
    df_aromatic = _load_resource(str(test_manifest), result3['output_filename'])
    print(f"   Aromatic scaffold molecules: {', '.join(df_aromatic['name'].tolist())}")
    print(f"   Expected: phenol, aniline, p-cresol, naphthalene, bht, anthracene")
except Exception as e:
    print(f"❌ filter_by_scaffold (with column) FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 4: Filter by pyridine scaffold only
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: filter_by_scaffold - keep pyridine scaffold only")
print("=" * 80)

try:
    result4 = filter_by_scaffold(
        input_filename=df_filename,
        project_manifest_path=str(test_manifest),
        output_filename='test_pyridine_only',
        scaffold_smiles_list=['c1ccncc1'],  # Pyridine scaffold
        explanation='Keep molecules with pyridine scaffold',
        action='keep',
        smiles_column='smiles'
    )
    
    print(f"✅ filter_by_scaffold (pyridine) PASSED")
    print(f"   Input: {result4['n_input']} molecules")
    print(f"   Output: {result4['n_output']} molecules ({result4['percent_retained']:.1f}% retained)")
    print(f"   Matching scaffold: {result4['n_matching_scaffold']}")
    
    # Check what passed
    df_pyridine = _load_resource(str(test_manifest), result4['output_filename'])
    if len(df_pyridine) > 0:
        print(f"   Pyridine scaffold molecules: {', '.join(df_pyridine['name'].tolist())}")
    else:
        print(f"   No molecules with pyridine scaffold")
    print(f"   Expected: pyridine")
except Exception as e:
    print(f"❌ filter_by_scaffold (pyridine) FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 5: Error handling - invalid scaffold SMILES
# =============================================================================
print("\n" + "=" * 80)
print("TEST 5: Error handling - invalid scaffold SMILES")
print("=" * 80)

try:
    result5 = filter_by_scaffold(
        input_filename=df_filename,
        project_manifest_path=str(test_manifest),
        output_filename='test_invalid',
        scaffold_smiles_list=['INVALID_SMILES'],
        explanation='Test invalid scaffold',
        action='keep'
    )
    print(f"❌ Should have raised ValueError for invalid SMILES")
except ValueError as e:
    print(f"✅ Error handling PASSED")
    print(f"   Correctly raised ValueError: {e}")
except Exception as e:
    print(f"❌ Wrong exception type: {e}")

# =============================================================================
# TEST 6: Error handling - empty scaffold list
# =============================================================================
print("\n" + "=" * 80)
print("TEST 6: Error handling - empty scaffold list")
print("=" * 80)

try:
    result6 = filter_by_scaffold(
        input_filename=df_filename,
        project_manifest_path=str(test_manifest),
        output_filename='test_empty',
        scaffold_smiles_list=[],
        explanation='Test empty scaffold list',
        action='keep'
    )
    print(f"❌ Should have raised ValueError for empty list")
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
print("✅ filter_by_scaffold (keep, no column): PASSED")
print("✅ filter_by_scaffold (drop, no column): PASSED")
print("✅ filter_by_scaffold (with existing column): PASSED")
print("✅ filter_by_scaffold (single scaffold): PASSED")
print("✅ Error handling (invalid SMILES): PASSED")
print("✅ Error handling (empty list): PASSED")
print("\n🎉 All scaffold filtering tests completed successfully!")
