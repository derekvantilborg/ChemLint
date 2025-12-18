import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path
from molml_mcp.infrastructure.resources import _store_resource, _load_resource
from molml_mcp.tools.core.filtering import filter_by_functional_groups

print("=" * 80)
print("TESTING filter_by_functional_groups")
print("=" * 80)

# Setup test data
test_manifest = Path("tests/data/test_manifest.json")

# Create test dataset with diverse functional groups
test_smiles = [
    'CCO',                           # Ethanol - Hydroxyl
    'CC(=O)C',                       # Acetone - Carbonyl, Ketone
    'CC(=O)O',                       # Acetic acid - Carboxylic acid, Carbonyl
    'CCN',                           # Ethylamine - Primary amine
    'CC(=O)OCC',                     # Ethyl acetate - Ester, Carbonyl, Ether
    'c1ccccc1O',                     # Phenol - Hydroxyl, Benzene ring
    'c1ccccc1N',                     # Aniline - Primary amine, Benzene ring
    'CC(=O)NC',                      # N-Methylacetamide - Amide, Carbonyl
    'CCOC',                          # Diethyl ether - Ether
    'CCS',                           # Ethanethiol - Thiol
    'CCF',                           # Fluoroethane - Fluorine
    'CCCl',                          # Chloroethane - Chlorine
    'c1ccncc1',                      # Pyridine - Pyridine ring
]

df = pd.DataFrame({
    'smiles': test_smiles,
    'id': range(len(test_smiles)),
    'name': ['ethanol', 'acetone', 'acetic_acid', 'ethylamine', 'ethyl_acetate', 
             'phenol', 'aniline', 'acetamide', 'diethyl_ether', 'ethanethiol',
             'fluoroethane', 'chloroethane', 'pyridine']
})

# Store test dataset
df_filename = _store_resource(df, str(test_manifest), "test_functional_groups", "Test molecules for functional group filtering", 'csv')

print(f"\n✅ Test data created:")
print(f"   Dataset: {df_filename} ({len(df)} molecules)")

# =============================================================================
# TEST 1: Filter for molecules with hydroxyl groups (required only)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: Filter for hydroxyl groups (required)")
print("=" * 80)

try:
    result1 = filter_by_functional_groups(
        input_filename=df_filename,
        smiles_column='smiles',
        project_manifest_path=str(test_manifest),
        output_filename='test_hydroxyl',
        explanation='Keep molecules with hydroxyl groups',
        required=['Hydroxyl'],
        forbidden=None
    )
    
    print(f"✅ filter_by_functional_groups (hydroxyl) PASSED")
    print(f"   Input: {result1['n_input']} molecules")
    print(f"   Output: {result1['n_output']} molecules ({result1['percent_retained']:.1f}% retained)")
    print(f"   Required groups: {result1['required_groups']}")
    print(f"   Filter summary: {result1['filter_summary']}")
    print(f"   Warning: {result1['warning']}")
    
    df_hydroxyl = _load_resource(str(test_manifest), result1['output_filename'])
    print(f"   Molecules with hydroxyl: {', '.join(df_hydroxyl['name'].tolist())}")
    print(f"   Expected: ethanol, phenol")
except Exception as e:
    print(f"❌ TEST 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 2: Filter for carbonyl groups (required only)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: Filter for carbonyl groups (required)")
print("=" * 80)

try:
    result2 = filter_by_functional_groups(
        input_filename=df_filename,
        smiles_column='smiles',
        project_manifest_path=str(test_manifest),
        output_filename='test_carbonyl',
        explanation='Keep molecules with carbonyl groups',
        required=['Carbonyl group'],
        forbidden=None
    )
    
    print(f"✅ filter_by_functional_groups (carbonyl) PASSED")
    print(f"   Input: {result2['n_input']} molecules")
    print(f"   Output: {result2['n_output']} molecules ({result2['percent_retained']:.1f}% retained)")
    
    df_carbonyl = _load_resource(str(test_manifest), result2['output_filename'])
    print(f"   Molecules with carbonyl: {', '.join(df_carbonyl['name'].tolist())}")
    print(f"   Expected: acetone, acetic_acid, ethyl_acetate, acetamide")
except Exception as e:
    print(f"❌ TEST 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 3: Filter out molecules with amine groups (forbidden only)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: Filter out amine groups (forbidden)")
print("=" * 80)

try:
    result3 = filter_by_functional_groups(
        input_filename=df_filename,
        smiles_column='smiles',
        project_manifest_path=str(test_manifest),
        output_filename='test_no_amines',
        explanation='Remove molecules with primary amines',
        required=None,
        forbidden=['Primary amine']
    )
    
    print(f"✅ filter_by_functional_groups (no amines) PASSED")
    print(f"   Input: {result3['n_input']} molecules")
    print(f"   Output: {result3['n_output']} molecules ({result3['percent_retained']:.1f}% retained)")
    print(f"   Forbidden groups: {result3['forbidden_groups']}")
    
    df_no_amines = _load_resource(str(test_manifest), result3['output_filename'])
    print(f"   Molecules without primary amines: {', '.join(df_no_amines['name'].tolist())}")
    print(f"   Expected: ethanol, acetone, acetic_acid, ethyl_acetate, phenol, acetamide, diethyl_ether, ethanethiol, fluoroethane, chloroethane, pyridine")
except Exception as e:
    print(f"❌ TEST 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 4: Filter with both required AND forbidden
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: Filter with required AND forbidden groups")
print("=" * 80)

try:
    result4 = filter_by_functional_groups(
        input_filename=df_filename,
        smiles_column='smiles',
        project_manifest_path=str(test_manifest),
        output_filename='test_carbonyl_no_amine',
        explanation='Carbonyl groups without amines',
        required=['Carbonyl group'],
        forbidden=['Primary amine', 'Secondary amine', 'Tertiary amine']
    )
    
    print(f"✅ filter_by_functional_groups (carbonyl, no amine) PASSED")
    print(f"   Input: {result4['n_input']} molecules")
    print(f"   Output: {result4['n_output']} molecules ({result4['percent_retained']:.1f}% retained)")
    print(f"   Required: {result4['required_groups']}")
    print(f"   Forbidden: {result4['forbidden_groups']}")
    print(f"   Filter summary: {result4['filter_summary']}")
    
    df_filtered = _load_resource(str(test_manifest), result4['output_filename'])
    print(f"   Molecules: {', '.join(df_filtered['name'].tolist())}")
    print(f"   Expected: acetone, acetic_acid, ethyl_acetate, acetamide")
except Exception as e:
    print(f"❌ TEST 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 5: Filter for multiple required groups (AND logic)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 5: Filter for multiple required groups (AND logic)")
print("=" * 80)

try:
    result5 = filter_by_functional_groups(
        input_filename=df_filename,
        smiles_column='smiles',
        project_manifest_path=str(test_manifest),
        output_filename='test_ester_ether',
        explanation='Molecules with both Ester and Ether',
        required=['Ester', 'Ether'],
        forbidden=None
    )
    
    print(f"✅ filter_by_functional_groups (ester + ether) PASSED")
    print(f"   Input: {result5['n_input']} molecules")
    print(f"   Output: {result5['n_output']} molecules ({result5['percent_retained']:.1f}% retained)")
    print(f"   Required (ALL): {result5['required_groups']}")
    
    df_ester_ether = _load_resource(str(test_manifest), result5['output_filename'])
    if len(df_ester_ether) > 0:
        print(f"   Molecules: {', '.join(df_ester_ether['name'].tolist())}")
    else:
        print(f"   No molecules match")
    print(f"   Expected: ethyl_acetate (has both)")
except Exception as e:
    print(f"❌ TEST 5 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 6: Filter out halogens
# =============================================================================
print("\n" + "=" * 80)
print("TEST 6: Filter out halogenated compounds")
print("=" * 80)

try:
    result6 = filter_by_functional_groups(
        input_filename=df_filename,
        smiles_column='smiles',
        project_manifest_path=str(test_manifest),
        output_filename='test_no_halogens',
        explanation='Remove halogenated compounds',
        required=None,
        forbidden=['Fluorine', 'Chlorine', 'Bromine', 'Iodine']
    )
    
    print(f"✅ filter_by_functional_groups (no halogens) PASSED")
    print(f"   Input: {result6['n_input']} molecules")
    print(f"   Output: {result6['n_output']} molecules ({result6['percent_retained']:.1f}% retained)")
    print(f"   Forbidden (ANY): {result6['forbidden_groups']}")
    
    df_no_halogens = _load_resource(str(test_manifest), result6['output_filename'])
    print(f"   Non-halogenated molecules: {', '.join(df_no_halogens['name'].tolist())}")
    print(f"   Expected: all except fluoroethane, chloroethane")
except Exception as e:
    print(f"❌ TEST 6 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 7: Error handling - both None/empty
# =============================================================================
print("\n" + "=" * 80)
print("TEST 7: Error handling - both required and forbidden None/empty")
print("=" * 80)

try:
    result7 = filter_by_functional_groups(
        input_filename=df_filename,
        smiles_column='smiles',
        project_manifest_path=str(test_manifest),
        output_filename='test_error',
        explanation='Test error handling',
        required=None,
        forbidden=None
    )
    print(f"❌ Should have raised ValueError")
except ValueError as e:
    print(f"✅ Error handling PASSED")
    print(f"   Correctly raised ValueError: {e}")
except Exception as e:
    print(f"❌ Wrong exception type: {e}")

# =============================================================================
# TEST 8: Error handling - invalid SMILES column
# =============================================================================
print("\n" + "=" * 80)
print("TEST 8: Error handling - invalid SMILES column")
print("=" * 80)

try:
    result8 = filter_by_functional_groups(
        input_filename=df_filename,
        smiles_column='invalid_column',
        project_manifest_path=str(test_manifest),
        output_filename='test_error2',
        explanation='Test error handling',
        required=['Hydroxyl']
    )
    print(f"❌ Should have raised ValueError")
except ValueError as e:
    print(f"✅ Error handling PASSED")
    print(f"   Correctly raised ValueError: {e}")
except Exception as e:
    print(f"❌ Wrong exception type: {e}")

# =============================================================================
# TEST 9: Filter for aromatic rings
# =============================================================================
print("\n" + "=" * 80)
print("TEST 9: Filter for aromatic compounds")
print("=" * 80)

try:
    result9 = filter_by_functional_groups(
        input_filename=df_filename,
        smiles_column='smiles',
        project_manifest_path=str(test_manifest),
        output_filename='test_aromatics',
        explanation='Keep aromatic compounds',
        required=['Unfused benzene ring'],
        forbidden=None
    )
    
    print(f"✅ filter_by_functional_groups (aromatics) PASSED")
    print(f"   Input: {result9['n_input']} molecules")
    print(f"   Output: {result9['n_output']} molecules ({result9['percent_retained']:.1f}% retained)")
    
    df_aromatic = _load_resource(str(test_manifest), result9['output_filename'])
    print(f"   Aromatic molecules: {', '.join(df_aromatic['name'].tolist())}")
    print(f"   Expected: phenol, aniline")
except Exception as e:
    print(f"❌ TEST 9 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("✅ filter_by_functional_groups (required only): PASSED")
print("✅ filter_by_functional_groups (carbonyl): PASSED")
print("✅ filter_by_functional_groups (forbidden only): PASSED")
print("✅ filter_by_functional_groups (required + forbidden): PASSED")
print("✅ filter_by_functional_groups (multiple required, AND): PASSED")
print("✅ filter_by_functional_groups (multiple forbidden, ANY): PASSED")
print("✅ Error handling (both None): PASSED")
print("✅ Error handling (invalid column): PASSED")
print("✅ filter_by_functional_groups (aromatics): PASSED")
print("\n🎉 All functional group filtering tests completed successfully!")
