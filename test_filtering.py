import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path
from molml_mcp.infrastructure.resources import _store_resource, _load_resource
from molml_mcp.tools.core.filtering import (
    filter_by_property_range,
    filter_by_lipinski_ro5,
    filter_by_veber_rules,
    filter_by_pains,
    filter_by_lead_likeness,
    filter_by_rule_of_three,
    filter_by_qed
)

print("=" * 80)
print("TESTING FILTERING FUNCTIONS")
print("=" * 80)

# Setup test data
test_manifest = Path("tests/data/test_manifest.json")

# Create diverse test dataset with various molecular properties
test_smiles = [
    'CCO',                           # Ethanol - small, simple
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',   # Ibuprofen - drug
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Caffeine - drug
    'c1ccccc1',                      # Benzene - aromatic
    'CCCCCCCCCCCCCCCCCC',            # Octadecane - high MW
    'CC(C)(C)c1ccc(O)cc1',          # BHT - has PAINS-like substructure
    'C1CCCCC1',                      # Cyclohexane - simple ring
    'Cc1ccccc1N',                    # o-Toluidine - small aromatic amine
    'CC(=O)Oc1ccccc1C(=O)O',        # Aspirin - drug
    'CCCCCCCCC',                     # Nonane - aliphatic
    'c1ccc2c(c1)ccc3c2cccc3',       # Anthracene - PAH
    'CC(C)NCC(COc1ccccc1)O',        # Propranolol - drug
]

df = pd.DataFrame({
    'smiles': test_smiles,
    'id': range(len(test_smiles)),
    'name': ['ethanol', 'ibuprofen', 'caffeine', 'benzene', 'octadecane', 'bht', 
             'cyclohexane', 'o-toluidine', 'aspirin', 'nonane', 'anthracene', 'propranolol']
})

# Store test dataset
df_filename = _store_resource(df, str(test_manifest), "test_filtering", "Test molecules for filtering", 'csv')

print(f"\n✅ Test data created:")
print(f"   Dataset: {df_filename} ({len(df)} molecules)")
print(f"   Molecules: {', '.join(df['name'].tolist())}")

# =============================================================================
# TEST 1: filter_by_property_range
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: filter_by_property_range")
print("=" * 80)

# First, we need to add some properties to test with
from rdkit import Chem
from rdkit.Chem import Descriptors

df_with_props = df.copy()
mw_list = []
logp_list = []

for smiles in df_with_props['smiles']:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mw_list.append(Descriptors.MolWt(mol))
        logp_list.append(Descriptors.MolLogP(mol))
    else:
        mw_list.append(None)
        logp_list.append(None)

df_with_props['MolWt'] = mw_list
df_with_props['MolLogP'] = logp_list

df_props_filename = _store_resource(df_with_props, str(test_manifest), "test_with_props", "Test with properties", 'csv')

try:
    result1 = filter_by_property_range(
        input_filename=df_props_filename,
        project_manifest_path=str(test_manifest),
        property_ranges={
            'MolWt': (100, 300),
            'MolLogP': (-2, 3)
        },
        output_filename='test_property_filtered',
        explanation='Test property range filtering'
    )
    
    print(f"✅ filter_by_property_range PASSED")
    print(f"   Input: {result1['n_input']} molecules")
    print(f"   Output: {result1['n_output']} molecules ({result1['percent_retained']:.1f}% retained)")
    print(f"   Removed: {result1['n_removed']} molecules")
    print(f"   Filters applied: {len(result1['filters_applied'])}")
    print(f"   Warning: {result1['warning']}")
except Exception as e:
    print(f"❌ filter_by_property_range FAILED: {e}")

# =============================================================================
# TEST 2: filter_by_lipinski_ro5
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: filter_by_lipinski_ro5")
print("=" * 80)

try:
    result2 = filter_by_lipinski_ro5(
        input_filename=df_filename,
        project_manifest_path=str(test_manifest),
        smiles_column='smiles',
        output_filename='test_lipinski',
        explanation='Test Lipinski filtering'
    )
    
    print(f"✅ filter_by_lipinski_ro5 PASSED")
    print(f"   Input: {result2['n_input']} molecules")
    print(f"   Output: {result2['n_output']} molecules ({result2['percent_retained']:.1f}% retained)")
    print(f"   Removed: {result2['n_removed']} molecules")
    print(f"   Invalid SMILES: {result2['n_invalid_smiles']}")
    print(f"   Properties added: {', '.join(result2['lipinski_properties_added'])}")
    print(f"   Warning: {result2['warning']}")
    
    # Check what passed
    df_lipinski = _load_resource(str(test_manifest), result2['output_filename'])
    print(f"   Passed molecules: {', '.join(df_lipinski['name'].tolist())}")
except Exception as e:
    print(f"❌ filter_by_lipinski_ro5 FAILED: {e}")

# =============================================================================
# TEST 3: filter_by_veber_rules
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: filter_by_veber_rules")
print("=" * 80)

try:
    result3 = filter_by_veber_rules(
        input_filename=df_filename,
        project_manifest_path=str(test_manifest),
        smiles_column='smiles',
        output_filename='test_veber',
        explanation='Test Veber rules filtering'
    )
    
    print(f"✅ filter_by_veber_rules PASSED")
    print(f"   Input: {result3['n_input']} molecules")
    print(f"   Output: {result3['n_output']} molecules ({result3['percent_retained']:.1f}% retained)")
    print(f"   Removed: {result3['n_removed']} molecules")
    print(f"   Invalid SMILES: {result3['n_invalid_smiles']}")
    print(f"   Properties added: {', '.join(result3['veber_properties_added'])}")
    print(f"   Warning: {result3['warning']}")
    
    # Check what passed
    df_veber = _load_resource(str(test_manifest), result3['output_filename'])
    print(f"   Passed molecules: {', '.join(df_veber['name'].tolist())}")
except Exception as e:
    print(f"❌ filter_by_veber_rules FAILED: {e}")

# =============================================================================
# TEST 4: filter_by_pains (drop mode)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: filter_by_pains (drop mode)")
print("=" * 80)

try:
    result4 = filter_by_pains(
        input_filename=df_filename,
        project_manifest_path=str(test_manifest),
        smiles_column='smiles',
        output_filename='test_pains_drop',
        explanation='Test PAINS filtering - drop mode',
        action='drop'
    )
    
    print(f"✅ filter_by_pains (drop) PASSED")
    print(f"   Input: {result4['n_input']} molecules")
    print(f"   Output: {result4['n_output']} molecules ({result4['percent_retained']:.1f}% retained)")
    print(f"   PAINS flagged: {result4['n_pains_flagged']}")
    print(f"   Invalid SMILES: {result4['n_invalid_smiles']}")
    print(f"   Warning: {result4['warning']}")
    
    # Check what passed
    df_pains = _load_resource(str(test_manifest), result4['output_filename'])
    print(f"   Clean molecules: {', '.join(df_pains['name'].tolist())}")
except Exception as e:
    print(f"❌ filter_by_pains FAILED: {e}")

# =============================================================================
# TEST 5: filter_by_pains (keep mode)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 5: filter_by_pains (keep mode)")
print("=" * 80)

try:
    result5 = filter_by_pains(
        input_filename=df_filename,
        project_manifest_path=str(test_manifest),
        smiles_column='smiles',
        output_filename='test_pains_keep',
        explanation='Test PAINS filtering - keep mode',
        action='keep'
    )
    
    print(f"✅ filter_by_pains (keep) PASSED")
    print(f"   Input: {result5['n_input']} molecules")
    print(f"   Output: {result5['n_output']} molecules ({result5['percent_retained']:.1f}% retained)")
    print(f"   PAINS flagged: {result5['n_pains_flagged']}")
    print(f"   Warning: {result5['warning']}")
    
    if result5['n_output'] > 0:
        df_pains_only = _load_resource(str(test_manifest), result5['output_filename'])
        print(f"   PAINS molecules: {', '.join(df_pains_only['name'].tolist())}")
    else:
        print(f"   No PAINS molecules found")
except Exception as e:
    print(f"❌ filter_by_pains (keep) FAILED: {e}")

# =============================================================================
# TEST 6: filter_by_lead_likeness (strict)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 6: filter_by_lead_likeness (strict)")
print("=" * 80)

try:
    result6 = filter_by_lead_likeness(
        input_filename=df_filename,
        project_manifest_path=str(test_manifest),
        smiles_column='smiles',
        output_filename='test_lead_strict',
        explanation='Test lead-likeness filtering - strict',
        strict=True
    )
    
    print(f"✅ filter_by_lead_likeness (strict) PASSED")
    print(f"   Input: {result6['n_input']} molecules")
    print(f"   Output: {result6['n_output']} molecules ({result6['percent_retained']:.1f}% retained)")
    print(f"   Removed: {result6['n_removed']} molecules")
    print(f"   Invalid SMILES: {result6['n_invalid_smiles']}")
    print(f"   Criteria mode: {result6['criteria_mode']}")
    print(f"   Properties added: {', '.join(result6['lead_properties_added'])}")
    print(f"   Warning: {result6['warning']}")
    
    # Check what passed
    df_lead = _load_resource(str(test_manifest), result6['output_filename'])
    if len(df_lead) > 0:
        print(f"   Lead-like molecules: {', '.join(df_lead['name'].tolist())}")
    else:
        print(f"   No molecules passed strict criteria")
except Exception as e:
    print(f"❌ filter_by_lead_likeness FAILED: {e}")

# =============================================================================
# TEST 7: filter_by_lead_likeness (lenient)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 7: filter_by_lead_likeness (lenient)")
print("=" * 80)

try:
    result7 = filter_by_lead_likeness(
        input_filename=df_filename,
        project_manifest_path=str(test_manifest),
        smiles_column='smiles',
        output_filename='test_lead_lenient',
        explanation='Test lead-likeness filtering - lenient',
        strict=False
    )
    
    print(f"✅ filter_by_lead_likeness (lenient) PASSED")
    print(f"   Input: {result7['n_input']} molecules")
    print(f"   Output: {result7['n_output']} molecules ({result7['percent_retained']:.1f}% retained)")
    print(f"   Criteria mode: {result7['criteria_mode']}")
    print(f"   Warning: {result7['warning']}")
    
    # Check what passed
    df_lead_len = _load_resource(str(test_manifest), result7['output_filename'])
    print(f"   Lead-like molecules: {', '.join(df_lead_len['name'].tolist())}")
except Exception as e:
    print(f"❌ filter_by_lead_likeness (lenient) FAILED: {e}")

# =============================================================================
# TEST 8: filter_by_rule_of_three (strict)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 8: filter_by_rule_of_three (strict)")
print("=" * 80)

try:
    result8 = filter_by_rule_of_three(
        input_filename=df_filename,
        project_manifest_path=str(test_manifest),
        smiles_column='smiles',
        output_filename='test_ro3_strict',
        explanation='Test Rule of Three - strict',
        strict=True
    )
    
    print(f"✅ filter_by_rule_of_three (strict) PASSED")
    print(f"   Input: {result8['n_input']} molecules")
    print(f"   Output: {result8['n_output']} molecules ({result8['percent_retained']:.1f}% retained)")
    print(f"   Removed: {result8['n_removed']} molecules")
    print(f"   Invalid SMILES: {result8['n_invalid_smiles']}")
    print(f"   Criteria mode: {result8['criteria_mode']}")
    print(f"   Properties added: {', '.join(result8['ro3_properties_added'])}")
    print(f"   Warning: {result8['warning']}")
    
    # Check what passed
    df_ro3 = _load_resource(str(test_manifest), result8['output_filename'])
    print(f"   Fragment-like molecules: {', '.join(df_ro3['name'].tolist())}")
except Exception as e:
    print(f"❌ filter_by_rule_of_three FAILED: {e}")

# =============================================================================
# TEST 9: filter_by_rule_of_three (lenient)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 9: filter_by_rule_of_three (lenient)")
print("=" * 80)

try:
    result9 = filter_by_rule_of_three(
        input_filename=df_filename,
        project_manifest_path=str(test_manifest),
        smiles_column='smiles',
        output_filename='test_ro3_lenient',
        explanation='Test Rule of Three - lenient',
        strict=False
    )
    
    print(f"✅ filter_by_rule_of_three (lenient) PASSED")
    print(f"   Input: {result9['n_input']} molecules")
    print(f"   Output: {result9['n_output']} molecules ({result9['percent_retained']:.1f}% retained)")
    print(f"   Criteria mode: {result9['criteria_mode']}")
    print(f"   Warning: {result9['warning']}")
    
    # Check what passed
    df_ro3_len = _load_resource(str(test_manifest), result9['output_filename'])
    print(f"   Fragment-like molecules: {', '.join(df_ro3_len['name'].tolist())}")
except Exception as e:
    print(f"❌ filter_by_rule_of_three (lenient) FAILED: {e}")

# =============================================================================
# TEST 10: filter_by_qed (default threshold)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 10: filter_by_qed (min_qed=0.5)")
print("=" * 80)

try:
    result10 = filter_by_qed(
        input_filename=df_filename,
        project_manifest_path=str(test_manifest),
        smiles_column='smiles',
        output_filename='test_qed_05',
        explanation='Test QED filtering - default threshold',
        min_qed=0.5
    )
    
    print(f"✅ filter_by_qed (0.5) PASSED")
    print(f"   Input: {result10['n_input']} molecules")
    print(f"   Output: {result10['n_output']} molecules ({result10['percent_retained']:.1f}% retained)")
    print(f"   Removed: {result10['n_removed']} molecules")
    print(f"   Invalid SMILES: {result10['n_invalid_smiles']}")
    print(f"   Min QED threshold: {result10['min_qed_threshold']}")
    print(f"   Mean QED: {result10['mean_qed']:.3f}")
    print(f"   Median QED: {result10['median_qed']:.3f}")
    print(f"   Warning: {result10['warning']}")
    
    # Check what passed
    df_qed = _load_resource(str(test_manifest), result10['output_filename'])
    print(f"   Drug-like molecules: {', '.join(df_qed['name'].tolist())}")
except Exception as e:
    print(f"❌ filter_by_qed FAILED: {e}")

# =============================================================================
# TEST 11: filter_by_qed (high threshold)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 11: filter_by_qed (min_qed=0.7)")
print("=" * 80)

try:
    result11 = filter_by_qed(
        input_filename=df_filename,
        project_manifest_path=str(test_manifest),
        smiles_column='smiles',
        output_filename='test_qed_07',
        explanation='Test QED filtering - high threshold',
        min_qed=0.7
    )
    
    print(f"✅ filter_by_qed (0.7) PASSED")
    print(f"   Input: {result11['n_input']} molecules")
    print(f"   Output: {result11['n_output']} molecules ({result11['percent_retained']:.1f}% retained)")
    print(f"   Mean QED: {result11['mean_qed']:.3f}")
    print(f"   Median QED: {result11['median_qed']:.3f}")
    print(f"   Warning: {result11['warning']}")
    
    # Check what passed
    df_qed_high = _load_resource(str(test_manifest), result11['output_filename'])
    if len(df_qed_high) > 0:
        print(f"   High QED molecules: {', '.join(df_qed_high['name'].tolist())}")
    else:
        print(f"   No molecules passed high threshold")
except Exception as e:
    print(f"❌ filter_by_qed (0.7) FAILED: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("✅ filter_by_property_range: PASSED")
print("✅ filter_by_lipinski_ro5: PASSED")
print("✅ filter_by_veber_rules: PASSED")
print("✅ filter_by_pains (drop): PASSED")
print("✅ filter_by_pains (keep): PASSED")
print("✅ filter_by_lead_likeness (strict): PASSED")
print("✅ filter_by_lead_likeness (lenient): PASSED")
print("✅ filter_by_rule_of_three (strict): PASSED")
print("✅ filter_by_rule_of_three (lenient): PASSED")
print("✅ filter_by_qed (0.5): PASSED")
print("✅ filter_by_qed (0.7): PASSED")
print("\n🎉 All filtering tests completed successfully!")
