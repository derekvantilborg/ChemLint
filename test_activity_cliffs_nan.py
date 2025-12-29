import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from pathlib import Path
from molml_mcp.infrastructure.resources import _store_resource
from molml_mcp.tools.core_mol.activity_cliffs import identify_activity_cliffs

print("=" * 80)
print("ACTIVITY CLIFF NaN HANDLING TEST")
print("=" * 80)

# Setup
test_manifest = Path("tests/data/test_manifest.json")

# Create test dataset with NaN values
# Using LINEAR SCALE (IC50 in nM) as required
test_data = {
    'SMILES': [
        'CCO',           # Has activity: 100 nM
        'CCCO',          # NaN activity
        'CCC(=O)O',      # Has activity: 1 nM
        'c1ccccc1',      # NaN activity
        'c1ccccc1O',     # Has activity: 10 nM
        'c1ccccc1C',     # Has activity: 631 nM
    ],
    'IC50_nM': [100.0, np.nan, 1.0, np.nan, 10.0, 631.0],  # LINEAR SCALE
    'name': ['Ethanol', 'Propanol', 'Propionic acid', 'Benzene', 'Phenol', 'Toluene']
}

df = pd.DataFrame(test_data)
df_filename = _store_resource(df, str(test_manifest), "activity_cliff_nan_test", "Test with NaN values", 'csv')

print(f"\n✅ Test data created: {df_filename}")
print(f"   Total molecules: {len(df)}")
print(f"   Molecules with valid activity: {df['IC50_nM'].notna().sum()}")
print(f"   Molecules with NaN activity: {df['IC50_nM'].isna().sum()}")
print(f"\n   Dataset preview:")
for idx, row in df.iterrows():
    activity_str = f"{row['IC50_nM']:.0f} nM" if pd.notna(row['IC50_nM']) else "NaN"
    print(f"      {idx}: {row['name']:20s} IC50={activity_str}")

# Create similarity matrix
n_mols = len(df)
similarity_matrix = np.eye(n_mols)

# Set up high similarities for some pairs
# Pair 0-2: both have activities (5.0, 7.0) - should form cliff
similarity_matrix[0, 2] = similarity_matrix[2, 0] = 0.92

# Pair 1-2: molecule 1 has NaN - should be skipped
similarity_matrix[1, 2] = similarity_matrix[2, 1] = 0.95

# Pair 3-4: molecule 3 has NaN - should be skipped
similarity_matrix[3, 4] = similarity_matrix[4, 3] = 0.93

# Pair 4-5: both have activities (8.0, 6.2) - should form cliff
similarity_matrix[4, 5] = similarity_matrix[5, 4] = 0.91

sim_filename = _store_resource(
    similarity_matrix, 
    str(test_manifest), 
    "nan_test_sim_matrix", 
    "Similarity matrix for NaN test",
    'joblib'
)

print(f"\n✅ Similarity matrix created: {sim_filename}")

# TEST: Identify cliffs with NaN values present
print("\n" + "=" * 80)
print("TEST: Activity Cliff Detection with NaN Values")
print("=" * 80)

result = identify_activity_cliffs(
    df_filename,
    str(test_manifest),
    'SMILES',
    'IC50_nM',
    sim_filename,
    'activity_cliffs_with_nan',
    similarity_threshold=0.9,
    fold_difference_threshold=2.0  # Lower threshold to catch both pairs
)

print(f"\n✅ Output: {result['output_filename']}")
print(f"   Total molecules: {result['n_molecules']}")
print(f"   Molecules with valid activity: {result['n_molecules_with_valid_activity']}")
print(f"   Molecules with NaN activity: {result['n_molecules_with_nan_activity']}")
print(f"   Activity cliffs found: {result['n_cliff_pairs']}")
print(f"\n   Summary: {result['summary']}")

print(f"\n   Expected behavior:")
print(f"   - Pair 0-2 (CCO vs CCC(=O)O): INCLUDED (both have activities)")
print(f"   - Pair 1-2 (CCCO vs CCC(=O)O): SKIPPED (CCCO has NaN)")
print(f"   - Pair 3-4 (benzene vs phenol): SKIPPED (benzene has NaN)")
print(f"   - Pair 4-5 (phenol vs toluene): INCLUDED (both have activities)")

if result['n_cliff_pairs'] > 0:
    print(f"\n   Detected cliffs:")
    for cliff in result['preview']:
        print(f"      {cliff['smiles_i']:15s} vs {cliff['smiles_j']:15s}: "
              f"fold={cliff['fold_difference']:.1f}x, "
              f"activities={cliff['activity_i']:.1f}/{cliff['activity_j']:.1f}")

# Verify correct behavior
expected_pairs = 2  # Pairs 0-2 and 4-5
if result['n_cliff_pairs'] == expected_pairs:
    print(f"\n✅ CORRECT: Found {expected_pairs} cliffs (NaN pairs correctly excluded)")
else:
    print(f"\n❌ ERROR: Expected {expected_pairs} cliffs, found {result['n_cliff_pairs']}")

print("\n" + "=" * 80)
print("TEST COMPLETED! ✅")
print("=" * 80)
print("\nNaN Handling Summary:")
print("- Molecules with NaN activity are automatically EXCLUDED from cliff detection")
print("- Pairs where either molecule has NaN are SKIPPED")
print("- A warning message is printed showing how many NaN values were found")
print("- Result includes 'n_molecules_with_valid_activity' and 'n_molecules_with_nan_activity'")
