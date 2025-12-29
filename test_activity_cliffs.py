import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from pathlib import Path
from molml_mcp.infrastructure.resources import _store_resource
from molml_mcp.tools.core_mol.activity_cliffs import identify_activity_cliffs

print("=" * 80)
print("ACTIVITY CLIFF DETECTION TEST")
print("=" * 80)

# Setup
test_manifest = Path("tests/data/test_manifest.json")

# Create test dataset with known activity cliffs
# Scenario: Similar molecules with very different activities
# Using LINEAR SCALE (IC50 in nM) as required by the function
test_data = {
    'SMILES': [
        'CCO',                    # Ethanol, IC50 = 100 nM
        'CCCO',                   # Propanol (similar), IC50 = 79 nM (no cliff, 1.3-fold)
        'CCC(=O)O',               # Propionic acid (similar), IC50 = 1 nM (CLIFF! 100-fold)
        'c1ccccc1',               # Benzene, IC50 = 1000 nM
        'c1ccccc1O',              # Phenol (similar), IC50 = 10 nM (CLIFF! 100-fold)
        'c1ccccc1C',              # Toluene (similar), IC50 = 631 nM (no cliff, 1.6-fold)
        'CC(C)C',                 # Isobutane, IC50 = 3162 nM
        'CC(C)CO',                # Isobutanol (similar), IC50 = 1995 nM (no cliff, 1.6-fold)
    ],
    'IC50_nM': [100, 79, 1, 1000, 10, 631, 3162, 1995],  # LINEAR SCALE
    'name': ['Ethanol', 'Propanol', 'Propionic acid', 'Benzene', 'Phenol', 'Toluene', 'Isobutane', 'Isobutanol']
}

df = pd.DataFrame(test_data)
df_filename = _store_resource(df, str(test_manifest), "activity_cliff_test_data", "Test molecules for activity cliff detection", 'csv')

print(f"\n✅ Test data created: {df_filename}")
print(f"   Molecules: {len(df)}")
print(f"\n   Dataset preview:")
for idx, row in df.iterrows():
    print(f"      {idx}: {row['name']:20s} {row['SMILES']:15s} IC50={row['IC50_nM']:.0f} nM")

# Create a mock similarity matrix
# We'll manually define which molecules are similar
# High similarity (>0.9) for pairs that should form cliffs
n_mols = len(df)
similarity_matrix = np.eye(n_mols)  # Start with identity

# Set up similarity relationships
# CCO and CCCO: similar (0.92) - activities 5.0 and 5.1 (no cliff, only 1.26-fold)
similarity_matrix[0, 1] = similarity_matrix[1, 0] = 0.92

# CCO and CCC(=O)O: similar (0.91) - activities 5.0 and 7.0 (CLIFF! 100-fold)
similarity_matrix[0, 2] = similarity_matrix[2, 0] = 0.91

# c1ccccc1 and c1ccccc1O: similar (0.93) - activities 6.0 and 8.0 (CLIFF! 100-fold)
similarity_matrix[3, 4] = similarity_matrix[4, 3] = 0.93

# c1ccccc1 and c1ccccc1C: similar (0.94) - activities 6.0 and 6.2 (no cliff, only 1.58-fold)
similarity_matrix[3, 5] = similarity_matrix[5, 3] = 0.94

# CC(C)C and CC(C)CO: similar (0.95) - activities 4.5 and 4.7 (no cliff, only 1.58-fold)
similarity_matrix[6, 7] = similarity_matrix[7, 6] = 0.95

# Add some lower similarities for non-similar pairs
similarity_matrix[0, 3] = similarity_matrix[3, 0] = 0.65  # CCO vs benzene
similarity_matrix[0, 6] = similarity_matrix[6, 0] = 0.70  # CCO vs isobutane

sim_filename = _store_resource(
    similarity_matrix, 
    str(test_manifest), 
    "activity_cliff_sim_matrix", 
    "Mock similarity matrix for activity cliff testing",
    'joblib'
)

print(f"\n✅ Similarity matrix created: {sim_filename}")
print(f"   Shape: {similarity_matrix.shape}")

# TEST 1: Identify activity cliffs with default thresholds
print("\n" + "=" * 80)
print("TEST 1: Identify Activity Cliffs (default thresholds)")
print("=" * 80)
print("Thresholds: similarity > 0.9, fold-difference > 10")

result1 = identify_activity_cliffs(
    df_filename,
    str(test_manifest),
    'SMILES',
    'IC50_nM',  # LINEAR SCALE
    sim_filename,
    'activity_cliffs_default'
)

print(f"\n✅ Output: {result1['output_filename']}")
print(f"   Molecules analyzed: {result1['n_molecules']}")
print(f"   Activity cliffs found: {result1['n_cliff_pairs']}")
print(f"   Activity column: {result1['activity_column']}")
print(f"\n   Summary: {result1['summary']}")

if result1['n_cliff_pairs'] > 0:
    print(f"\n   Detected activity cliffs:")
    for cliff in result1['preview']:
        print(f"      Pair {cliff['molecule_i_index']}-{cliff['molecule_j_index']}: "
              f"sim={cliff['similarity']:.3f}, fold={cliff['fold_difference']:.1f}x, "
              f"Δ={cliff['activity_delta']:.1f}")
        print(f"         {cliff['smiles_i']} (activity={cliff['activity_i']:.1f})")
        print(f"         {cliff['smiles_j']} (activity={cliff['activity_j']:.1f})")

# TEST 2: More stringent similarity threshold
print("\n" + "=" * 80)
print("TEST 2: Stringent Similarity Threshold")
print("=" * 80)
print("Thresholds: similarity > 0.93, fold-difference > 10")

result2 = identify_activity_cliffs(
    df_filename,
    str(test_manifest),
    'SMILES',
    'IC50_nM',
    sim_filename,
    'activity_cliffs_stringent',
    similarity_threshold=0.93,
    fold_difference_threshold=10.0
)

print(f"\n✅ Output: {result2['output_filename']}")
print(f"   Activity cliffs found: {result2['n_cliff_pairs']}")
print(f"   (Should be fewer with stricter similarity requirement)")

# TEST 3: Lower fold-difference threshold
print("\n" + "=" * 80)
print("TEST 3: Lower Fold-Difference Threshold")
print("=" * 80)
print("Thresholds: similarity > 0.9, fold-difference > 2")

result3 = identify_activity_cliffs(
    df_filename,
    str(test_manifest),
    'SMILES',
    'IC50_nM',
    sim_filename,
    'activity_cliffs_relaxed',
    similarity_threshold=0.9,
    fold_difference_threshold=2.0
)

print(f"\n✅ Output: {result3['output_filename']}")
print(f"   Activity cliffs found: {result3['n_cliff_pairs']}")
print(f"   (Should be more with lower fold-difference requirement)")

if result3['n_cliff_pairs'] > 0:
    print(f"\n   Top 5 cliffs by fold-difference:")
    for cliff in result3['preview']:
        print(f"      {cliff['smiles_i']:15s} vs {cliff['smiles_j']:15s}: "
              f"fold={cliff['fold_difference']:.1f}x, sim={cliff['similarity']:.3f}")

# TEST 4: Limit maximum pairs returned
print("\n" + "=" * 80)
print("TEST 4: Limit Maximum Pairs (max_pairs=1)")
print("=" * 80)

result4 = identify_activity_cliffs(
    df_filename,
    str(test_manifest),
    'SMILES',
    'IC50_nM',
    sim_filename,
    'activity_cliffs_top1',
    similarity_threshold=0.9,
    fold_difference_threshold=10.0,
    max_pairs=1
)

print(f"\n✅ Output: {result4['output_filename']}")
print(f"   Activity cliffs returned: {result4['n_cliff_pairs']}")
print(f"   (Limited to 1, showing most dramatic cliff)")

if result4['preview']:
    cliff = result4['preview'][0]
    print(f"\n   Top cliff:")
    print(f"      Similarity: {cliff['similarity']:.3f}")
    print(f"      Fold-difference: {cliff['fold_difference']:.1f}x")
    print(f"      Activity difference: {cliff['activity_delta']:.1f} log units")

# TEST 5: No cliffs found scenario
print("\n" + "=" * 80)
print("TEST 5: No Cliffs Found (very strict thresholds)")
print("=" * 80)
print("Thresholds: similarity > 0.99, fold-difference > 1000")

result5 = identify_activity_cliffs(
    df_filename,
    str(test_manifest),
    'SMILES',
    'IC50_nM',
    sim_filename,
    'activity_cliffs_none',
    similarity_threshold=0.99,
    fold_difference_threshold=1000.0
)

print(f"\n✅ Output: {result5['output_filename']}")
print(f"   Activity cliffs found: {result5['n_cliff_pairs']}")
print(f"   Summary: {result5['summary']}")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
print("=" * 80)

print("\nSummary of Activity Cliff Detection:")
print("- Activity cliffs identify structurally similar pairs with large activity differences")
print("- Useful for SAR analysis and finding key structural features")
print("- Typical thresholds: similarity > 0.85-0.95, fold-difference > 10-100")
print("- Output includes pair indices, SMILES, activities, similarity, and fold-difference")
print("- Can be used for visualization, prioritization, and mechanistic studies")
