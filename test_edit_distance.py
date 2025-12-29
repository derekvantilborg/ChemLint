#!/usr/bin/env python
"""Test the edit_distance similarity metric"""

from molml_mcp.tools.core_mol.similarity import compute_similarity_matrix
from molml_mcp.infrastructure.resources import _store_resource, _load_resource
import pandas as pd
from pathlib import Path

test_manifest = Path("tests/data/test_manifest.json")

# Create test data
test_smiles = [
    "CCO",           # Ethanol
    "CCCO",          # Propanol (1 char longer)
    "CCC(=O)O",      # Propionic acid (more different)
    "c1ccccc1",      # Benzene (completely different)
]
df = pd.DataFrame({
    "SMILES": test_smiles, 
    "name": ["Ethanol", "Propanol", "Propionic acid", "Benzene"]
})

# Store test dataset
df_filename = _store_resource(
    df, str(test_manifest), "edit_dist_test", "Test for edit distance", "csv"
)
print(f"Created test dataset: {df_filename}")

# Compute edit distance similarity (no feature vectors needed!)
result = compute_similarity_matrix(
    df_filename,
    str(test_manifest),
    "SMILES",
    "",  # Not used for edit_distance
    "edit_dist_similarity",
    "Edit distance similarity test",
    similarity_metric="edit_distance"
)

print(f"\n✅ Success! Computed {result['matrix_shape']} similarity matrix")
print(f"   Metric: {result['similarity_metric']}")
print(f"   Mean similarity: {result['mean_similarity']:.3f}")
print(f"   Min/Max: {result['min_similarity']:.3f} / {result['max_similarity']:.3f}")

# Load and show matrix
sim_matrix = _load_resource(str(test_manifest), result["output_filename"])
print(f"\n📊 Similarity matrix:")
for i, smi1 in enumerate(test_smiles):
    for j, smi2 in enumerate(test_smiles):
        if i <= j:
            print(f"  {smi1:15s} vs {smi2:15s}: {sim_matrix[i,j]:.3f}")

# Verify expected properties
print("\n🔍 Verification:")
print(f"   Diagonal all 1.0? {all(sim_matrix[i,i] == 1.0 for i in range(len(test_smiles)))}")
print(f"   Symmetric? {(sim_matrix == sim_matrix.T).all()}")
print(f"   All values in [0,1]? {(sim_matrix >= 0).all() and (sim_matrix <= 1).all()}")
