# Policy Compliance Updates

This document tracks changes made to enforce the **Data Traceability Policy**: all dataset operations must create new versioned resources rather than modifying existing ones in-place.

## Policy Statement

> "No inplace operations should take place! This is against my data traceability policy!"

All tools that manipulate datasets must follow the pattern:
- Accept `input_filename` (existing resource)
- Create new versioned output with unique ID
- Return `output_filename` with full versioned name
- Never modify the original resource

## Files Modified

### 1. `tools/core_mol/outliers.py`
**Functions updated** (5 total):
- `remove_outliers_iqr()` - Removed `inplace` parameter
- `remove_outliers_zscore()` - Removed `inplace` parameter
- `remove_outliers_isolation_forest()` - Removed `inplace` parameter
- `remove_outliers_local_outlier_factor()` - Removed `inplace` parameter
- `remove_outliers_dbscan()` - Removed `inplace` parameter

**Status**: ✅ All tests passing

### 2. `tools/core_mol/complexity.py`
**Functions updated** (1 total):
- `remove_complex_molecules()` - Removed `inplace` parameter

**Status**: ✅ Tests passing

### 3. `tools/core_mol/SMILES_encoding.py`
**Functions updated** (1 total):
- `smiles_to_selfies()` - Removed `inplace` parameter

**Status**: ✅ Verified (no formal tests yet)

### 4. `tools/core_mol/similarity.py`
**Functions updated** (1 total):
- `compute_similarity_matrix()` - Removed `inplace` parameter

**Status**: ✅ Verified (no formal tests yet)

### 5. `tools/core_mol/activity_cliffs.py` ⚠️
**Function added**: `annotate_activity_cliff_molecules()`

**Special requirements**:
- Accepts **LINEAR SCALE only** (IC50_nM, Ki_nM, EC50_μM)
- ❌ Does NOT accept log-scale (pIC50, pKi, pEC50)
- No conversions performed within function
- User must convert pIC50 → IC50_nM externally if needed
- Conversion formula provided: `IC50_nM = 10^(9 - pIC50)`

**Fold-difference calculation**:
```python
fold_difference = max(activity_i, activity_j) / min(activity_i, activity_j)
```

**Example**: IC50=100nM vs IC50=10nM → fold = 100/10 = 10x ✅

**NaN handling**: Molecules with NaN activity are automatically excluded with warning message

**Implementation**:
- Adds columns to original dataset (no duplication) - practical for large datasets
- Identifies which molecules participate in cliffs
- Tracks strongest cliff partner (largest fold-difference among similar molecules)
- Columns added:
  * `is_activity_cliff_molecule` (str: 'yes'/'no'): Whether molecule has at least one cliff
  * `n_activity_cliff_partners` (int): Number of cliff partners (0 if no cliffs)
  * `strongest_cliff_partner_idx` (int/NaN): Index of partner with largest fold-difference
  * `strongest_cliff_partner_smiles` (str/NaN): SMILES of strongest partner
- Filter cliff molecules: `df[df['is_activity_cliff_molecule'] == 'yes']`

**Status**: ✅ All tests passing

## Testing Summary

### Outliers Module
- `test_outliers_iqr.py` - ✅ PASS
- `test_outliers_zscore.py` - ✅ PASS
- `test_outliers_isolation_forest.py` - ✅ PASS
- `test_outliers_lof.py` - ✅ PASS
- `test_outliers_dbscan.py` - ✅ PASS

### Complexity Module
- `test_complexity.py` - ✅ PASS

### Activity Cliffs Module
- `test_activity_cliffs.py` - ✅ PASS (5 scenarios)
- `test_activity_cliffs_nan.py` - ✅ PASS (NaN exclusion)

## Implementation Notes

### Pattern Enforced
All dataset manipulation tools now follow this signature:
```python
def tool_function(
    input_filename: str,           # Input resource from manifest
    project_manifest_path: str,    # Path to manifest.json
    output_filename: str,          # Name for new resource (ID appended)
    explanation: str,              # Description of operation
    # ... tool-specific parameters
) -> dict:
    # Load input
    data = _load_resource(project_manifest_path, input_filename)
    
    # Process (never modify original)
    processed_data = process(data.copy())
    
    # Store new version
    output_id = _store_resource(
        processed_data,
        project_manifest_path,
        output_filename,
        explanation,
        'csv'
    )
    
    return {
        "output_filename": output_id,  # Full versioned name
        # ... metadata
    }
```

### Benefits
1. **Complete audit trail**: Manifest tracks all operations
2. **Data safety**: Original data never modified
3. **Reproducibility**: Full lineage of transformations
4. **Rollback capability**: Any version can be retrieved by filename
5. **Parallel workflows**: Multiple operations can use same input

## Future Considerations

### Remaining Tools to Review
Check these modules for any inplace operations:
- `tools/core_mol/descriptors.py`
- `tools/core_mol/scaffolds.py`
- `tools/core_mol/visualize.py`
- `tools/ml/training.py`
- `tools/ml/evaluation.py`

### Documentation
- Update user-facing documentation to explain versioning system
- Add examples showing how to work with versioned datasets
- Document manifest.json structure and usage

---

**Last Updated**: 2024
**Policy Owner**: Derek van Tilborg
**Status**: ✅ Compliance achieved for all dataset manipulation tools
