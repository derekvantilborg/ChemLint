"""
Comprehensive test suite for _analyze_functional_group_distribution().

This helper detects functional groups across splits and identifies groups unique to
each split or shared across all splits. It's intentionally lenient - many groups will
naturally be shared. Focus is on identifying groups that appear ONLY in one split.

Functional groups tested (19 total):
- Aromatic rings, Carbonyl, Carboxylic acid, Amide, Ether, Ester
- Alcohol, Ketone, Aldehyde, Amine (primary/secondary/tertiary)
- Halogen (with F/Cl/Br/I breakdown), Nitro, Sulfur-containing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from molml_mcp.tools.reports.data_splitting import _analyze_functional_group_distribution
from molml_mcp.infrastructure.resources import _store_resource


# Test manifest path
TEST_MANIFEST = Path(__file__).parent / 'data' / 'test_manifest.json'


def test_basic_structure():
    """Test that the function returns the expected structure."""
    # Create dataset with diverse functional groups
    train_data = pd.DataFrame({
        'smiles': [
            'c1ccccc1',  # aromatic
            'CC(=O)C',  # ketone
            'CCO',  # alcohol
            'CCN',  # primary amine
        ],
        'activity': [0, 1, 0, 1]
    })
    test_data = pd.DataFrame({
        'smiles': [
            'c1ccccc1C(=O)O',  # aromatic + carboxylic acid
            'CCOC',  # ether
        ],
        'activity': [1, 0]
    })
    
    # Store resources
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'fg_basic_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'fg_basic_test', 'test', 'csv')
    
    # Run analysis
    result = _analyze_functional_group_distribution(
        train_file, test_file, None, str(TEST_MANIFEST), 'smiles'
    )
    
    # Check structure
    assert 'n_functional_groups_tested' in result
    assert result['n_functional_groups_tested'] == 19
    assert 'computation_stats' in result
    assert 'train' in result
    assert 'test' in result
    assert 'counts' in result['train']
    assert 'percentages' in result['train']
    assert 'unique_to_train' in result
    assert 'unique_to_test' in result
    assert 'shared_across_all_splits' in result
    assert 'n_shared_groups' in result
    assert 'overall_severity' in result
    assert 'summary' in result
    
    print("✓ Basic structure test passed")


def test_unique_groups_detection():
    """Test detection of groups unique to each split."""
    # Train: Only aromatic compounds
    train_data = pd.DataFrame({
        'smiles': [
            'c1ccccc1',  # benzene
            'c1ccccc1C',  # toluene
            'c1ccc(cc1)C',  # more aromatics
        ],
        'activity': [0, 1, 0]
    })
    
    # Test: Only aliphatic ketones (no aromatics)
    test_data = pd.DataFrame({
        'smiles': [
            'CC(=O)C',  # acetone
            'CCC(=O)C',  # 2-butanone
            'CC(=O)CC',  # more ketones
        ],
        'activity': [1, 0, 1]
    })
    
    # Store resources
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'fg_unique_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'fg_unique_test', 'test', 'csv')
    
    # Run analysis
    result = _analyze_functional_group_distribution(
        train_file, test_file, None, str(TEST_MANIFEST), 'smiles', min_occurrence_threshold=2
    )
    
    # Check unique groups
    unique_train_names = [g['group'] for g in result['unique_to_train']]
    unique_test_names = [g['group'] for g in result['unique_to_test']]
    
    # Aromatic should be unique to train
    assert 'Aromatic rings' in unique_train_names
    
    # Ketone should be unique to test
    assert 'Ketone' in unique_test_names
    
    # Check counts
    assert result['train']['counts']['Aromatic rings'] == 3
    assert result['test']['counts']['Ketone'] == 3
    assert result['test']['counts']['Aromatic rings'] == 0
    assert result['train']['counts']['Ketone'] == 0
    
    print(f"✓ Unique groups detection: {len(unique_train_names)} unique to train, {len(unique_test_names)} unique to test")


def test_shared_groups():
    """Test detection of groups shared across all splits."""
    # Both splits have aromatic rings and alcohols
    train_data = pd.DataFrame({
        'smiles': [
            'c1ccccc1O',  # phenol
            'c1ccc(O)cc1',  # another phenol
            'c1ccccc1CO',  # benzyl alcohol
        ],
        'activity': [0, 1, 0]
    })
    test_data = pd.DataFrame({
        'smiles': [
            'c1ccc(cc1)O',  # phenol
            'c1ccccc1CCO',  # phenethyl alcohol
        ],
        'activity': [1, 0]
    })
    
    # Store resources
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'fg_shared_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'fg_shared_test', 'test', 'csv')
    
    # Run analysis
    result = _analyze_functional_group_distribution(
        train_file, test_file, None, str(TEST_MANIFEST), 'smiles'
    )
    
    # Check shared groups
    shared = result['shared_across_all_splits']
    assert 'Aromatic rings' in shared
    assert 'Alcohol' in shared
    
    # Both splits should have these groups
    assert result['train']['counts']['Aromatic rings'] > 0
    assert result['test']['counts']['Aromatic rings'] > 0
    assert result['train']['counts']['Alcohol'] > 0
    assert result['test']['counts']['Alcohol'] > 0
    
    print(f"✓ Shared groups detection: {len(shared)} groups shared across all splits")


def test_three_way_split():
    """Test with train/test/val splits."""
    # Train: Aromatic + Alcohol
    train_data = pd.DataFrame({
        'smiles': ['c1ccccc1O', 'c1ccc(O)cc1', 'c1ccccc1CO'],
        'activity': [0, 1, 0]
    })
    
    # Test: Aromatic + Ketone
    test_data = pd.DataFrame({
        'smiles': ['c1ccccc1C(=O)C', 'c1ccc(cc1)C(=O)C'],
        'activity': [1, 0]
    })
    
    # Val: Aromatic + Amine
    val_data = pd.DataFrame({
        'smiles': ['c1ccccc1N', 'c1ccc(N)cc1'],
        'activity': [0, 1]
    })
    
    # Store resources
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'fg_three_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'fg_three_test', 'test', 'csv')
    val_file = _store_resource(val_data, str(TEST_MANIFEST), 'fg_three_val', 'test', 'csv')
    
    # Run analysis
    result = _analyze_functional_group_distribution(
        train_file, test_file, val_file, str(TEST_MANIFEST), 'smiles', min_occurrence_threshold=1
    )
    
    # Check all three splits present
    assert 'val' in result
    assert 'unique_to_val' in result
    assert 'val_molecules' in result['computation_stats']
    
    # Aromatic should be shared (in all three)
    assert 'Aromatic rings' in result['shared_across_all_splits']
    
    # Alcohol unique to train, Ketone unique to test, Amine unique to val
    unique_train = [g['group'] for g in result['unique_to_train']]
    unique_test = [g['group'] for g in result['unique_to_test']]
    unique_val = [g['group'] for g in result['unique_to_val']]
    
    assert 'Alcohol' in unique_train
    assert 'Ketone' in unique_test
    assert 'Primary amine' in unique_val
    
    print(f"✓ Three-way split: {len(unique_train)} train-only, {len(unique_test)} test-only, {len(unique_val)} val-only")


def test_all_functional_groups():
    """Test dataset with many different functional groups."""
    # Create molecules with various functional groups
    train_data = pd.DataFrame({
        'smiles': [
            'c1ccccc1',  # aromatic
            'CC(=O)C',  # ketone + carbonyl
            'CC(=O)O',  # carboxylic acid
            'CC(=O)N',  # amide
            'CCOC',  # ether
            'CC(=O)OC',  # ester
            'CCO',  # alcohol
            'CC=O',  # aldehyde
            'CCN',  # primary amine
            'CC(C)N',  # secondary amine
            'CC(C)(C)N',  # tertiary amine
            'CCF',  # fluorine + halogen
            'CCCl',  # chlorine + halogen
            'CCBr',  # bromine + halogen
            'CCI',  # iodine + halogen
            'CC[N+](=O)[O-]',  # nitro
            'CCS',  # sulfur
        ],
        'activity': [0] * 17
    })
    test_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CCO'],  # aromatic + alcohol (shared)
        'activity': [1, 0]
    })
    
    # Store resources
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'fg_all_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'fg_all_test', 'test', 'csv')
    
    # Run analysis
    result = _analyze_functional_group_distribution(
        train_file, test_file, None, str(TEST_MANIFEST), 'smiles'
    )
    
    # Check that many groups were detected in train
    train_counts = result['train']['counts']
    groups_with_counts = sum(1 for count in train_counts.values() if count > 0)
    
    # Should detect at least 15 different functional groups
    assert groups_with_counts >= 15
    
    # Check specific groups
    assert train_counts['Aromatic rings'] > 0
    assert train_counts['Ketone'] > 0
    assert train_counts['Carboxylic acid'] > 0
    assert train_counts['Amide'] > 0
    assert train_counts['Ether'] > 0
    assert train_counts['Ester'] > 0
    assert train_counts['Alcohol'] > 0
    assert train_counts['Aldehyde'] > 0
    assert train_counts['Primary amine'] > 0
    assert train_counts['Halogen'] >= 4  # F, Cl, Br, I
    assert train_counts['Fluorine'] > 0
    assert train_counts['Chlorine'] > 0
    assert train_counts['Bromine'] > 0
    assert train_counts['Iodine'] > 0
    assert train_counts['Nitro'] > 0
    assert train_counts['Sulfur-containing'] > 0
    
    print(f"✓ All functional groups test: {groups_with_counts}/19 groups detected in train")


def test_min_occurrence_threshold():
    """Test min_occurrence_threshold parameter."""
    # Train has 1 ketone, test has 3 ketones
    train_data = pd.DataFrame({
        'smiles': ['CC(=O)C', 'c1ccccc1', 'c1ccccc1'],  # 1 ketone, 2 aromatics
        'activity': [0, 1, 0]
    })
    test_data = pd.DataFrame({
        'smiles': ['CC(=O)C', 'CCC(=O)C', 'CC(=O)CC'],  # 3 ketones
        'activity': [1, 0, 1]
    })
    
    # Store resources
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'fg_threshold_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'fg_threshold_test', 'test', 'csv')
    
    # Test with threshold=1 (ketone should NOT be unique because it's in both)
    result_low = _analyze_functional_group_distribution(
        train_file, test_file, None, str(TEST_MANIFEST), 'smiles', min_occurrence_threshold=1
    )
    
    # Test with threshold=2 (ketone should be unique to test because train has only 1)
    result_high = _analyze_functional_group_distribution(
        train_file, test_file, None, str(TEST_MANIFEST), 'smiles', min_occurrence_threshold=2
    )
    
    # With threshold=1, ketone is in both (not unique)
    unique_test_low = [g['group'] for g in result_low['unique_to_test']]
    
    # With threshold=2, ketone is unique to test (train has only 1 < 2)
    unique_test_high = [g['group'] for g in result_high['unique_to_test']]
    
    assert 'Ketone' not in unique_test_low  # Present in both with threshold=1
    assert 'Ketone' in unique_test_high  # Unique to test with threshold=2
    
    print(f"✓ Min occurrence threshold: threshold=2 correctly identifies unique groups")


def test_invalid_smiles():
    """Test handling of invalid SMILES."""
    train_data = pd.DataFrame({
        'smiles': [
            'c1ccccc1',  # valid aromatic
            'INVALID',  # invalid
            'CC(=O)C',  # valid ketone
            'BADSMILES',  # invalid
        ],
        'activity': [0, 1, 0, 1]
    })
    test_data = pd.DataFrame({
        'smiles': [
            'CCO',  # valid alcohol
            'NOTASMILES',  # invalid
        ],
        'activity': [1, 0]
    })
    
    # Store resources
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'fg_invalid_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'fg_invalid_test', 'test', 'csv')
    
    # Run analysis
    result = _analyze_functional_group_distribution(
        train_file, test_file, None, str(TEST_MANIFEST), 'smiles'
    )
    
    # Check computation stats
    assert result['computation_stats']['train_molecules'] == 2  # 2 valid
    assert result['computation_stats']['train_failed'] == 2  # 2 invalid
    assert result['computation_stats']['test_molecules'] == 1  # 1 valid
    assert result['computation_stats']['test_failed'] == 1  # 1 invalid
    
    # Check that valid molecules were processed
    assert result['train']['counts']['Aromatic rings'] == 1
    assert result['train']['counts']['Ketone'] == 1
    assert result['test']['counts']['Alcohol'] == 1
    
    print("✓ Invalid SMILES handled correctly: 2 train valid, 1 test valid")


def test_empty_split():
    """Test with empty test split."""
    train_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CC(=O)C', 'CCO'],
        'activity': [0, 1, 0]
    })
    test_data = pd.DataFrame({
        'smiles': [],
        'activity': []
    })
    
    # Store resources
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'fg_empty_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'fg_empty_test', 'test', 'csv')
    
    # Run analysis with threshold=1 (since we only have 1 of each group)
    result = _analyze_functional_group_distribution(
        train_file, test_file, None, str(TEST_MANIFEST), 'smiles', min_occurrence_threshold=1
    )
    
    # Check that train was processed
    assert result['computation_stats']['train_molecules'] == 3
    assert result['computation_stats']['test_molecules'] == 0
    
    # All train groups should be unique (test has none)
    train_groups_present = [g for g, count in result['train']['counts'].items() if count > 0]
    unique_train_names = [g['group'] for g in result['unique_to_train']]
    
    # All present groups in train should be unique (test is empty)
    for group in train_groups_present:
        assert group in unique_train_names
    
    print(f"✓ Empty split: {len(unique_train_names)} groups unique to train (test empty)")


def test_percentages_calculation():
    """Test that percentages are calculated correctly."""
    # 4 molecules, 2 with aromatic rings = 50%
    train_data = pd.DataFrame({
        'smiles': [
            'c1ccccc1',  # aromatic
            'c1ccccc1C',  # aromatic
            'CCO',  # no aromatic
            'CC(=O)C',  # no aromatic
        ],
        'activity': [0, 1, 0, 1]
    })
    test_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CCO'],  # 1 aromatic, 1 not = 50%
        'activity': [1, 0]
    })
    
    # Store resources
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'fg_pct_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'fg_pct_test', 'test', 'csv')
    
    # Run analysis
    result = _analyze_functional_group_distribution(
        train_file, test_file, None, str(TEST_MANIFEST), 'smiles'
    )
    
    # Check percentages
    assert result['train']['percentages']['Aromatic rings'] == 50.0
    assert result['test']['percentages']['Aromatic rings'] == 50.0
    
    # 1 alcohol out of 4 in train = 25%
    assert result['train']['percentages']['Alcohol'] == 25.0
    # 1 alcohol out of 2 in test = 50%
    assert result['test']['percentages']['Alcohol'] == 50.0
    
    print("✓ Percentages calculated correctly: 50% aromatic in both splits")


def test_severity_levels():
    """Test severity level assignment."""
    # No unique groups (all shared) -> OK
    train_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CCO', 'CC(=O)C'],
        'activity': [0, 1, 0]
    })
    test_data = pd.DataFrame({
        'smiles': ['c1ccccc1C', 'CCCO', 'CCC(=O)C'],  # Same groups
        'activity': [1, 0, 1]
    })
    
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'fg_sev_ok_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'fg_sev_ok_test', 'test', 'csv')
    
    result_ok = _analyze_functional_group_distribution(
        train_file, test_file, None, str(TEST_MANIFEST), 'smiles'
    )
    
    # Should be OK (no unique groups or very few)
    assert result_ok['overall_severity'] in ['OK', 'LOW']
    
    # Create dataset with many unique groups in test -> MEDIUM
    train_diverse = pd.DataFrame({
        'smiles': ['c1ccccc1'] * 5,  # Only aromatics
        'activity': [0] * 5
    })
    test_diverse = pd.DataFrame({
        'smiles': [
            'CC(=O)C',  # ketone
            'CC(=O)O',  # carboxylic acid
            'CC(=O)N',  # amide
            'CCOC',  # ether
            'CC(=O)OC',  # ester
            'CCN',  # amine
        ],
        'activity': [1] * 6
    })
    
    train_file2 = _store_resource(train_diverse, str(TEST_MANIFEST), 'fg_sev_med_train', 'test', 'csv')
    test_file2 = _store_resource(test_diverse, str(TEST_MANIFEST), 'fg_sev_med_test', 'test', 'csv')
    
    result_medium = _analyze_functional_group_distribution(
        train_file2, test_file2, None, str(TEST_MANIFEST), 'smiles', min_occurrence_threshold=1
    )
    
    # Should be MEDIUM (>3 unique groups in test)
    assert result_medium['overall_severity'] == 'MEDIUM'
    assert len(result_medium['unique_to_test']) > 3
    
    print(f"✓ Severity levels: OK for balanced, MEDIUM for {len(result_medium['unique_to_test'])} unique test groups")


def test_summary_statistics():
    """Test summary statistics."""
    train_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CC(=O)C', 'CCO'],
        'activity': [0, 1, 0]
    })
    test_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CCN'],  # aromatic shared, amine unique
        'activity': [1, 0]
    })
    
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'fg_summary_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'fg_summary_test', 'test', 'csv')
    
    result = _analyze_functional_group_distribution(
        train_file, test_file, None, str(TEST_MANIFEST), 'smiles', min_occurrence_threshold=1
    )
    
    # Check summary
    assert 'summary' in result
    assert 'n_unique_to_train' in result['summary']
    assert 'n_unique_to_test' in result['summary']
    assert 'n_shared' in result['summary']
    assert 'pct_shared' in result['summary']
    
    # Check counts match
    assert result['summary']['n_unique_to_train'] == len(result['unique_to_train'])
    assert result['summary']['n_unique_to_test'] == len(result['unique_to_test'])
    assert result['summary']['n_shared'] == len(result['shared_across_all_splits'])
    
    # Check percentage
    expected_pct = len(result['shared_across_all_splits']) / 19 * 100
    assert abs(result['summary']['pct_shared'] - expected_pct) < 0.1
    
    print(f"✓ Summary: {result['summary']['n_shared']} shared ({result['summary']['pct_shared']}%)")


def test_halogen_breakdown():
    """Test individual halogen detection (F, Cl, Br, I)."""
    train_data = pd.DataFrame({
        'smiles': [
            'CCF',  # fluorine
            'CCCl',  # chlorine
            'CCBr',  # bromine
            'CCI',  # iodine
        ],
        'activity': [0, 1, 0, 1]
    })
    test_data = pd.DataFrame({
        'smiles': ['CCCC'],  # no halogens
        'activity': [0]
    })
    
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'fg_halogen_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'fg_halogen_test', 'test', 'csv')
    
    result = _analyze_functional_group_distribution(
        train_file, test_file, None, str(TEST_MANIFEST), 'smiles'
    )
    
    # Check individual halogens
    assert result['train']['counts']['Fluorine'] == 1
    assert result['train']['counts']['Chlorine'] == 1
    assert result['train']['counts']['Bromine'] == 1
    assert result['train']['counts']['Iodine'] == 1
    assert result['train']['counts']['Halogen'] == 4  # Total halogens
    
    # Test should have none
    assert result['test']['counts']['Halogen'] == 0
    assert result['test']['counts']['Fluorine'] == 0
    assert result['test']['counts']['Chlorine'] == 0
    assert result['test']['counts']['Bromine'] == 0
    assert result['test']['counts']['Iodine'] == 0
    
    print("✓ Halogen breakdown: F=1, Cl=1, Br=1, I=1, Total=4")


def test_amine_types():
    """Test primary/secondary/tertiary amine detection."""
    train_data = pd.DataFrame({
        'smiles': [
            'CCN',  # primary amine (NH2)
            'CC(C)NC',  # secondary amine (NH)
            'CC(C)N(C)C',  # tertiary amine (N)
        ],
        'activity': [0, 1, 0]
    })
    test_data = pd.DataFrame({
        'smiles': ['CCCC'],  # no amines
        'activity': [0]
    })
    
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'fg_amine_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'fg_amine_test', 'test', 'csv')
    
    result = _analyze_functional_group_distribution(
        train_file, test_file, None, str(TEST_MANIFEST), 'smiles'
    )
    
    # Check amine types
    assert result['train']['counts']['Primary amine'] == 1
    assert result['train']['counts']['Secondary amine'] == 1
    assert result['train']['counts']['Tertiary amine'] == 1
    
    # Test should have none
    assert result['test']['counts']['Primary amine'] == 0
    assert result['test']['counts']['Secondary amine'] == 0
    assert result['test']['counts']['Tertiary amine'] == 0
    
    print("✓ Amine types: Primary=1, Secondary=1, Tertiary=1")


def test_consistency():
    """Test that results are consistent across runs."""
    train_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CC(=O)C', 'CCO', 'CCN'],
        'activity': [0, 1, 0, 1]
    })
    test_data = pd.DataFrame({
        'smiles': ['c1ccccc1C', 'CCC(=O)C'],
        'activity': [1, 0]
    })
    
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'fg_consist_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'fg_consist_test', 'test', 'csv')
    
    # Run twice
    result1 = _analyze_functional_group_distribution(
        train_file, test_file, None, str(TEST_MANIFEST), 'smiles'
    )
    result2 = _analyze_functional_group_distribution(
        train_file, test_file, None, str(TEST_MANIFEST), 'smiles'
    )
    
    # Check consistency
    assert result1['train']['counts'] == result2['train']['counts']
    assert result1['test']['counts'] == result2['test']['counts']
    assert result1['train']['percentages'] == result2['train']['percentages']
    assert result1['test']['percentages'] == result2['test']['percentages']
    assert len(result1['unique_to_train']) == len(result2['unique_to_train'])
    assert len(result1['unique_to_test']) == len(result2['unique_to_test'])
    assert len(result1['shared_across_all_splits']) == len(result2['shared_across_all_splits'])
    assert result1['overall_severity'] == result2['overall_severity']
    
    print("✓ Consistency test passed: identical results across runs")


def test_unique_group_details():
    """Test that unique group details include all expected fields."""
    train_data = pd.DataFrame({
        'smiles': ['CC(=O)C', 'CCC(=O)C', 'CC(=O)CC'],  # Only ketones
        'activity': [0, 1, 0]
    })
    test_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'c1ccccc1C'],  # Only aromatics
        'activity': [1, 0]
    })
    
    train_file = _store_resource(train_data, str(TEST_MANIFEST), 'fg_details_train', 'test', 'csv')
    test_file = _store_resource(test_data, str(TEST_MANIFEST), 'fg_details_test', 'test', 'csv')
    
    result = _analyze_functional_group_distribution(
        train_file, test_file, None, str(TEST_MANIFEST), 'smiles', min_occurrence_threshold=2
    )
    
    # Check unique group structure
    for group_info in result['unique_to_train']:
        assert 'group' in group_info
        assert 'count' in group_info
        assert 'pct_molecules' in group_info
        assert isinstance(group_info['count'], (int, np.integer))
        assert isinstance(group_info['pct_molecules'], (float, np.floating))
    
    for group_info in result['unique_to_test']:
        assert 'group' in group_info
        assert 'count' in group_info
        assert 'pct_molecules' in group_info
    
    # Check specific values
    ketone_info = next(g for g in result['unique_to_train'] if g['group'] == 'Ketone')
    assert ketone_info['count'] == 3
    assert ketone_info['pct_molecules'] == 100.0  # All 3 molecules
    
    aromatic_info = next(g for g in result['unique_to_test'] if g['group'] == 'Aromatic rings')
    assert aromatic_info['count'] == 2
    assert aromatic_info['pct_molecules'] == 100.0  # All 2 molecules
    
    print("✓ Unique group details: All fields present and correct")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Testing _analyze_functional_group_distribution()")
    print("="*70 + "\n")
    
    test_basic_structure()
    test_unique_groups_detection()
    test_shared_groups()
    test_three_way_split()
    test_all_functional_groups()
    test_min_occurrence_threshold()
    test_invalid_smiles()
    test_empty_split()
    test_percentages_calculation()
    test_severity_levels()
    test_summary_statistics()
    test_halogen_breakdown()
    test_amine_types()
    test_consistency()
    test_unique_group_details()
    
    print("\n" + "="*70)
    print("✓ All 15 tests passed!")
    print("="*70)
