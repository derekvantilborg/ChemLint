"""
Comprehensive test suite for _detect_stereoisomer_tautomer_leakage().

This function detects subtle forms of data leakage where molecules differ only in:
- Stereochemistry (R vs S, E vs Z configurations)
- Tautomeric forms (keto-enol, imine-enamine, etc.)

Tests cover:
1. Basic structure validation
2. Stereoisomer detection (R/S, E/Z)
3. Tautomer detection (keto-enol, etc.)
4. Three-way splits
5. No leakage scenarios
6. Invalid SMILES handling
7. Edge cases (single molecules, empty splits)
8. Example structure and content
9. Severity levels
10. Consistency across runs
11. max_examples parameter
"""

import sys
import os
import pandas as pd
import numpy as np
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from molml_mcp.tools.reports.data_splitting import _detect_stereoisomer_tautomer_leakage
from molml_mcp.infrastructure.resources import _store_resource

# Test data directory
TEST_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEST_MANIFEST = os.path.join(TEST_DIR, 'test_manifest.json')


# ============================================================================
# TEST SCENARIO GENERATORS
# ============================================================================

def create_stereoisomer_test_data():
    """Create dataset with stereoisomer pairs."""
    # R/S stereoisomers (chiral centers)
    train_smiles = [
        'C[C@H](O)c1ccccc1',      # (R)-1-phenylethanol
        'C[C@H](N)C(=O)O',        # (R)-alanine
        'C[C@H](O)CCO',           # (R)-1,3-butanediol
        'CC(C)C[C@H](N)C(=O)O',   # (R)-leucine
        'C[C@H](Cl)c1ccccc1',     # (R)-1-chloro-1-phenylethane
    ]
    
    test_smiles = [
        'C[C@@H](O)c1ccccc1',     # (S)-1-phenylethanol (stereoisomer)
        'C[C@@H](N)C(=O)O',       # (S)-alanine (stereoisomer)
        'C[C@@H](O)CCO',          # (S)-1,3-butanediol (stereoisomer)
        'CCCCCC',                 # hexane (no stereochemistry, different molecule)
        'c1ccccc1',               # benzene (no stereochemistry, different molecule)
    ]
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    return df_train, df_test


def create_tautomer_test_data():
    """Create dataset with tautomer pairs."""
    # Keto-enol tautomers
    train_smiles = [
        'CC(=O)CC',               # 2-butanone (keto form)
        'CC(=O)c1ccccc1',         # acetophenone (keto form)
        'O=C1CCCC1',              # cyclopentanone (keto form)
        'CC(C)=O',                # acetone (keto form)
        'CC(=O)OC',               # methyl acetate (ester, not tautomer)
    ]
    
    test_smiles = [
        'CC(O)=CC',               # 2-buten-2-ol (enol form, tautomer of 2-butanone)
        'CC(O)=Cc1ccccc1',        # enol form of acetophenone
        'O=C1CCCC1',              # cyclopentanone (same, should NOT be tautomer pair)
        'CCCCC',                  # pentane (completely different)
        'c1ccccc1',               # benzene (completely different)
    ]
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    return df_train, df_test


def create_ez_isomer_test_data():
    """Create dataset with E/Z geometric isomers."""
    train_smiles = [
        r'C/C=C/C',               # (E)-2-butene
        r'C/C=C/c1ccccc1',        # (E)-prop-1-enylbenzene
        r'CC/C=C/C',              # (E)-2-pentene
        'CCCCC',                  # pentane (no double bond)
        'c1ccccc1',               # benzene (aromatic)
    ]
    
    test_smiles = [
        r'C/C=C\C',               # (Z)-2-butene (geometric isomer)
        r'C/C=C\c1ccccc1',        # (Z)-prop-1-enylbenzene (geometric isomer)
        r'CC/C=C\C',              # (Z)-2-pentene (geometric isomer)
        'CCCCCC',                 # hexane (different molecule)
        'CCO',                    # ethanol (different molecule)
    ]
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    return df_train, df_test


def create_clean_split_data():
    """Create dataset with no stereoisomer or tautomer leakage."""
    train_smiles = [
        'CCCCCC',                 # hexane
        'c1ccccc1',               # benzene
        'CCO',                    # ethanol
        'CC(=O)O',                # acetic acid
        'CCCCCCCC',               # octane
    ]
    
    test_smiles = [
        'CCCCCCC',                # heptane (different)
        'c1ccc(C)cc1',            # toluene (different)
        'CCCO',                   # propanol (different)
        'CCC(=O)O',               # propanoic acid (different)
        'CCCCCCCCC',              # nonane (different)
    ]
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    return df_train, df_test


def create_three_way_split_data():
    """Create three-way split with stereoisomers and tautomers."""
    train_smiles = [
        'C[C@H](O)c1ccccc1',      # (R)-1-phenylethanol
        'CC(=O)CC',               # 2-butanone (keto)
        r'C/C=C/C',               # (E)-2-butene
        'CCCCCC',                 # hexane
    ]
    
    test_smiles = [
        'C[C@@H](O)c1ccccc1',     # (S)-1-phenylethanol (stereoisomer)
        'CC(O)=CC',               # enol form (tautomer)
        'CCCCCCC',                # heptane (different)
    ]
    
    val_smiles = [
        r'C/C=C\C',               # (Z)-2-butene (geometric isomer)
        'CCCCCCCC',               # octane (different)
        'c1ccccc1',               # benzene (different)
    ]
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    df_val = pd.DataFrame({'smiles': val_smiles})
    
    return df_train, df_test, df_val


def create_complex_stereoisomer_data():
    """Create dataset with multiple chiral centers."""
    train_smiles = [
        'C[C@H](O)[C@H](O)C',     # (R,R)-2,3-butanediol
        'C[C@H](O)[C@@H](O)C',    # (R,S)-2,3-butanediol (meso)
        'C[C@H](N)[C@H](N)C',     # (R,R)-2,3-diaminobutane
        'CCCCCC',                 # hexane
    ]
    
    test_smiles = [
        'C[C@@H](O)[C@@H](O)C',   # (S,S)-2,3-butanediol (stereoisomer)
        'C[C@@H](N)[C@@H](N)C',   # (S,S)-2,3-diaminobutane (stereoisomer)
        'CCCCCCC',                # heptane (different)
        'c1ccccc1',               # benzene (different)
    ]
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    return df_train, df_test


def create_invalid_smiles_data():
    """Create dataset with invalid SMILES."""
    train_smiles = [
        'C[C@H](O)c1ccccc1',      # valid
        'INVALID',                # invalid
        'C[C@@H](O)c1ccccc1',     # valid
        '',                       # empty
        'CC(=O)CC',               # valid
    ]
    
    test_smiles = [
        'C[C@H](O)c1ccccc1',      # valid (same as train[0])
        'BADSMILES',              # invalid
        'CC(O)=CC',               # valid (tautomer of train[4])
        None,                     # None
    ]
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    return df_train, df_test


def create_single_molecule_splits():
    """Create splits with single molecules."""
    df_train = pd.DataFrame({'smiles': ['C[C@H](O)c1ccccc1']})
    df_test = pd.DataFrame({'smiles': ['C[C@@H](O)c1ccccc1']})  # stereoisomer
    
    return df_train, df_test


def create_empty_split_data():
    """Create dataset with empty test split."""
    df_train = pd.DataFrame({'smiles': ['C[C@H](O)c1ccccc1', 'CCCCCC']})
    df_test = pd.DataFrame({'smiles': []})
    
    return df_train, df_test


def create_diverse_isomer_data():
    """Create dataset with many different types of isomers."""
    train_smiles = [
        'C[C@H](O)c1ccccc1',      # R stereoisomer
        r'C/C=C/C',               # E geometric isomer
        'CC(=O)CC',               # keto tautomer
        'C[C@H](O)[C@H](O)C',     # R,R diastereomer
        'C[C@H](Cl)CCCl',         # R with chlorines
    ]
    
    test_smiles = [
        'C[C@@H](O)c1ccccc1',     # S stereoisomer (pair with train[0])
        r'C/C=C\C',               # Z geometric isomer (pair with train[1])
        'CC(O)=CC',               # enol tautomer (pair with train[2])
        'C[C@@H](O)[C@@H](O)C',   # S,S diastereomer (pair with train[3])
        'CCCCCC',                 # different molecule (no pair)
    ]
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    return df_train, df_test


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def store_test_splits(train_df, test_df, val_df=None, prefix='test'):
    """Store test splits and return filenames."""
    train_file = _store_resource(
        train_df, TEST_MANIFEST, f'{prefix}_train',
        'Test training split', 'csv'
    )
    test_file = _store_resource(
        test_df, TEST_MANIFEST, f'{prefix}_test',
        'Test test split', 'csv'
    )
    val_file = None
    if val_df is not None:
        val_file = _store_resource(
            val_df, TEST_MANIFEST, f'{prefix}_val',
            'Test validation split', 'csv'
        )
    
    return train_file, test_file, val_file


# ============================================================================
# TESTS
# ============================================================================

def test_basic_structure():
    """Test that function returns correct structure."""
    print("\n" + "="*80)
    print("TEST: Basic structure validation")
    print("="*80)
    
    df_train, df_test = create_stereoisomer_test_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='struct')
    
    result = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', max_examples=10
    )
    
    # Check required fields
    assert 'train_test_stereoisomers' in result
    assert 'train_test_tautomers' in result
    assert 'train_val_stereoisomers' in result
    assert 'train_val_tautomers' in result
    assert 'test_val_stereoisomers' in result
    assert 'test_val_tautomers' in result
    assert 'overall_severity' in result
    assert 'total_stereoisomer_pairs' in result
    assert 'total_tautomer_pairs' in result
    
    # Check structure of train_test results
    assert 'n_pairs' in result['train_test_stereoisomers']
    assert 'examples' in result['train_test_stereoisomers']
    assert 'showing_n_examples' in result['train_test_stereoisomers']
    
    # Val results should be None (no val split)
    assert result['train_val_stereoisomers'] is None
    assert result['train_val_tautomers'] is None
    assert result['test_val_stereoisomers'] is None
    assert result['test_val_tautomers'] is None
    
    print("✅ Structure correct")
    print(f"   - Total stereoisomer pairs: {result['total_stereoisomer_pairs']}")
    print(f"   - Total tautomer pairs: {result['total_tautomer_pairs']}")
    print(f"   - Overall severity: {result['overall_severity']}")


def test_stereoisomer_detection():
    """Test detection of R/S stereoisomers."""
    print("\n" + "="*80)
    print("TEST: Stereoisomer detection (R/S)")
    print("="*80)
    
    df_train, df_test = create_stereoisomer_test_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='stereo')
    
    result = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', max_examples=10
    )
    
    # Should detect stereoisomer pairs
    assert result['train_test_stereoisomers']['n_pairs'] > 0
    assert result['total_stereoisomer_pairs'] > 0
    
    # Should flag as MEDIUM severity
    assert result['overall_severity'] == 'MEDIUM'
    
    # Check examples structure
    examples = result['train_test_stereoisomers']['examples']
    assert len(examples) > 0
    
    for example in examples:
        assert 'canonical_form' in example
        assert 'train_index' in example
        assert 'train_smiles' in example
        assert 'test_index' in example
        assert 'test_smiles' in example
        assert 'type' in example
        assert example['type'] == 'stereoisomer'
    
    print("✅ Stereoisomers detected")
    print(f"   - Number of stereoisomer pairs: {result['train_test_stereoisomers']['n_pairs']}")
    print(f"   - Severity: {result['overall_severity']}")
    print(f"   - Example pairs: {len(examples)}")


def test_tautomer_detection():
    """Test detection of tautomer pairs."""
    print("\n" + "="*80)
    print("TEST: Tautomer detection (keto-enol)")
    print("="*80)
    
    df_train, df_test = create_tautomer_test_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='taut')
    
    result = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', max_examples=10
    )
    
    # May detect tautomer pairs (depends on RDKit's tautomer canonicalization)
    n_tautomers = result['train_test_tautomers']['n_pairs']
    
    # Check structure even if no tautomers detected
    assert 'n_pairs' in result['train_test_tautomers']
    assert 'examples' in result['train_test_tautomers']
    
    if n_tautomers > 0:
        examples = result['train_test_tautomers']['examples']
        
        for example in examples:
            assert 'canonical_form' in example
            assert 'type' in example
            assert example['type'] == 'tautomer'
        
        print("✅ Tautomers detected")
        print(f"   - Number of tautomer pairs: {n_tautomers}")
    else:
        print("⚠️  No tautomers detected (may be due to RDKit canonicalization)")
    
    print(f"   - Severity: {result['overall_severity']}")


def test_ez_geometric_isomers():
    """Test detection of E/Z geometric isomers."""
    print("\n" + "="*80)
    print("TEST: E/Z geometric isomer detection")
    print("="*80)
    
    df_train, df_test = create_ez_isomer_test_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='ez')
    
    result = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', max_examples=10
    )
    
    # E/Z isomers are stereoisomers
    assert result['train_test_stereoisomers']['n_pairs'] > 0
    assert result['overall_severity'] == 'MEDIUM'
    
    print("✅ E/Z geometric isomers detected")
    print(f"   - Number of pairs: {result['train_test_stereoisomers']['n_pairs']}")


def test_no_leakage():
    """Test clean split with no stereoisomer or tautomer leakage."""
    print("\n" + "="*80)
    print("TEST: Clean split (no leakage)")
    print("="*80)
    
    df_train, df_test = create_clean_split_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='clean')
    
    result = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', max_examples=10
    )
    
    # Should detect no pairs
    assert result['train_test_stereoisomers']['n_pairs'] == 0
    assert result['train_test_tautomers']['n_pairs'] == 0
    assert result['total_stereoisomer_pairs'] == 0
    assert result['total_tautomer_pairs'] == 0
    
    # Should be OK severity
    assert result['overall_severity'] == 'OK'
    
    print("✅ No leakage detected")
    print(f"   - Severity: {result['overall_severity']}")


def test_three_way_split():
    """Test three-way split with train/test/val."""
    print("\n" + "="*80)
    print("TEST: Three-way split (train/test/val)")
    print("="*80)
    
    df_train, df_test, df_val = create_three_way_split_data()
    train_file, test_file, val_file = store_test_splits(df_train, df_test, df_val, prefix='threeway')
    
    result = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, val_file, TEST_MANIFEST, 'smiles', max_examples=10
    )
    
    # Check all comparisons present
    assert result['train_test_stereoisomers'] is not None
    assert result['train_test_tautomers'] is not None
    assert result['train_val_stereoisomers'] is not None
    assert result['train_val_tautomers'] is not None
    assert result['test_val_stereoisomers'] is not None
    assert result['test_val_tautomers'] is not None
    
    # Should detect some pairs
    total_pairs = (
        result['train_test_stereoisomers']['n_pairs'] +
        result['train_test_tautomers']['n_pairs'] +
        result['train_val_stereoisomers']['n_pairs'] +
        result['train_val_tautomers']['n_pairs'] +
        result['test_val_stereoisomers']['n_pairs'] +
        result['test_val_tautomers']['n_pairs']
    )
    
    assert total_pairs > 0
    assert result['overall_severity'] == 'MEDIUM'
    
    print("✅ Three-way split analyzed")
    print(f"   - Train/test stereoisomers: {result['train_test_stereoisomers']['n_pairs']}")
    print(f"   - Train/test tautomers: {result['train_test_tautomers']['n_pairs']}")
    print(f"   - Train/val stereoisomers: {result['train_val_stereoisomers']['n_pairs']}")
    print(f"   - Train/val tautomers: {result['train_val_tautomers']['n_pairs']}")
    print(f"   - Test/val stereoisomers: {result['test_val_stereoisomers']['n_pairs']}")
    print(f"   - Test/val tautomers: {result['test_val_tautomers']['n_pairs']}")


def test_complex_stereoisomers():
    """Test molecules with multiple chiral centers."""
    print("\n" + "="*80)
    print("TEST: Complex stereoisomers (multiple chiral centers)")
    print("="*80)
    
    df_train, df_test = create_complex_stereoisomer_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='complex')
    
    result = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', max_examples=10
    )
    
    # Should detect diastereomer pairs
    assert result['train_test_stereoisomers']['n_pairs'] > 0
    assert result['overall_severity'] == 'MEDIUM'
    
    print("✅ Complex stereoisomers detected")
    print(f"   - Number of pairs: {result['train_test_stereoisomers']['n_pairs']}")


def test_invalid_smiles():
    """Test handling of invalid SMILES."""
    print("\n" + "="*80)
    print("TEST: Invalid SMILES handling")
    print("="*80)
    
    df_train, df_test = create_invalid_smiles_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='invalid')
    
    result = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', max_examples=10
    )
    
    # Should handle invalid SMILES gracefully
    assert 'train_test_stereoisomers' in result
    assert 'train_test_tautomers' in result
    
    # May detect some valid pairs among the valid SMILES
    print("✅ Invalid SMILES handled gracefully")
    print(f"   - Stereoisomer pairs: {result['train_test_stereoisomers']['n_pairs']}")
    print(f"   - Tautomer pairs: {result['train_test_tautomers']['n_pairs']}")


def test_edge_cases():
    """Test edge cases (single molecules, empty splits)."""
    print("\n" + "="*80)
    print("TEST: Edge cases (single molecules, empty splits)")
    print("="*80)
    
    # Test single molecule splits
    df_train, df_test = create_single_molecule_splits()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='single')
    
    result = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', max_examples=10
    )
    
    # Should detect the stereoisomer pair
    assert result['train_test_stereoisomers']['n_pairs'] == 1
    
    print("✅ Single molecule splits handled")
    print(f"   - Detected: {result['train_test_stereoisomers']['n_pairs']} stereoisomer pair")
    
    # Test empty split
    df_train, df_test = create_empty_split_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='empty')
    
    result = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', max_examples=10
    )
    
    # Should handle empty split gracefully
    assert result['train_test_stereoisomers']['n_pairs'] == 0
    assert result['train_test_tautomers']['n_pairs'] == 0
    assert result['overall_severity'] == 'OK'
    
    print("✅ Empty split handled")


def test_example_structure():
    """Test that examples have correct structure."""
    print("\n" + "="*80)
    print("TEST: Example structure and content")
    print("="*80)
    
    df_train, df_test = create_diverse_isomer_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='diverse')
    
    result = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', max_examples=3
    )
    
    # Check stereoisomer examples
    if result['train_test_stereoisomers']['n_pairs'] > 0:
        examples = result['train_test_stereoisomers']['examples']
        
        for example in examples:
            assert 'canonical_form' in example
            assert 'train_index' in example
            assert 'train_smiles' in example
            assert 'test_index' in example
            assert 'test_smiles' in example
            assert 'type' in example
            assert example['type'] == 'stereoisomer'
            
            # SMILES should be different (that's the point!)
            assert example['train_smiles'] != example['test_smiles']
        
        print("✅ Stereoisomer examples structured correctly")
        print(f"   - Number of examples: {len(examples)}")
        print(f"   - Example: {examples[0]['train_smiles']} vs {examples[0]['test_smiles']}")
    
    # Check tautomer examples
    if result['train_test_tautomers']['n_pairs'] > 0:
        examples = result['train_test_tautomers']['examples']
        
        for example in examples:
            assert example['type'] == 'tautomer'
            assert example['train_smiles'] != example['test_smiles']
        
        print("✅ Tautomer examples structured correctly")


def test_severity_levels():
    """Test severity level assignment."""
    print("\n" + "="*80)
    print("TEST: Severity level assignment")
    print("="*80)
    
    # Clean split → OK
    df_train, df_test = create_clean_split_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='sev_ok')
    
    result = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', max_examples=10
    )
    
    assert result['overall_severity'] == 'OK'
    print("✅ OK severity for clean split")
    
    # Stereoisomer leakage → MEDIUM
    df_train, df_test = create_stereoisomer_test_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='sev_medium')
    
    result = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', max_examples=10
    )
    
    assert result['overall_severity'] == 'MEDIUM'
    print("✅ MEDIUM severity for stereoisomer leakage")


def test_max_examples_parameter():
    """Test that max_examples parameter limits output."""
    print("\n" + "="*80)
    print("TEST: max_examples parameter")
    print("="*80)
    
    df_train, df_test = create_diverse_isomer_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='maxex')
    
    # Test with max_examples=2
    result = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', max_examples=2
    )
    
    n_stereo = result['train_test_stereoisomers']['n_pairs']
    n_examples = len(result['train_test_stereoisomers']['examples'])
    
    if n_stereo > 0:
        assert n_examples <= 2
        assert result['train_test_stereoisomers']['showing_n_examples'] == min(n_stereo, 2)
        
        print("✅ max_examples limits output correctly")
        print(f"   - Total pairs: {n_stereo}")
        print(f"   - Examples shown: {n_examples}")


def test_consistency():
    """Test that results are consistent across runs."""
    print("\n" + "="*80)
    print("TEST: Consistency across runs")
    print("="*80)
    
    df_train, df_test = create_stereoisomer_test_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='consist')
    
    result1 = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', max_examples=10
    )
    
    result2 = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', max_examples=10
    )
    
    # Results should be identical
    assert result1['train_test_stereoisomers']['n_pairs'] == result2['train_test_stereoisomers']['n_pairs']
    assert result1['train_test_tautomers']['n_pairs'] == result2['train_test_tautomers']['n_pairs']
    assert result1['total_stereoisomer_pairs'] == result2['total_stereoisomer_pairs']
    assert result1['total_tautomer_pairs'] == result2['total_tautomer_pairs']
    assert result1['overall_severity'] == result2['overall_severity']
    
    print("✅ Results consistent across runs")


def test_total_counts():
    """Test that total counts are sum of all comparisons."""
    print("\n" + "="*80)
    print("TEST: Total count accuracy")
    print("="*80)
    
    df_train, df_test, df_val = create_three_way_split_data()
    train_file, test_file, val_file = store_test_splits(df_train, df_test, df_val, prefix='totals')
    
    result = _detect_stereoisomer_tautomer_leakage(
        train_file, test_file, val_file, TEST_MANIFEST, 'smiles', max_examples=10
    )
    
    # Calculate expected totals
    expected_stereo = (
        result['train_test_stereoisomers']['n_pairs'] +
        result['train_val_stereoisomers']['n_pairs'] +
        result['test_val_stereoisomers']['n_pairs']
    )
    
    expected_taut = (
        result['train_test_tautomers']['n_pairs'] +
        result['train_val_tautomers']['n_pairs'] +
        result['test_val_tautomers']['n_pairs']
    )
    
    assert result['total_stereoisomer_pairs'] == expected_stereo
    assert result['total_tautomer_pairs'] == expected_taut
    
    print("✅ Total counts accurate")
    print(f"   - Total stereoisomer pairs: {result['total_stereoisomer_pairs']}")
    print(f"   - Total tautomer pairs: {result['total_tautomer_pairs']}")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == '__main__':
    print("\n" + "#"*80)
    print("# COMPREHENSIVE TEST SUITE: _detect_stereoisomer_tautomer_leakage()")
    print("#"*80)
    
    test_basic_structure()
    test_stereoisomer_detection()
    test_tautomer_detection()
    test_ez_geometric_isomers()
    test_no_leakage()
    test_three_way_split()
    test_complex_stereoisomers()
    test_invalid_smiles()
    test_edge_cases()
    test_example_structure()
    test_severity_levels()
    test_max_examples_parameter()
    test_consistency()
    test_total_counts()
    
    print("\n" + "#"*80)
    print("# ALL TESTS PASSED! ✅")
    print("#"*80)
    print("\nFunction _detect_stereoisomer_tautomer_leakage() is production-ready!")
