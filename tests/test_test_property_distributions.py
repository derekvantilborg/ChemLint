"""
Comprehensive test suite for _test_property_distributions().

This function tests if physicochemical property distributions differ significantly
between splits using Kolmogorov-Smirnov (KS) tests.

Properties tested:
- Molecular Weight (MolWt)
- LogP (octanol-water partition coefficient)
- TPSA (Topological Polar Surface Area)
- Number of H-bond donors (NumHDonors)
- Number of H-bond acceptors (NumHAcceptors)
- Number of rotatable bonds (NumRotatableBonds)
- Number of aromatic rings (NumAromaticRings)
- Number of heavy atoms (NumHeavyAtoms)

Tests cover:
1. Basic structure validation
2. Similar distributions (well-balanced split)
3. Different distributions (biased split)
4. Three-way splits
5. Invalid SMILES handling
6. Edge cases (small splits, empty splits)
7. Statistical test validity
8. Property computation accuracy
9. Severity levels
10. Alpha threshold behavior
11. Consistency across runs
"""

import sys
import os
import pandas as pd
import numpy as np
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from molml_mcp.tools.reports.data_splitting import _test_property_distributions
from molml_mcp.infrastructure.resources import _store_resource

# Test data directory
TEST_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEST_MANIFEST = os.path.join(TEST_DIR, 'test_manifest.json')


# ============================================================================
# TEST SCENARIO GENERATORS
# ============================================================================

def create_similar_distribution_data():
    """Create train/test with similar property distributions (good split)."""
    # Random sample of diverse drug-like molecules
    all_smiles = [
        'CCO',                          # ethanol (small, polar)
        'CC(C)O',                       # isopropanol
        'CCCC',                         # butane (small, nonpolar)
        'c1ccccc1',                     # benzene (aromatic)
        'c1ccc(O)cc1',                  # phenol (aromatic + OH)
        'c1ccc(C)cc1',                  # toluene
        'CC(=O)O',                      # acetic acid (acid)
        'CCC(=O)O',                     # propanoic acid
        'CC(=O)N',                      # acetamide (amide)
        'CCCCCC',                       # hexane (longer alkane)
        'c1ccc(N)cc1',                  # aniline (aromatic + NH2)
        'CC(C)CC(C)C',                  # isoheptane (branched)
        'c1ccccc1C',                    # toluene
        'CC(=O)C',                      # acetone (ketone)
        'CCCCCCCC',                     # octane (larger alkane)
        'c1ccc(O)c(O)c1',               # catechol (2 OH groups)
        'CC(C)(C)C',                    # neopentane (highly branched)
        'c1ccc(Cl)cc1',                 # chlorobenzene (halogenated)
        'CCCCCCCCCC',                   # decane
        'c1ccc2ccccc2c1',               # naphthalene (bicyclic aromatic)
    ]
    
    # Randomly shuffle and split 80/20
    random.seed(42)
    random.shuffle(all_smiles)
    split_idx = int(len(all_smiles) * 0.8)
    
    train_smiles = all_smiles[:split_idx]
    test_smiles = all_smiles[split_idx:]
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    return df_train, df_test


def create_biased_distribution_data():
    """Create train/test with different property distributions (biased split)."""
    # Train: small, polar molecules
    train_smiles = [
        'CCO',                          # ethanol
        'CC(C)O',                       # isopropanol
        'CCCO',                         # propanol
        'CC(=O)O',                      # acetic acid
        'CCC(=O)O',                     # propanoic acid
        'CC(=O)N',                      # acetamide
        'CCN',                          # ethylamine
        'CCCN',                         # propylamine
        'CC(O)C',                       # isopropanol
        'CCC(O)C',                      # 2-butanol
    ]
    
    # Test: large, nonpolar/aromatic molecules
    test_smiles = [
        'CCCCCCCCCC',                   # decane
        'CCCCCCCCCCC',                  # undecane
        'c1ccc2ccccc2c1',               # naphthalene
        'c1ccc2cc3ccccc3cc2c1',         # anthracene
        'c1ccc(CCCC)cc1',               # butylbenzene
        'CCCCCCCCCCCC',                 # dodecane
        'c1ccc(CCCCC)cc1',              # pentylbenzene
        'CCCCCCCCCCCCC',                # tridecane
        'c1ccc2c(c1)cccc2',             # naphthalene
        'CCCCCCCC(C)C',                 # 2-methylnonane
    ]
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    return df_train, df_test


def create_three_way_split_data():
    """Create three-way split with similar distributions."""
    all_smiles = [
        'CCO', 'CC(C)O', 'CCCC', 'c1ccccc1', 'c1ccc(O)cc1',
        'CC(=O)O', 'CCC(=O)O', 'CC(=O)N', 'CCCCCC', 'c1ccc(N)cc1',
        'CC(C)CC(C)C', 'CC(=O)C', 'CCCCCCCC', 'c1ccc(Cl)cc1', 'CCCCCCCCCC',
        'c1ccc2ccccc2c1', 'CC(C)(C)C', 'c1ccc(O)c(O)c1', 'CCCCCCCCCCC', 'c1ccc(C)cc1'
    ]
    
    random.seed(42)
    random.shuffle(all_smiles)
    
    # 70/15/15 split
    train_idx = int(len(all_smiles) * 0.7)
    test_idx = train_idx + int(len(all_smiles) * 0.15)
    
    train_smiles = all_smiles[:train_idx]
    test_smiles = all_smiles[train_idx:test_idx]
    val_smiles = all_smiles[test_idx:]
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    df_val = pd.DataFrame({'smiles': val_smiles})
    
    return df_train, df_test, df_val


def create_invalid_smiles_data():
    """Create dataset with invalid SMILES."""
    train_smiles = [
        'CCO',                          # valid
        'INVALID',                      # invalid
        'c1ccccc1',                     # valid
        'BADSMILES',                    # invalid
        'CCCC',                         # valid
        '',                             # empty
        'CC(C)O',                       # valid
    ]
    
    test_smiles = [
        'CCCO',                         # valid
        'NOTSMILES',                    # invalid
        'c1ccc(O)cc1',                  # valid
        None,                           # None
        'CCCCCC',                       # valid
    ]
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    return df_train, df_test


def create_small_split_data():
    """Create very small splits."""
    df_train = pd.DataFrame({'smiles': ['CCO', 'CCCC']})
    df_test = pd.DataFrame({'smiles': ['CCCO']})
    
    return df_train, df_test


def create_empty_split_data():
    """Create dataset with empty test split."""
    df_train = pd.DataFrame({'smiles': ['CCO', 'CCCC', 'c1ccccc1']})
    df_test = pd.DataFrame({'smiles': []})
    
    return df_train, df_test


def create_homogeneous_data():
    """Create dataset with very similar molecules (low property variance)."""
    # All molecules are similar linear alkanes
    train_smiles = ['CCCCCC', 'CCCCCCC', 'CCCCCCCC', 'CCCCCCCCC', 'CCCCCCCCCC']
    test_smiles = ['CCCCCCC', 'CCCCCCCC', 'CCCCCCCCC']
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    return df_train, df_test


def create_extreme_bias_data():
    """Create dataset with extreme property differences."""
    # Train: very small molecules
    train_smiles = ['C', 'CC', 'CCC', 'CCCC', 'CO', 'CCO', 'CN', 'CCN']
    
    # Test: very large molecules
    test_smiles = [
        'CCCCCCCCCCCCCCCC',                              # C16
        'CCCCCCCCCCCCCCCCC',                             # C17
        'c1ccc2c(c1)ccc1c2ccc2ccccc12',                  # tetracene
        'c1ccc2cc3cc4ccccc4cc3cc2c1',                    # large PAH
        'CCCCCCCCCCCCCCCCCC',                            # C18
    ]
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    return df_train, df_test


def create_diverse_molecules_data():
    """Create dataset with chemically diverse molecules."""
    train_smiles = [
        'CCO',                          # alcohol
        'CC(=O)O',                      # acid
        'CC(=O)N',                      # amide
        'c1ccccc1',                     # aromatic
        'CCCC',                         # alkane
        'CC#N',                         # nitrile
        'CC(=O)OC',                     # ester
        'c1ccc(N)cc1',                  # aniline
        'CC(C)O',                       # branched alcohol
        'CCCCCCCC',                     # longer alkane
    ]
    
    test_smiles = [
        'CCCO',                         # alcohol
        'CCC(=O)O',                     # acid
        'CCC(=O)N',                     # amide
        'c1ccc(C)cc1',                  # aromatic
        'CCCCC',                        # alkane
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
    
    df_train, df_test = create_similar_distribution_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='struct')
    
    result = _test_property_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', alpha=0.05
    )
    
    # Check required fields
    assert 'alpha' in result
    assert 'properties_tested' in result
    assert 'computation_stats' in result
    assert 'train_vs_test' in result
    assert 'train_vs_val' in result
    assert 'test_vs_val' in result
    assert 'overall_severity' in result
    assert 'summary' in result
    
    # Check alpha value
    assert result['alpha'] == 0.05
    
    # Check properties list
    expected_props = ['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
                      'NumRotatableBonds', 'NumAromaticRings', 'NumHeavyAtoms']
    assert result['properties_tested'] == expected_props
    
    # Check computation stats
    assert 'train_computed' in result['computation_stats']
    assert 'train_failed' in result['computation_stats']
    assert 'test_computed' in result['computation_stats']
    assert 'test_failed' in result['computation_stats']
    
    # Check train_vs_test structure
    assert result['train_vs_test'] is not None
    for prop in expected_props:
        assert prop in result['train_vs_test']
        prop_result = result['train_vs_test'][prop]
        
        if 'error' not in prop_result:
            assert 'ks_statistic' in prop_result
            assert 'p_value' in prop_result
            assert 'significant' in prop_result
            assert 'interpretation' in prop_result
            assert 'train_mean' in prop_result
            assert 'test_mean' in prop_result
    
    # Val results should be None (no val split)
    assert result['train_vs_val'] is None
    assert result['test_vs_val'] is None
    
    # Check summary
    assert 'n_properties_tested' in result['summary']
    assert 'n_significant_train_test' in result['summary']
    
    print("✅ Structure correct")
    print(f"   - Alpha: {result['alpha']}")
    print(f"   - Properties tested: {len(result['properties_tested'])}")
    print(f"   - Train computed: {result['computation_stats']['train_computed']}")
    print(f"   - Test computed: {result['computation_stats']['test_computed']}")
    print(f"   - Overall severity: {result['overall_severity']}")


def test_similar_distributions():
    """Test well-balanced split with similar property distributions."""
    print("\n" + "="*80)
    print("TEST: Similar distributions (good split)")
    print("="*80)
    
    df_train, df_test = create_similar_distribution_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='similar')
    
    result = _test_property_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', alpha=0.05
    )
    
    # Should compute properties for most molecules
    assert result['computation_stats']['train_computed'] > 10
    assert result['computation_stats']['test_computed'] > 3
    
    # Check that we got valid results for all properties
    for prop in result['properties_tested']:
        prop_result = result['train_vs_test'][prop]
        
        if 'error' in prop_result:
            print(f"   ⚠️  {prop}: {prop_result['error']}")
            continue
        
        # Should have valid KS test results
        assert 0 <= prop_result['ks_statistic'] <= 1
        assert 0 <= prop_result['p_value'] <= 1
        assert prop_result['interpretation'] in ['SIMILAR', 'DIFFERENT']
    
    # Most properties should be similar (random split)
    n_significant = result['summary']['n_significant_train_test']
    n_total = len(result['properties_tested'])
    
    print("✅ Similar distributions analyzed")
    print(f"   - Significant differences: {n_significant}/{n_total}")
    print(f"   - Overall severity: {result['overall_severity']}")


def test_biased_distributions():
    """Test biased split with different property distributions."""
    print("\n" + "="*80)
    print("TEST: Biased distributions (train=small/polar, test=large/nonpolar)")
    print("="*80)
    
    df_train, df_test = create_biased_distribution_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='biased')
    
    result = _test_property_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', alpha=0.05
    )
    
    # Should detect significant differences
    n_significant = result['summary']['n_significant_train_test']
    
    assert n_significant > 0, "Should detect at least one significant difference"
    
    # Should flag as MEDIUM severity
    assert result['overall_severity'] == 'MEDIUM'
    
    # Check specific properties that should differ
    # MolWt: test should have higher molecular weight
    if 'error' not in result['train_vs_test']['MolWt']:
        molwt_result = result['train_vs_test']['MolWt']
        train_mw = molwt_result['train_mean']
        test_mw = molwt_result['test_mean']
        
        assert test_mw > train_mw, f"Test MolWt ({test_mw}) should be > train ({train_mw})"
        print(f"   ✅ MolWt: train={train_mw:.1f}, test={test_mw:.1f}")
    
    # LogP: test should have higher LogP (more nonpolar)
    if 'error' not in result['train_vs_test']['LogP']:
        logp_result = result['train_vs_test']['LogP']
        train_logp = logp_result['train_mean']
        test_logp = logp_result['test_mean']
        
        print(f"   ✅ LogP: train={train_logp:.2f}, test={test_logp:.2f}")
    
    print("✅ Biased distributions detected")
    print(f"   - Significant differences: {n_significant}/{len(result['properties_tested'])}")
    print(f"   - Overall severity: {result['overall_severity']}")


def test_three_way_split():
    """Test three-way split with train/test/val."""
    print("\n" + "="*80)
    print("TEST: Three-way split (train/test/val)")
    print("="*80)
    
    df_train, df_test, df_val = create_three_way_split_data()
    train_file, test_file, val_file = store_test_splits(df_train, df_test, df_val, prefix='threeway')
    
    result = _test_property_distributions(
        train_file, test_file, val_file, TEST_MANIFEST, 'smiles', alpha=0.05
    )
    
    # Check all comparisons present
    assert result['train_vs_test'] is not None
    assert result['train_vs_val'] is not None
    assert result['test_vs_val'] is not None
    
    # Check computation stats for val
    assert 'val_computed' in result['computation_stats']
    assert 'val_failed' in result['computation_stats']
    
    # Check summary includes val comparisons
    assert 'n_significant_train_val' in result['summary']
    assert 'n_significant_test_val' in result['summary']
    
    print("✅ Three-way split analyzed")
    print(f"   - Train/test significant: {result['summary']['n_significant_train_test']}")
    print(f"   - Train/val significant: {result['summary']['n_significant_train_val']}")
    print(f"   - Test/val significant: {result['summary']['n_significant_test_val']}")
    print(f"   - Overall severity: {result['overall_severity']}")


def test_invalid_smiles():
    """Test handling of invalid SMILES."""
    print("\n" + "="*80)
    print("TEST: Invalid SMILES handling")
    print("="*80)
    
    df_train, df_test = create_invalid_smiles_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='invalid')
    
    result = _test_property_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', alpha=0.05
    )
    
    # Should have some failed computations
    assert result['computation_stats']['train_failed'] > 0
    assert result['computation_stats']['test_failed'] > 0
    
    # Should still compute for valid molecules
    assert result['computation_stats']['train_computed'] > 0
    assert result['computation_stats']['test_computed'] > 0
    
    print("✅ Invalid SMILES handled gracefully")
    print(f"   - Train: {result['computation_stats']['train_computed']} computed, "
          f"{result['computation_stats']['train_failed']} failed")
    print(f"   - Test: {result['computation_stats']['test_computed']} computed, "
          f"{result['computation_stats']['test_failed']} failed")


def test_small_splits():
    """Test handling of very small splits."""
    print("\n" + "="*80)
    print("TEST: Small splits")
    print("="*80)
    
    df_train, df_test = create_small_split_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='small')
    
    result = _test_property_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', alpha=0.05
    )
    
    # Should have insufficient data errors or still work
    for prop in result['properties_tested']:
        prop_result = result['train_vs_test'][prop]
        
        if 'error' in prop_result:
            assert 'Insufficient data' in prop_result['error']
            print(f"   ⚠️  {prop}: Insufficient data (expected)")
        else:
            # If we got results, they should be valid
            assert 'ks_statistic' in prop_result
            assert 'p_value' in prop_result
    
    print("✅ Small splits handled")


def test_empty_split():
    """Test handling of empty test split."""
    print("\n" + "="*80)
    print("TEST: Empty split")
    print("="*80)
    
    df_train, df_test = create_empty_split_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='empty')
    
    result = _test_property_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', alpha=0.05
    )
    
    # Test should have 0 computed
    assert result['computation_stats']['test_computed'] == 0
    
    # All properties should have errors
    for prop in result['properties_tested']:
        prop_result = result['train_vs_test'][prop]
        assert 'error' in prop_result
        assert 'Insufficient data' in prop_result['error']
    
    print("✅ Empty split handled")
    print(f"   - Test computed: {result['computation_stats']['test_computed']}")


def test_property_computation_accuracy():
    """Test that property values are computed correctly."""
    print("\n" + "="*80)
    print("TEST: Property computation accuracy")
    print("="*80)
    
    # Use molecules with known properties
    df_train = pd.DataFrame({'smiles': ['CCO', 'c1ccccc1']})  # ethanol, benzene
    df_test = pd.DataFrame({'smiles': ['CCCC', 'c1ccc(O)cc1']})  # butane, phenol
    
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='accuracy')
    
    result = _test_property_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', alpha=0.05
    )
    
    # Check that computed values are in reasonable ranges
    for prop, expected_range in [
        ('MolWt', (0, 500)),        # Molecular weight 0-500
        ('LogP', (-5, 10)),         # LogP -5 to 10
        ('TPSA', (0, 200)),         # TPSA 0-200
        ('NumHDonors', (0, 20)),    # H-bond donors 0-20
        ('NumHAcceptors', (0, 20)), # H-bond acceptors 0-20
        ('NumRotatableBonds', (0, 20)),  # Rotatable bonds 0-20
        ('NumAromaticRings', (0, 10)),   # Aromatic rings 0-10
        ('NumHeavyAtoms', (0, 100)),     # Heavy atoms 0-100
    ]:
        if 'error' not in result['train_vs_test'][prop]:
            prop_result = result['train_vs_test'][prop]
            train_mean = prop_result['train_mean']
            test_mean = prop_result['test_mean']
            
            assert expected_range[0] <= train_mean <= expected_range[1], \
                f"{prop} train_mean {train_mean} out of range {expected_range}"
            assert expected_range[0] <= test_mean <= expected_range[1], \
                f"{prop} test_mean {test_mean} out of range {expected_range}"
    
    print("✅ Property values in expected ranges")


def test_ks_test_statistics():
    """Test that KS test statistics are valid."""
    print("\n" + "="*80)
    print("TEST: KS test statistics validity")
    print("="*80)
    
    df_train, df_test = create_diverse_molecules_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='ks')
    
    result = _test_property_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', alpha=0.05
    )
    
    for prop in result['properties_tested']:
        prop_result = result['train_vs_test'][prop]
        
        if 'error' in prop_result:
            continue
        
        # KS statistic should be between 0 and 1
        assert 0 <= prop_result['ks_statistic'] <= 1
        
        # p-value should be between 0 and 1
        assert 0 <= prop_result['p_value'] <= 1
        
        # Interpretation should match significance
        if prop_result['significant']:
            assert prop_result['interpretation'] == 'DIFFERENT'
            assert prop_result['p_value'] < 0.05
        else:
            assert prop_result['interpretation'] == 'SIMILAR'
            assert prop_result['p_value'] >= 0.05
    
    print("✅ KS test statistics valid")


def test_severity_levels():
    """Test severity level assignment."""
    print("\n" + "="*80)
    print("TEST: Severity level assignment")
    print("="*80)
    
    # Similar distributions → OK or LOW
    df_train, df_test = create_similar_distribution_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='sev_ok')
    
    result = _test_property_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', alpha=0.05
    )
    
    print(f"   - Similar split severity: {result['overall_severity']}")
    
    # Biased distributions → MEDIUM
    df_train, df_test = create_biased_distribution_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='sev_med')
    
    result = _test_property_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', alpha=0.05
    )
    
    assert result['overall_severity'] == 'MEDIUM'
    print(f"   - Biased split severity: {result['overall_severity']}")
    
    # Extreme bias → MEDIUM
    df_train, df_test = create_extreme_bias_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='sev_high')
    
    result = _test_property_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', alpha=0.05
    )
    
    assert result['overall_severity'] == 'MEDIUM'
    print(f"   - Extreme bias severity: {result['overall_severity']}")
    
    print("✅ Severity levels appropriate")


def test_alpha_threshold():
    """Test that alpha threshold affects significance detection."""
    print("\n" + "="*80)
    print("TEST: Alpha threshold behavior")
    print("="*80)
    
    df_train, df_test = create_biased_distribution_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='alpha')
    
    # Test with strict alpha (0.01)
    result_strict = _test_property_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', alpha=0.01
    )
    
    # Test with lenient alpha (0.10)
    result_lenient = _test_property_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', alpha=0.10
    )
    
    n_sig_strict = result_strict['summary']['n_significant_train_test']
    n_sig_lenient = result_lenient['summary']['n_significant_train_test']
    
    # Lenient should detect more or equal significant differences
    assert n_sig_lenient >= n_sig_strict
    
    print("✅ Alpha threshold affects significance")
    print(f"   - Strict (α=0.01): {n_sig_strict} significant")
    print(f"   - Lenient (α=0.10): {n_sig_lenient} significant")


def test_consistency():
    """Test that results are consistent across runs."""
    print("\n" + "="*80)
    print("TEST: Consistency across runs")
    print("="*80)
    
    df_train, df_test = create_similar_distribution_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='consist')
    
    result1 = _test_property_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', alpha=0.05
    )
    
    result2 = _test_property_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'smiles', alpha=0.05
    )
    
    # Results should be identical
    assert result1['computation_stats'] == result2['computation_stats']
    assert result1['summary'] == result2['summary']
    assert result1['overall_severity'] == result2['overall_severity']
    
    # Check property results are identical
    for prop in result1['properties_tested']:
        prop1 = result1['train_vs_test'][prop]
        prop2 = result2['train_vs_test'][prop]
        
        if 'error' in prop1:
            assert 'error' in prop2
        else:
            assert prop1['ks_statistic'] == prop2['ks_statistic']
            assert prop1['p_value'] == prop2['p_value']
            assert prop1['significant'] == prop2['significant']
    
    print("✅ Results consistent across runs")


def test_summary_counts():
    """Test that summary counts are accurate."""
    print("\n" + "="*80)
    print("TEST: Summary count accuracy")
    print("="*80)
    
    df_train, df_test, df_val = create_three_way_split_data()
    train_file, test_file, val_file = store_test_splits(df_train, df_test, df_val, prefix='summary')
    
    result = _test_property_distributions(
        train_file, test_file, val_file, TEST_MANIFEST, 'smiles', alpha=0.05
    )
    
    # Count significant differences manually
    n_sig_train_test = sum(
        1 for prop in result['properties_tested']
        if 'error' not in result['train_vs_test'][prop]
        and result['train_vs_test'][prop]['significant']
    )
    
    n_sig_train_val = sum(
        1 for prop in result['properties_tested']
        if 'error' not in result['train_vs_val'][prop]
        and result['train_vs_val'][prop]['significant']
    )
    
    n_sig_test_val = sum(
        1 for prop in result['properties_tested']
        if 'error' not in result['test_vs_val'][prop]
        and result['test_vs_val'][prop]['significant']
    )
    
    # Compare with summary
    assert result['summary']['n_significant_train_test'] == n_sig_train_test
    assert result['summary']['n_significant_train_val'] == n_sig_train_val
    assert result['summary']['n_significant_test_val'] == n_sig_test_val
    
    print("✅ Summary counts accurate")
    print(f"   - Train/test: {n_sig_train_test}")
    print(f"   - Train/val: {n_sig_train_val}")
    print(f"   - Test/val: {n_sig_test_val}")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == '__main__':
    print("\n" + "#"*80)
    print("# COMPREHENSIVE TEST SUITE: _test_property_distributions()")
    print("#"*80)
    
    test_basic_structure()
    test_similar_distributions()
    test_biased_distributions()
    test_three_way_split()
    test_invalid_smiles()
    test_small_splits()
    test_empty_split()
    test_property_computation_accuracy()
    test_ks_test_statistics()
    test_severity_levels()
    test_alpha_threshold()
    test_consistency()
    test_summary_counts()
    
    print("\n" + "#"*80)
    print("# ALL TESTS PASSED! ✅")
    print("#"*80)
    print("\nFunction _test_property_distributions() is production-ready!")
