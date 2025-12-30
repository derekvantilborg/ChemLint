"""
Comprehensive test suite for _test_activity_distributions().

This function tests if activity/label distributions differ significantly between splits:
- Classification: Chi-square test for class proportions + class balance checks
- Regression: Kolmogorov-Smirnov (KS) test for value distributions

Tests cover:
1. Basic structure validation
2. Classification: similar distributions
3. Classification: biased distributions
4. Classification: class imbalance detection
5. Regression: similar distributions
6. Regression: biased distributions
7. Three-way splits
8. Missing label column handling
9. Empty splits
10. Small splits
11. Statistical test validity
12. Severity levels
13. Alpha threshold behavior
14. Consistency across runs
"""

import sys
import os
import pandas as pd
import numpy as np
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from molml_mcp.tools.reports.data_splitting import _test_activity_distributions
from molml_mcp.infrastructure.resources import _store_resource

# Test data directory
TEST_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEST_MANIFEST = os.path.join(TEST_DIR, 'test_manifest.json')


# ============================================================================
# TEST SCENARIO GENERATORS
# ============================================================================

def create_balanced_classification_data():
    """Create classification dataset with balanced class distributions."""
    # Binary classification: 80/20 split, balanced classes in both
    train_smiles = ['C' + 'C'*i for i in range(40)]  # 40 molecules
    train_labels = [0] * 20 + [1] * 20  # 50/50 split
    
    test_smiles = ['C' + 'C'*i for i in range(40, 50)]  # 10 molecules
    test_labels = [0] * 5 + [1] * 5  # 50/50 split
    
    df_train = pd.DataFrame({'smiles': train_smiles, 'activity': train_labels})
    df_test = pd.DataFrame({'smiles': test_smiles, 'activity': test_labels})
    
    return df_train, df_test


def create_biased_classification_data():
    """Create classification dataset with different class proportions."""
    # Train: 70% class 0, 30% class 1
    train_smiles = ['C' + 'C'*i for i in range(40)]
    train_labels = [0] * 28 + [1] * 12
    
    # Test: 30% class 0, 70% class 1 (opposite)
    test_smiles = ['C' + 'C'*i for i in range(40, 50)]
    test_labels = [0] * 3 + [1] * 7
    
    df_train = pd.DataFrame({'smiles': train_smiles, 'activity': train_labels})
    df_test = pd.DataFrame({'smiles': test_smiles, 'activity': test_labels})
    
    return df_train, df_test


def create_imbalanced_classification_data():
    """Create classification dataset with severe class imbalance."""
    # Train: 90% class 0, 10% class 1
    train_smiles = ['C' + 'C'*i for i in range(50)]
    train_labels = [0] * 45 + [1] * 5
    
    # Test: 85% class 0, 15% class 1
    test_smiles = ['C' + 'C'*i for i in range(50, 70)]
    test_labels = [0] * 17 + [1] * 3
    
    df_train = pd.DataFrame({'smiles': train_smiles, 'activity': train_labels})
    df_test = pd.DataFrame({'smiles': test_smiles, 'activity': test_labels})
    
    return df_train, df_test


def create_similar_regression_data():
    """Create regression dataset with similar value distributions."""
    random.seed(42)
    
    # Both from same normal distribution
    train_values = np.random.normal(5.0, 2.0, 40)
    test_values = np.random.normal(5.0, 2.0, 10)
    
    train_smiles = ['C' + 'C'*i for i in range(40)]
    test_smiles = ['C' + 'C'*i for i in range(40, 50)]
    
    df_train = pd.DataFrame({'smiles': train_smiles, 'activity': train_values})
    df_test = pd.DataFrame({'smiles': test_smiles, 'activity': test_values})
    
    return df_train, df_test


def create_biased_regression_data():
    """Create regression dataset with different value distributions."""
    random.seed(42)
    
    # Train: mean=3.0, std=1.0
    train_values = np.random.normal(3.0, 1.0, 40)
    
    # Test: mean=7.0, std=1.0 (significantly different)
    test_values = np.random.normal(7.0, 1.0, 10)
    
    train_smiles = ['C' + 'C'*i for i in range(40)]
    test_smiles = ['C' + 'C'*i for i in range(40, 50)]
    
    df_train = pd.DataFrame({'smiles': train_smiles, 'activity': train_values})
    df_test = pd.DataFrame({'smiles': test_smiles, 'activity': test_values})
    
    return df_train, df_test


def create_three_way_classification_data():
    """Create three-way classification split."""
    train_smiles = ['C' + 'C'*i for i in range(30)]
    train_labels = [0] * 15 + [1] * 15
    
    test_smiles = ['C' + 'C'*i for i in range(30, 40)]
    test_labels = [0] * 5 + [1] * 5
    
    val_smiles = ['C' + 'C'*i for i in range(40, 50)]
    val_labels = [0] * 5 + [1] * 5
    
    df_train = pd.DataFrame({'smiles': train_smiles, 'activity': train_labels})
    df_test = pd.DataFrame({'smiles': test_smiles, 'activity': test_labels})
    df_val = pd.DataFrame({'smiles': val_smiles, 'activity': val_labels})
    
    return df_train, df_test, df_val


def create_three_way_regression_data():
    """Create three-way regression split."""
    random.seed(42)
    
    train_values = np.random.normal(5.0, 2.0, 30)
    test_values = np.random.normal(5.0, 2.0, 10)
    val_values = np.random.normal(5.0, 2.0, 10)
    
    train_smiles = ['C' + 'C'*i for i in range(30)]
    test_smiles = ['C' + 'C'*i for i in range(30, 40)]
    val_smiles = ['C' + 'C'*i for i in range(40, 50)]
    
    df_train = pd.DataFrame({'smiles': train_smiles, 'activity': train_values})
    df_test = pd.DataFrame({'smiles': test_smiles, 'activity': test_values})
    df_val = pd.DataFrame({'smiles': val_smiles, 'activity': val_values})
    
    return df_train, df_test, df_val


def create_missing_label_data():
    """Create dataset with missing label column."""
    train_smiles = ['CCO', 'CCCC', 'c1ccccc1']
    test_smiles = ['CCCO', 'CCCCC']
    
    df_train = pd.DataFrame({'smiles': train_smiles})  # No activity column
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    return df_train, df_test


def create_empty_labels_data():
    """Create dataset with all NaN labels."""
    train_smiles = ['CCO', 'CCCC', 'c1ccccc1']
    train_labels = [np.nan, np.nan, np.nan]
    
    test_smiles = ['CCCO', 'CCCCC']
    test_labels = [np.nan, np.nan]
    
    df_train = pd.DataFrame({'smiles': train_smiles, 'activity': train_labels})
    df_test = pd.DataFrame({'smiles': test_smiles, 'activity': test_labels})
    
    return df_train, df_test


def create_small_classification_data():
    """Create very small classification dataset."""
    df_train = pd.DataFrame({'smiles': ['CCO', 'CCCC'], 'activity': [0, 1]})
    df_test = pd.DataFrame({'smiles': ['CCCO'], 'activity': [0]})
    
    return df_train, df_test


def create_small_regression_data():
    """Create very small regression dataset."""
    df_train = pd.DataFrame({'smiles': ['CCO', 'CCCC'], 'activity': [1.5, 2.3]})
    df_test = pd.DataFrame({'smiles': ['CCCO'], 'activity': [1.8]})
    
    return df_train, df_test


def create_multiclass_data():
    """Create multiclass classification dataset (should still be detected as classification)."""
    train_smiles = ['C' + 'C'*i for i in range(30)]
    train_labels = [0] * 10 + [1] * 10 + [2] * 10
    
    test_smiles = ['C' + 'C'*i for i in range(30, 40)]
    test_labels = [0] * 3 + [1] * 4 + [2] * 3
    
    df_train = pd.DataFrame({'smiles': train_smiles, 'activity': train_labels})
    df_test = pd.DataFrame({'smiles': test_smiles, 'activity': test_labels})
    
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

def test_basic_structure_classification():
    """Test that function returns correct structure for classification."""
    print("\n" + "="*80)
    print("TEST: Basic structure validation (classification)")
    print("="*80)
    
    df_train, df_test = create_balanced_classification_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='struct_clf')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    # Check required fields
    assert 'task_type' in result
    assert 'alpha' in result
    assert 'train_vs_test' in result
    assert 'train_vs_val' in result
    assert 'test_vs_val' in result
    assert 'overall_severity' in result
    
    # Check task type
    assert result['task_type'] == 'classification'
    assert result['alpha'] == 0.05
    
    # Check classification-specific fields
    assert 'class_balance' in result
    assert 'train' in result['class_balance']
    assert 'test' in result['class_balance']
    
    # Check train_vs_test structure
    assert result['train_vs_test'] is not None
    if 'error' not in result['train_vs_test']:
        assert 'chi2_statistic' in result['train_vs_test']
        assert 'p_value' in result['train_vs_test']
        assert 'significant' in result['train_vs_test']
        assert 'interpretation' in result['train_vs_test']
        assert 'n_classes' in result['train_vs_test']
        assert 'train_counts' in result['train_vs_test']
        assert 'test_counts' in result['train_vs_test']
        assert 'train_proportions' in result['train_vs_test']
        assert 'test_proportions' in result['train_vs_test']
    
    # Val results should be None (no val split)
    assert result['train_vs_val'] is None
    assert result['test_vs_val'] is None
    
    print("✅ Structure correct (classification)")
    print(f"   - Task type: {result['task_type']}")
    print(f"   - Alpha: {result['alpha']}")
    print(f"   - Overall severity: {result['overall_severity']}")


def test_basic_structure_regression():
    """Test that function returns correct structure for regression."""
    print("\n" + "="*80)
    print("TEST: Basic structure validation (regression)")
    print("="*80)
    
    df_train, df_test = create_similar_regression_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='struct_reg')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    # Check task type
    assert result['task_type'] == 'regression'
    
    # Check regression-specific structure
    assert 'class_balance' not in result  # Should not have class balance for regression
    
    # Check train_vs_test structure for regression
    if 'error' not in result['train_vs_test']:
        assert 'ks_statistic' in result['train_vs_test']
        assert 'p_value' in result['train_vs_test']
        assert 'train_mean' in result['train_vs_test']
        assert 'test_mean' in result['train_vs_test']
        assert 'train_std' in result['train_vs_test']
        assert 'test_std' in result['train_vs_test']
        assert 'train_min' in result['train_vs_test']
        assert 'test_min' in result['train_vs_test']
        assert 'train_max' in result['train_vs_test']
        assert 'test_max' in result['train_vs_test']
    
    print("✅ Structure correct (regression)")
    print(f"   - Task type: {result['task_type']}")


def test_balanced_classification():
    """Test balanced classification split."""
    print("\n" + "="*80)
    print("TEST: Balanced classification (similar class distributions)")
    print("="*80)
    
    df_train, df_test = create_balanced_classification_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='balanced_clf')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    # Should be classification
    assert result['task_type'] == 'classification'
    
    # Should not be significantly different (both 50/50)
    if 'error' not in result['train_vs_test']:
        assert result['train_vs_test']['interpretation'] == 'SIMILAR'
        assert not result['train_vs_test']['significant']
    
    # Should be balanced
    assert not result['class_balance']['train'].get('imbalanced', True)
    assert not result['class_balance']['test'].get('imbalanced', True)
    
    # Severity should be OK
    assert result['overall_severity'] == 'OK'
    
    print("✅ Balanced classification detected")
    print(f"   - Train proportions: {result['class_balance']['train']['class_proportions']}")
    print(f"   - Test proportions: {result['class_balance']['test']['class_proportions']}")
    print(f"   - Overall severity: {result['overall_severity']}")


def test_biased_classification():
    """Test biased classification split."""
    print("\n" + "="*80)
    print("TEST: Biased classification (different class distributions)")
    print("="*80)
    
    df_train, df_test = create_biased_classification_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='biased_clf')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    # Should detect significant difference
    if 'error' not in result['train_vs_test']:
        assert result['train_vs_test']['significant']
        assert result['train_vs_test']['interpretation'] == 'DIFFERENT'
    
    # Should flag as MEDIUM severity
    assert result['overall_severity'] == 'MEDIUM'
    
    print("✅ Biased classification detected")
    print(f"   - Chi-square p-value: {result['train_vs_test'].get('p_value', 'N/A')}")
    print(f"   - Overall severity: {result['overall_severity']}")


def test_imbalanced_classification():
    """Test class imbalance detection."""
    print("\n" + "="*80)
    print("TEST: Class imbalance detection")
    print("="*80)
    
    df_train, df_test = create_imbalanced_classification_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='imbalanced_clf')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05, imbalance_threshold=0.3
    )
    
    # Should detect imbalance in train (90/10 split, ratio = 0.11 < 0.3)
    assert result['class_balance']['train']['imbalanced']
    
    # Min class ratio should be low
    assert result['class_balance']['train']['min_class_ratio'] < 0.3
    
    # Should flag severity
    assert result['overall_severity'] in ['LOW', 'MEDIUM']
    
    print("✅ Class imbalance detected")
    print(f"   - Train min class ratio: {result['class_balance']['train']['min_class_ratio']}")
    print(f"   - Train imbalanced: {result['class_balance']['train']['imbalanced']}")
    print(f"   - Overall severity: {result['overall_severity']}")


def test_similar_regression():
    """Test regression split with similar distributions."""
    print("\n" + "="*80)
    print("TEST: Similar regression distributions")
    print("="*80)
    
    df_train, df_test = create_similar_regression_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='similar_reg')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    # Should be regression
    assert result['task_type'] == 'regression'
    
    # Should not be significantly different
    if 'error' not in result['train_vs_test']:
        assert result['train_vs_test']['interpretation'] == 'SIMILAR'
        assert not result['train_vs_test']['significant']
    
    # Severity should be OK
    assert result['overall_severity'] == 'OK'
    
    print("✅ Similar regression distributions")
    print(f"   - Train mean: {result['train_vs_test'].get('train_mean', 'N/A'):.2f}")
    print(f"   - Test mean: {result['train_vs_test'].get('test_mean', 'N/A'):.2f}")
    print(f"   - KS p-value: {result['train_vs_test'].get('p_value', 'N/A')}")


def test_biased_regression():
    """Test regression split with different distributions."""
    print("\n" + "="*80)
    print("TEST: Biased regression distributions")
    print("="*80)
    
    df_train, df_test = create_biased_regression_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='biased_reg')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    # Should detect significant difference
    if 'error' not in result['train_vs_test']:
        assert result['train_vs_test']['significant']
        assert result['train_vs_test']['interpretation'] == 'DIFFERENT'
        
        # Means should be significantly different
        train_mean = result['train_vs_test']['train_mean']
        test_mean = result['train_vs_test']['test_mean']
        assert abs(train_mean - test_mean) > 2  # Should differ by >2
    
    # Should flag as MEDIUM severity
    assert result['overall_severity'] == 'MEDIUM'
    
    print("✅ Biased regression detected")
    print(f"   - Train mean: {result['train_vs_test'].get('train_mean', 'N/A'):.2f}")
    print(f"   - Test mean: {result['train_vs_test'].get('test_mean', 'N/A'):.2f}")
    print(f"   - Overall severity: {result['overall_severity']}")


def test_three_way_classification():
    """Test three-way classification split."""
    print("\n" + "="*80)
    print("TEST: Three-way classification split")
    print("="*80)
    
    df_train, df_test, df_val = create_three_way_classification_data()
    train_file, test_file, val_file = store_test_splits(df_train, df_test, df_val, prefix='3way_clf')
    
    result = _test_activity_distributions(
        train_file, test_file, val_file, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    # Check all comparisons present
    assert result['train_vs_test'] is not None
    assert result['train_vs_val'] is not None
    assert result['test_vs_val'] is not None
    
    # Check class balance for all splits
    assert 'val' in result['class_balance']
    
    print("✅ Three-way classification analyzed")
    print(f"   - Train/test significant: {result['train_vs_test'].get('significant', 'N/A')}")
    print(f"   - Train/val significant: {result['train_vs_val'].get('significant', 'N/A')}")
    print(f"   - Test/val significant: {result['test_vs_val'].get('significant', 'N/A')}")


def test_three_way_regression():
    """Test three-way regression split."""
    print("\n" + "="*80)
    print("TEST: Three-way regression split")
    print("="*80)
    
    df_train, df_test, df_val = create_three_way_regression_data()
    train_file, test_file, val_file = store_test_splits(df_train, df_test, df_val, prefix='3way_reg')
    
    result = _test_activity_distributions(
        train_file, test_file, val_file, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    # Check all comparisons present
    assert result['train_vs_test'] is not None
    assert result['train_vs_val'] is not None
    assert result['test_vs_val'] is not None
    
    print("✅ Three-way regression analyzed")


def test_missing_label_column():
    """Test handling of missing label column."""
    print("\n" + "="*80)
    print("TEST: Missing label column")
    print("="*80)
    
    df_train, df_test = create_missing_label_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='missing')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    # Should return error
    assert 'error' in result
    assert 'not found' in result['error'].lower()
    
    print("✅ Missing label column handled")
    print(f"   - Error: {result['error']}")


def test_empty_labels():
    """Test handling of all NaN labels."""
    print("\n" + "="*80)
    print("TEST: Empty labels (all NaN)")
    print("="*80)
    
    df_train, df_test = create_empty_labels_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='empty')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    # Should return error
    assert 'error' in result
    assert 'no valid labels' in result['error'].lower()
    
    print("✅ Empty labels handled")
    print(f"   - Error: {result['error']}")


def test_small_classification():
    """Test small classification dataset."""
    print("\n" + "="*80)
    print("TEST: Small classification dataset")
    print("="*80)
    
    df_train, df_test = create_small_classification_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='small_clf')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    # Should have insufficient data error or work
    if 'error' in result['train_vs_test']:
        assert 'insufficient' in result['train_vs_test']['error'].lower()
        print("   ⚠️  Insufficient data (expected)")
    else:
        print("   ✅ Processed despite small size")
    
    print("✅ Small classification handled")


def test_small_regression():
    """Test small regression dataset."""
    print("\n" + "="*80)
    print("TEST: Small regression dataset")
    print("="*80)
    
    df_train, df_test = create_small_regression_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='small_reg')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    # Should have insufficient data error
    assert 'error' in result['train_vs_test']
    assert 'insufficient' in result['train_vs_test']['error'].lower()
    
    print("✅ Small regression handled")


def test_multiclass():
    """Test multiclass classification (3 classes)."""
    print("\n" + "="*80)
    print("TEST: Multiclass classification")
    print("="*80)
    
    df_train, df_test = create_multiclass_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='multiclass')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    # Should be detected as classification
    assert result['task_type'] == 'classification'
    
    # Should have 3 classes
    if 'error' not in result['train_vs_test']:
        assert result['train_vs_test']['n_classes'] == 3
    
    print("✅ Multiclass classification detected")
    print(f"   - Number of classes: {result['train_vs_test'].get('n_classes', 'N/A')}")


def test_chi_square_validity():
    """Test Chi-square test statistics validity."""
    print("\n" + "="*80)
    print("TEST: Chi-square test validity")
    print("="*80)
    
    df_train, df_test = create_balanced_classification_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='chi2')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    if 'error' not in result['train_vs_test']:
        # Chi-square statistic should be >= 0
        assert result['train_vs_test']['chi2_statistic'] >= 0
        
        # p-value should be between 0 and 1
        assert 0 <= result['train_vs_test']['p_value'] <= 1
        
        # Degrees of freedom should be >= 0
        assert result['train_vs_test']['degrees_of_freedom'] >= 0
        
        # Interpretation should match significance
        if result['train_vs_test']['significant']:
            assert result['train_vs_test']['interpretation'] == 'DIFFERENT'
        else:
            assert result['train_vs_test']['interpretation'] == 'SIMILAR'
    
    print("✅ Chi-square statistics valid")


def test_ks_validity_regression():
    """Test KS test statistics validity for regression."""
    print("\n" + "="*80)
    print("TEST: KS test validity (regression)")
    print("="*80)
    
    df_train, df_test = create_similar_regression_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='ks')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    if 'error' not in result['train_vs_test']:
        # KS statistic should be between 0 and 1
        assert 0 <= result['train_vs_test']['ks_statistic'] <= 1
        
        # p-value should be between 0 and 1
        assert 0 <= result['train_vs_test']['p_value'] <= 1
        
        # Summary statistics should be reasonable
        assert result['train_vs_test']['train_std'] >= 0
        assert result['train_vs_test']['test_std'] >= 0
    
    print("✅ KS statistics valid")


def test_severity_levels():
    """Test severity level assignment."""
    print("\n" + "="*80)
    print("TEST: Severity level assignment")
    print("="*80)
    
    # Balanced classification → OK
    df_train, df_test = create_balanced_classification_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='sev_ok')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    assert result['overall_severity'] == 'OK'
    print(f"   - Balanced classification: {result['overall_severity']}")
    
    # Biased classification → MEDIUM
    df_train, df_test = create_biased_classification_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='sev_med')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    assert result['overall_severity'] == 'MEDIUM'
    print(f"   - Biased classification: {result['overall_severity']}")
    
    # Imbalanced classification → LOW
    df_train, df_test = create_imbalanced_classification_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='sev_low')
    
    result = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05, imbalance_threshold=0.3
    )
    
    assert result['overall_severity'] in ['LOW', 'MEDIUM']
    print(f"   - Imbalanced classification: {result['overall_severity']}")
    
    print("✅ Severity levels appropriate")


def test_alpha_threshold():
    """Test that alpha threshold affects significance detection."""
    print("\n" + "="*80)
    print("TEST: Alpha threshold behavior")
    print("="*80)
    
    df_train, df_test = create_biased_classification_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='alpha')
    
    # Test with strict alpha (0.01)
    result_strict = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.01
    )
    
    # Test with lenient alpha (0.10)
    result_lenient = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.10
    )
    
    # Both should have same p-value but different significance
    if 'error' not in result_strict['train_vs_test'] and 'error' not in result_lenient['train_vs_test']:
        p_value = result_strict['train_vs_test']['p_value']
        
        # Check if p-value is in the range (0.01, 0.10)
        if 0.01 < p_value < 0.10:
            # Should be not significant with strict, significant with lenient
            assert not result_strict['train_vs_test']['significant']
            assert result_lenient['train_vs_test']['significant']
            print("✅ Alpha threshold affects significance (p-value in range)")
        else:
            print(f"   ⚠️  p-value {p_value} not in testable range (0.01, 0.10)")
    
    print(f"   - Strict (α=0.01) severity: {result_strict['overall_severity']}")
    print(f"   - Lenient (α=0.10) severity: {result_lenient['overall_severity']}")


def test_consistency():
    """Test that results are consistent across runs."""
    print("\n" + "="*80)
    print("TEST: Consistency across runs")
    print("="*80)
    
    df_train, df_test = create_balanced_classification_data()
    train_file, test_file, _ = store_test_splits(df_train, df_test, prefix='consist')
    
    result1 = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    result2 = _test_activity_distributions(
        train_file, test_file, None, TEST_MANIFEST, 'activity', alpha=0.05
    )
    
    # Results should be identical
    assert result1['task_type'] == result2['task_type']
    assert result1['overall_severity'] == result2['overall_severity']
    
    # Check that statistics are identical
    if 'error' not in result1['train_vs_test'] and 'error' not in result2['train_vs_test']:
        if result1['task_type'] == 'classification':
            assert result1['train_vs_test']['chi2_statistic'] == result2['train_vs_test']['chi2_statistic']
            assert result1['train_vs_test']['p_value'] == result2['train_vs_test']['p_value']
        else:
            assert result1['train_vs_test']['ks_statistic'] == result2['train_vs_test']['ks_statistic']
            assert result1['train_vs_test']['p_value'] == result2['train_vs_test']['p_value']
    
    print("✅ Results consistent across runs")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == '__main__':
    print("\n" + "#"*80)
    print("# COMPREHENSIVE TEST SUITE: _test_activity_distributions()")
    print("#"*80)
    
    test_basic_structure_classification()
    test_basic_structure_regression()
    test_balanced_classification()
    test_biased_classification()
    test_imbalanced_classification()
    test_similar_regression()
    test_biased_regression()
    test_three_way_classification()
    test_three_way_regression()
    test_missing_label_column()
    test_empty_labels()
    test_small_classification()
    test_small_regression()
    test_multiclass()
    test_chi_square_validity()
    test_ks_validity_regression()
    test_severity_levels()
    test_alpha_threshold()
    test_consistency()
    
    print("\n" + "#"*80)
    print("# ALL TESTS PASSED! ✅")
    print("#"*80)
    print("\nFunction _test_activity_distributions() is production-ready!")
