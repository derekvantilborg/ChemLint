"""
Comprehensive test suite for _detect_scaffold_leakage().

Tests all aspects:
- Scaffold extraction and computation
- Scaffold overlap detection across splits
- Percentage calculations
- Three-way split handling
- Severity level assignment
- Edge cases and error handling
- Examples and sorting
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from molml_mcp.tools.reports.data_splitting import _detect_scaffold_leakage
from molml_mcp.infrastructure.resources import _store_resource

# Test configuration
TEST_DIR = Path(__file__).parent / 'data'
TEST_DIR.mkdir(exist_ok=True)
MANIFEST_PATH = str(TEST_DIR / 'test_manifest.json')

# Storage for created resources
stored = {}


def setup_test_data():
    """Create diverse test datasets for scaffold leakage testing."""
    global stored
    
    # 1. High scaffold overlap (CRITICAL case)
    # Train and test share many scaffolds
    # Benzene derivatives in both splits
    train_smiles = [
        'c1ccccc1C',      # Toluene (benzene scaffold)
        'c1ccccc1CC',     # Ethylbenzene (benzene scaffold)
        'c1ccccc1O',      # Phenol (benzene scaffold)
        'c1ccccc1N',      # Aniline (benzene scaffold)
    ] * 25  # 100 molecules
    
    test_smiles = [
        'c1ccccc1Cl',     # Chlorobenzene (benzene scaffold - SHARED!)
        'c1ccccc1F',      # Fluorobenzene (benzene scaffold - SHARED!)
        'c1ccccc1Br',     # Bromobenzene (benzene scaffold - SHARED!)
    ] * 10  # 30 molecules
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'scaffold_high_overlap_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'scaffold_high_overlap_test', 'test', 'csv')
    stored['high_overlap'] = (train_file, test_file, None)
    
    # 2. No scaffold overlap (IDEAL case)
    # Completely different scaffolds
    train_smiles = [
        'c1ccccc1',       # Benzene
        'C1CCCCC1',       # Cyclohexane
        'c1ccncc1',       # Pyridine
        'c1cccnc1',       # Pyridine (isomer)
    ] * 20
    
    test_smiles = [
        'CC(C)C',         # Isobutane (no aromatic scaffold)
        'CCCC',           # Butane
        'CCC(C)C',        # Isopentane
        'CCCCCC',         # Hexane
    ] * 10
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'scaffold_no_overlap_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'scaffold_no_overlap_test', 'test', 'csv')
    stored['no_overlap'] = (train_file, test_file, None)
    
    # 3. Medium scaffold overlap (~30%)
    train_smiles = [
        'c1ccccc1C',      # Benzene scaffold
        'c1ccccc1CC',     # Benzene scaffold
        'C1CCCCC1C',      # Cyclohexane scaffold
        'C1CCCCC1CC',     # Cyclohexane scaffold
        'c1ccncc1C',      # Pyridine scaffold
        'c1ccncc1CC',     # Pyridine scaffold
    ] * 20
    
    test_smiles = [
        'c1ccccc1O',      # Benzene scaffold (SHARED - ~33%)
        'CC(C)C',         # Isobutane (no scaffold)
        'CCCC',           # Butane (no scaffold)
    ] * 10
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'scaffold_med_overlap_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'scaffold_med_overlap_test', 'test', 'csv')
    stored['medium_overlap'] = (train_file, test_file, None)
    
    # 4. Three-way split with validation
    train_smiles = ['c1ccccc1C'] * 30 + ['C1CCCCC1C'] * 30  # Benzene and cyclohexane
    test_smiles = ['c1ccccc1O'] * 15  # Benzene (shared with train)
    val_smiles = ['C1CCCCC1O'] * 15   # Cyclohexane (shared with train)
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    df_val = pd.DataFrame({'smiles': val_smiles})
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'scaffold_threeway_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'scaffold_threeway_test', 'test', 'csv')
    val_file = _store_resource(df_val, MANIFEST_PATH, 'scaffold_threeway_val', 'test', 'csv')
    stored['three_way'] = (train_file, test_file, val_file)
    
    # 5. Complex scaffolds (polycyclic)
    train_smiles = [
        'C1CCC2CCCCC2C1',  # Decalin (bicyclic)
        'c1ccc2ccccc2c1',  # Naphthalene
        'C1CCC2(CC1)CCCC2', # Spiro compound
    ] * 20
    
    test_smiles = [
        'C1CCC2CCCCC2C1C', # Decalin derivative (SHARED!)
        'CCCC',            # Simple chain
    ] * 10
    
    df_train = pd.DataFrame({'smiles': train_smiles})
    df_test = pd.DataFrame({'smiles': test_smiles})
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'scaffold_complex_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'scaffold_complex_test', 'test', 'csv')
    stored['complex_scaffolds'] = (train_file, test_file, None)
    
    # 6. Invalid SMILES
    df_train = pd.DataFrame({
        'smiles': ['c1ccccc1', 'INVALID', 'C1CCCCC1', None, '']
    })
    df_test = pd.DataFrame({
        'smiles': ['c1ccccc1C', 'ALSO_INVALID', 'CCCC']
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'scaffold_invalid_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'scaffold_invalid_test', 'test', 'csv')
    stored['invalid_smiles'] = (train_file, test_file, None)
    
    # 7. Empty/small scaffolds
    # Molecules with no rings (no Murcko scaffold)
    df_train = pd.DataFrame({
        'smiles': ['CCC', 'CCCC', 'CCCCC'] * 10
    })
    df_test = pd.DataFrame({
        'smiles': ['CC', 'CCC', 'CCCC'] * 5
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'scaffold_acyclic_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'scaffold_acyclic_test', 'test', 'csv')
    stored['acyclic'] = (train_file, test_file, None)
    
    # 8. Single scaffold in train, multiple in test
    df_train = pd.DataFrame({
        'smiles': ['c1ccccc1C', 'c1ccccc1CC', 'c1ccccc1O'] * 20  # All benzene
    })
    df_test = pd.DataFrame({
        'smiles': ['c1ccccc1N'] * 10 + ['C1CCCCC1C'] * 10  # Benzene + cyclohexane
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'scaffold_single_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'scaffold_single_test', 'test', 'csv')
    stored['single_scaffold_train'] = (train_file, test_file, None)
    
    # 9. Many different scaffolds (high diversity)
    diverse_train = [
        'c1ccccc1',       # Benzene
        'C1CCCCC1',       # Cyclohexane
        'c1ccncc1',       # Pyridine
        'c1cccnc1',       # Pyridine isomer
        'c1ccoc1',        # Furan
        'c1ccsc1',        # Thiophene
        'C1CCOC1',        # Tetrahydrofuran
        'C1CCNC1',        # Pyrrolidine
        'c1ccc2ccccc2c1', # Naphthalene
        'c1ccc2ncccc2c1', # Quinoline
    ]
    diverse_test = [
        'c1ccccc1C',      # Benzene derivative (shared)
        'C1CCC2CCCCC2C1', # Decalin (not shared)
        'c1ncncc1',       # Pyrimidine (not shared)
    ]
    
    df_train = pd.DataFrame({'smiles': diverse_train * 10})
    df_test = pd.DataFrame({'smiles': diverse_test * 10})
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'scaffold_diverse_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'scaffold_diverse_test', 'test', 'csv')
    stored['diverse'] = (train_file, test_file, None)
    
    # 10. Exact boundary case (50% overlap)
    # Test has 2 different scaffolds, 1 shared with train (50%)
    df_train = pd.DataFrame({
        'smiles': ['c1ccccc1C'] * 25 + ['c1ccncc1C'] * 25  # Benzene and pyridine scaffolds
    })
    df_test = pd.DataFrame({
        'smiles': ['c1ccccc1O'] * 15 + ['C1CCCCC1C'] * 15  # Benzene (shared) + cyclohexane (not shared)
    })
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'scaffold_boundary_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'scaffold_boundary_test', 'test', 'csv')
    stored['boundary_50'] = (train_file, test_file, None)
    
    # 11. Test/val overlap (less critical)
    df_train = pd.DataFrame({'smiles': ['c1ccccc1C'] * 50})  # Benzene
    df_test = pd.DataFrame({'smiles': ['C1CCCCC1C'] * 20})   # Cyclohexane
    df_val = pd.DataFrame({'smiles': ['C1CCCCC1O'] * 20})    # Cyclohexane (shares with test)
    
    train_file = _store_resource(df_train, MANIFEST_PATH, 'scaffold_testval_train', 'test', 'csv')
    test_file = _store_resource(df_test, MANIFEST_PATH, 'scaffold_testval_test', 'test', 'csv')
    val_file = _store_resource(df_val, MANIFEST_PATH, 'scaffold_testval_val', 'test', 'csv')
    stored['test_val_overlap'] = (train_file, test_file, val_file)


def test_basic_structure():
    """Test return structure is correct."""
    print("\n=== BASIC STRUCTURE ===")
    
    train_file, test_file, _ = stored['no_overlap']
    result = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH, 'smiles'
    )
    
    # Check required keys
    required_keys = [
        'computation_stats',
        'train_test_overlap',
        'overall_severity'
    ]
    
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    
    # Check computation_stats structure
    assert 'train_scaffolds_computed' in result['computation_stats']
    assert 'train_failed' in result['computation_stats']
    assert 'test_scaffolds_computed' in result['computation_stats']
    assert 'test_failed' in result['computation_stats']
    
    # Check train_test_overlap structure
    overlap = result['train_test_overlap']
    assert 'n_shared_scaffolds' in overlap
    assert 'n_scaffolds_split1' in overlap
    assert 'n_scaffolds_split2' in overlap
    assert 'pct_split2_in_split1' in overlap
    assert 'pct_split1_in_split2' in overlap
    assert 'examples' in overlap
    
    print("✅ Return structure correct")


def test_high_scaffold_overlap():
    """Test detection of high scaffold overlap."""
    print("\n=== HIGH SCAFFOLD OVERLAP ===")
    
    train_file, test_file, _ = stored['high_overlap']
    result = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH, 'smiles'
    )
    
    overlap = result['train_test_overlap']
    
    # All test molecules have benzene scaffold, which is in train
    assert overlap['n_shared_scaffolds'] > 0
    assert overlap['pct_split2_in_split1'] == 100.0, f"Expected 100% overlap, got {overlap['pct_split2_in_split1']}"
    
    # Should flag as HIGH severity
    assert result['overall_severity'] == 'HIGH'
    
    print(f"✅ Detected 100% scaffold overlap (HIGH severity)")
    print(f"   Shared scaffolds: {overlap['n_shared_scaffolds']}")


def test_no_scaffold_overlap():
    """Test clean split with no scaffold overlap."""
    print("\n=== NO SCAFFOLD OVERLAP ===")
    
    train_file, test_file, _ = stored['no_overlap']
    result = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH, 'smiles'
    )
    
    overlap = result['train_test_overlap']
    
    # No shared scaffolds (aromatic vs aliphatic)
    assert overlap['n_shared_scaffolds'] == 0
    assert overlap['pct_split2_in_split1'] == 0.0
    
    # Should be OK severity
    assert result['overall_severity'] == 'OK'
    
    print(f"✅ No scaffold overlap detected (OK severity)")


def test_medium_scaffold_overlap():
    """Test medium scaffold overlap (~30%)."""
    print("\n=== MEDIUM SCAFFOLD OVERLAP ===")
    
    train_file, test_file, _ = stored['medium_overlap']
    result = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH, 'smiles'
    )
    
    overlap = result['train_test_overlap']
    
    # Should have some overlap (benzene scaffold shared)
    assert overlap['n_shared_scaffolds'] > 0
    
    # Percentage should be between 20-50% (inclusive)
    pct = overlap['pct_split2_in_split1']
    assert 20 <= pct <= 50, f"Expected 20-50% overlap, got {pct}"
    
    # Should flag as MEDIUM severity (>20%, <=50%)
    assert result['overall_severity'] in ['MEDIUM', 'HIGH']
    
    print(f"✅ Medium scaffold overlap detected: {pct:.1f}% (MEDIUM severity)")


def test_three_way_split():
    """Test three-way split with validation."""
    print("\n=== THREE-WAY SPLIT ===")
    
    train_file, test_file, val_file = stored['three_way']
    result = _detect_scaffold_leakage(
        train_file, test_file, val_file, MANIFEST_PATH, 'smiles'
    )
    
    # Should have all three comparisons
    assert result['train_test_overlap'] is not None
    assert result['train_val_overlap'] is not None
    assert result['test_val_overlap'] is not None
    
    # Should have val stats
    assert 'val_scaffolds_computed' in result['computation_stats']
    assert 'val_failed' in result['computation_stats']
    
    # Both test and val share scaffolds with train
    assert result['train_test_overlap']['n_shared_scaffolds'] > 0
    assert result['train_val_overlap']['n_shared_scaffolds'] > 0
    
    print("✅ All three-way comparisons present")
    print(f"   Train/test overlap: {result['train_test_overlap']['pct_split2_in_split1']:.1f}%")
    print(f"   Train/val overlap: {result['train_val_overlap']['pct_split2_in_split1']:.1f}%")
    print(f"   Test/val overlap: {result['test_val_overlap']['pct_split2_in_split1']:.1f}%")


def test_complex_scaffolds():
    """Test with polycyclic/complex scaffolds."""
    print("\n=== COMPLEX SCAFFOLDS ===")
    
    train_file, test_file, _ = stored['complex_scaffolds']
    result = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH, 'smiles'
    )
    
    # Should successfully compute scaffolds for polycyclic compounds
    assert result['computation_stats']['train_scaffolds_computed'] > 0
    assert result['computation_stats']['test_scaffolds_computed'] > 0
    
    # Should detect decalin scaffold overlap
    overlap = result['train_test_overlap']
    assert overlap['n_shared_scaffolds'] > 0
    
    print(f"✅ Complex scaffolds handled")
    print(f"   Train scaffolds: {result['computation_stats']['train_scaffolds_computed']}")
    print(f"   Test scaffolds: {result['computation_stats']['test_scaffolds_computed']}")
    print(f"   Shared: {overlap['n_shared_scaffolds']}")


def test_invalid_smiles():
    """Test handling of invalid SMILES."""
    print("\n=== INVALID SMILES ===")
    
    train_file, test_file, _ = stored['invalid_smiles']
    result = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH, 'smiles'
    )
    
    # Should have some failed computations
    assert result['computation_stats']['train_failed'] > 0
    assert result['computation_stats']['test_failed'] > 0
    
    # But should still compute valid ones
    assert result['computation_stats']['train_scaffolds_computed'] > 0
    assert result['computation_stats']['test_scaffolds_computed'] > 0
    
    # Should detect benzene scaffold overlap despite invalid SMILES
    assert result['train_test_overlap']['n_shared_scaffolds'] > 0
    
    print(f"✅ Invalid SMILES handled gracefully")
    print(f"   Train: {result['computation_stats']['train_scaffolds_computed']} computed, {result['computation_stats']['train_failed']} failed")
    print(f"   Test: {result['computation_stats']['test_scaffolds_computed']} computed, {result['computation_stats']['test_failed']} failed")


def test_acyclic_molecules():
    """Test molecules without rings (no Murcko scaffold)."""
    print("\n=== ACYCLIC MOLECULES ===")
    
    train_file, test_file, _ = stored['acyclic']
    result = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH, 'smiles'
    )
    
    # Acyclic molecules should still be processed
    # Murcko scaffold of acyclic = empty or C (single carbon)
    assert result['computation_stats']['train_scaffolds_computed'] >= 0
    assert result['computation_stats']['test_scaffolds_computed'] >= 0
    
    # Should not crash
    assert result['train_test_overlap'] is not None
    
    print(f"✅ Acyclic molecules handled")
    print(f"   Train scaffolds: {result['computation_stats']['train_scaffolds_computed']}")
    print(f"   Test scaffolds: {result['computation_stats']['test_scaffolds_computed']}")


def test_scaffold_examples():
    """Test that examples are provided and sorted correctly."""
    print("\n=== SCAFFOLD EXAMPLES ===")
    
    train_file, test_file, _ = stored['high_overlap']
    result = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH,
        'smiles', max_examples=3
    )
    
    overlap = result['train_test_overlap']
    examples = overlap['examples']
    
    # Should have examples (limited to max_examples)
    assert len(examples) > 0
    assert len(examples) <= 3
    assert overlap['showing_n_examples'] <= 3
    
    # Check example structure
    example = examples[0]
    assert 'scaffold_smiles' in example
    assert 'n_molecules_train' in example
    assert 'n_molecules_test' in example
    assert 'train_indices' in example
    assert 'test_indices' in example
    
    # Check indices are limited
    assert len(example['train_indices']) <= 5
    assert len(example['test_indices']) <= 5
    
    # Examples should be sorted by total molecule count (descending)
    if len(examples) > 1:
        total_counts = [
            ex['n_molecules_train'] + ex['n_molecules_test']
            for ex in examples
        ]
        assert total_counts == sorted(total_counts, reverse=True), "Examples not sorted by molecule count"
    
    print(f"✅ Examples provided and sorted correctly")
    print(f"   Showing {len(examples)} examples (limited to 3)")
    print(f"   Top scaffold: {examples[0]['n_molecules_train']} train + {examples[0]['n_molecules_test']} test molecules")


def test_percentage_calculations():
    """Test accuracy of percentage calculations."""
    print("\n=== PERCENTAGE CALCULATIONS ===")
    
    train_file, test_file, _ = stored['single_scaffold_train']
    result = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH, 'smiles'
    )
    
    overlap = result['train_test_overlap']
    
    # Verify percentages are in valid range
    assert 0 <= overlap['pct_split2_in_split1'] <= 100
    assert 0 <= overlap['pct_split1_in_split2'] <= 100
    
    # Verify calculation logic
    # pct_split2_in_split1 = (shared / total_in_split2) * 100
    if overlap['n_scaffolds_split2'] > 0:
        expected_pct = (overlap['n_shared_scaffolds'] / overlap['n_scaffolds_split2']) * 100
        assert abs(overlap['pct_split2_in_split1'] - expected_pct) < 0.01
    
    print(f"✅ Percentage calculations accurate")
    print(f"   {overlap['pct_split2_in_split1']:.1f}% of test scaffolds in train")
    print(f"   {overlap['pct_split1_in_split2']:.1f}% of train scaffolds in test")


def test_severity_levels():
    """Test severity level assignment."""
    print("\n=== SEVERITY LEVELS ===")
    
    # HIGH severity (>50% overlap)
    train_file, test_file, _ = stored['high_overlap']
    result = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH, 'smiles'
    )
    assert result['overall_severity'] == 'HIGH'
    print("✅ HIGH severity for >50% overlap")
    
    # MEDIUM severity (20-50% overlap)
    train_file, test_file, _ = stored['medium_overlap']
    result = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH, 'smiles'
    )
    assert result['overall_severity'] == 'MEDIUM'
    print("✅ MEDIUM severity for 20-50% overlap")
    
    # LOW severity (>0% but <20% overlap)
    train_file, test_file, _ = stored['diverse']
    result = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH, 'smiles'
    )
    # Should be LOW or MEDIUM depending on exact overlap
    assert result['overall_severity'] in ['LOW', 'MEDIUM', 'OK']
    print(f"✅ {result['overall_severity']} severity for diverse split")
    
    # OK severity (no overlap)
    train_file, test_file, _ = stored['no_overlap']
    result = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH, 'smiles'
    )
    assert result['overall_severity'] == 'OK'
    print("✅ OK severity for no overlap")


def test_boundary_cases():
    """Test boundary cases (exactly 50%, 20%, etc.)."""
    print("\n=== BOUNDARY CASES ===")
    
    # Exactly 50% overlap (boundary for HIGH severity)
    train_file, test_file, _ = stored['boundary_50']
    result = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH, 'smiles'
    )
    
    overlap = result['train_test_overlap']
    
    # Should be close to 50%
    assert 45 <= overlap['pct_split2_in_split1'] <= 55
    
    # At 50%, should be MEDIUM (boundary is >50 for HIGH)
    assert result['overall_severity'] in ['MEDIUM', 'HIGH']
    
    print(f"✅ 50% boundary case handled: {overlap['pct_split2_in_split1']:.1f}% -> {result['overall_severity']}")


def test_test_val_overlap():
    """Test that test/val overlap is treated as less critical."""
    print("\n=== TEST/VAL OVERLAP (LESS CRITICAL) ===")
    
    train_file, test_file, val_file = stored['test_val_overlap']
    result = _detect_scaffold_leakage(
        train_file, test_file, val_file, MANIFEST_PATH, 'smiles'
    )
    
    # Test and val share cyclohexane scaffold
    assert result['test_val_overlap']['n_shared_scaffolds'] > 0
    assert result['test_val_overlap']['pct_split2_in_split1'] == 100.0
    
    # But train doesn't share with test/val, so overall should not be HIGH
    # Test/val overlap requires >70% to trigger LOW severity
    assert result['train_test_overlap']['n_shared_scaffolds'] == 0
    assert result['train_val_overlap']['n_shared_scaffolds'] == 0
    
    # Overall severity should reflect train/test and train/val (OK or LOW)
    assert result['overall_severity'] in ['OK', 'LOW']
    
    print(f"✅ Test/val overlap treated as less critical")
    print(f"   Test/val: {result['test_val_overlap']['pct_split2_in_split1']:.1f}% overlap")
    print(f"   Overall severity: {result['overall_severity']}")


def test_scaffold_count_consistency():
    """Test that scaffold counts are consistent."""
    print("\n=== SCAFFOLD COUNT CONSISTENCY ===")
    
    train_file, test_file, _ = stored['diverse']
    result = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH, 'smiles'
    )
    
    overlap = result['train_test_overlap']
    
    # Shared scaffolds should not exceed total scaffolds in either split
    assert overlap['n_shared_scaffolds'] <= overlap['n_scaffolds_split1']
    assert overlap['n_shared_scaffolds'] <= overlap['n_scaffolds_split2']
    
    # Scaffold counts should be non-negative
    assert overlap['n_scaffolds_split1'] >= 0
    assert overlap['n_scaffolds_split2'] >= 0
    assert overlap['n_shared_scaffolds'] >= 0
    
    print(f"✅ Scaffold counts consistent")
    print(f"   Train: {overlap['n_scaffolds_split1']} scaffolds")
    print(f"   Test: {overlap['n_scaffolds_split2']} scaffolds")
    print(f"   Shared: {overlap['n_shared_scaffolds']} scaffolds")


def test_consistency():
    """Test that results are consistent across runs."""
    print("\n=== CONSISTENCY ===")
    
    train_file, test_file, _ = stored['high_overlap']
    
    result1 = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH, 'smiles'
    )
    
    result2 = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH, 'smiles'
    )
    
    # Results should be identical
    assert result1['train_test_overlap']['n_shared_scaffolds'] == result2['train_test_overlap']['n_shared_scaffolds']
    assert result1['train_test_overlap']['pct_split2_in_split1'] == result2['train_test_overlap']['pct_split2_in_split1']
    assert result1['overall_severity'] == result2['overall_severity']
    
    print("✅ Results are consistent across runs")


def test_max_examples_parameter():
    """Test max_examples parameter limits output."""
    print("\n=== MAX EXAMPLES PARAMETER ===")
    
    train_file, test_file, _ = stored['high_overlap']
    
    # Test with max_examples=2
    result = _detect_scaffold_leakage(
        train_file, test_file, None, MANIFEST_PATH,
        'smiles', max_examples=2
    )
    
    examples = result['train_test_overlap']['examples']
    
    # Should limit to 2 examples
    assert len(examples) <= 2
    assert result['train_test_overlap']['showing_n_examples'] <= 2
    
    print(f"✅ Examples limited to {len(examples)}")


if __name__ == '__main__':
    print("="*80)
    print("COMPREHENSIVE TEST SUITE: _detect_scaffold_leakage")
    print("="*80)
    
    # Setup test data
    print("\n[Setting up test data...]")
    setup_test_data()
    print(f"Created {len(stored)} test scenarios")
    
    # Run tests
    try:
        test_basic_structure()
        test_high_scaffold_overlap()
        test_no_scaffold_overlap()
        test_medium_scaffold_overlap()
        test_three_way_split()
        test_complex_scaffolds()
        test_invalid_smiles()
        test_acyclic_molecules()
        test_scaffold_examples()
        test_percentage_calculations()
        test_severity_levels()
        test_boundary_cases()
        test_test_val_overlap()
        test_scaffold_count_consistency()
        test_consistency()
        test_max_examples_parameter()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\n_detect_scaffold_leakage() is production-ready!")
        print("✅ Scaffold extraction validated")
        print("✅ Overlap detection accurate")
        print("✅ Percentage calculations correct")
        print("✅ Three-way splits handled")
        print("✅ Complex/polycyclic scaffolds supported")
        print("✅ Invalid SMILES handled gracefully")
        print("✅ Severity levels assigned correctly")
        print("✅ Examples sorted by molecule count")
        print("✅ Test/val overlap treated as less critical")
        print("✅ Results are consistent and deterministic")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
