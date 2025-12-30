"""
Tests for generate_split_quality_text_report()

This tests the text report generation from JSON analysis output.
"""
import pytest
import pandas as pd
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from molml_mcp.infrastructure.resources import _store_resource, _load_resource
from molml_mcp.tools.reports.data_splitting import generate_split_quality_text_report


@pytest.fixture
def temp_manifest():
    """Create temporary manifest for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.json"
        # Initialize manifest with proper structure
        manifest_path.write_text('{"resources": []}')
        yield str(manifest_path)


def test_basic_text_report_generation(temp_manifest):
    """Test basic text report generation from datasets."""
    # Create small train/test datasets
    train_smiles = ['c1ccccc1', 'CCO', 'CC(=O)C', 'CCN', 'CCC']
    train_labels = [0, 1, 0, 1, 0]
    train_df = pd.DataFrame({'smiles': train_smiles, 'label': train_labels})
    
    test_smiles = ['CCCC', 'CCCN', 'c1ccc(C)cc1']
    test_labels = [1, 0, 1]
    test_df = pd.DataFrame({'smiles': test_smiles, 'label': test_labels})
    
    # Store datasets
    train_file = _store_resource(train_df, temp_manifest, "train", "Train set", "csv")
    test_file = _store_resource(test_df, temp_manifest, "test", "Test set", "csv")
    
    # Generate text report directly from datasets
    result = generate_split_quality_text_report(
        train_path=train_file,
        test_path=test_file,
        val_path=None,
        project_manifest_path=temp_manifest,
        smiles_col='smiles',
        label_col='label',
        output_filename='quality_report_text',
        explanation="Test text report"
    )
    
    # Validate return structure
    assert 'output_filename' in result
    assert 'json_report_filename' in result
    assert 'n_lines' in result
    assert 'overall_severity' in result
    assert 'report_sections' in result
    assert 'issues_found' in result
    assert result['output_filename'].endswith('.txt')
    assert result['json_report_filename'].endswith('.json')
    assert result['n_lines'] > 50  # Should have substantial content
    assert len(result['report_sections']) == 10  # 10 sections


def test_text_report_content_structure(temp_manifest):
    """Test that text report contains expected sections."""
    # Create datasets
    train_df = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CCO', 'CC(=O)C'],
        'label': [0, 1, 0]
    })
    test_df = pd.DataFrame({
        'smiles': ['CCCC', 'CCCN'],
        'label': [1, 0]
    })
    
    train_file = _store_resource(train_df, temp_manifest, "train", "Train", "csv")
    test_file = _store_resource(test_df, temp_manifest, "test", "Test", "csv")
    
    # Generate text report
    text_result = generate_split_quality_text_report(
        train_path=train_file,
        test_path=test_file,
        val_path=None,
        project_manifest_path=temp_manifest,
        smiles_col='smiles',
        label_col='label',
        output_filename='quality_report_text'
    )
    
    # Load the text report
    text_content = _load_resource(temp_manifest, text_result['output_filename'])
    
    # Check for key section headers
    assert "DATA SPLITTING QUALITY ANALYSIS REPORT" in text_content
    assert "OVERALL ASSESSMENT" in text_content
    assert "1. SPLIT CHARACTERISTICS" in text_content
    assert "2. EXACT DUPLICATE DETECTION" in text_content
    assert "3. SIMILARITY-BASED LEAKAGE" in text_content
    assert "4. SCAFFOLD OVERLAP" in text_content
    assert "5. STEREOISOMER & TAUTOMER LEAKAGE" in text_content
    assert "6. PHYSICOCHEMICAL PROPERTY DISTRIBUTIONS" in text_content
    assert "7. ACTIVITY/LABEL DISTRIBUTIONS" in text_content
    assert "8. FUNCTIONAL GROUP DISTRIBUTION" in text_content
    assert "END OF REPORT" in text_content


def test_text_report_metadata_section(temp_manifest):
    """Test that metadata section is properly formatted."""
    train_df = pd.DataFrame({'smiles': ['c1ccccc1', 'CCO'], 'label': [0, 1]})
    test_df = pd.DataFrame({'smiles': ['CCCC'], 'label': [1]})
    
    train_file = _store_resource(train_df, temp_manifest, "train", "Train", "csv")
    test_file = _store_resource(test_df, temp_manifest, "test", "Test", "csv")
    
    text_result = generate_split_quality_text_report(
        train_path=train_file,
        test_path=test_file,
        val_path=None,
        project_manifest_path=temp_manifest,
        smiles_col='smiles',
        label_col='label',
        output_filename='report_text'
    )
    
    text_content = _load_resource(temp_manifest, text_result['output_filename'])
    
    # Check metadata fields
    assert "Analysis Date:" in text_content
    assert "Execution Time:" in text_content
    assert "Train File:" in text_content
    assert "Test File:" in text_content
    assert "SMILES Column:" in text_content
    assert "Label Column:" in text_content


def test_text_report_severity_indicators(temp_manifest):
    """Test that severity indicators are present."""
    train_df = pd.DataFrame({'smiles': ['c1ccccc1', 'CCO'], 'label': [0, 1]})
    test_df = pd.DataFrame({'smiles': ['CCCC'], 'label': [1]})
    
    train_file = _store_resource(train_df, temp_manifest, "train", "Train", "csv")
    test_file = _store_resource(test_df, temp_manifest, "test", "Test", "csv")
    
    text_result = generate_split_quality_text_report(
        train_path=train_file,
        test_path=test_file,
        val_path=None,
        project_manifest_path=temp_manifest,
        smiles_col='smiles',
        label_col='label',
        output_filename='report_text'
    )
    
    text_content = _load_resource(temp_manifest, text_result['output_filename'])
    
    # Check severity indicators
    assert "Overall Severity:" in text_content
    assert "Severity Breakdown:" in text_content
    assert "CRITICAL:" in text_content
    assert "HIGH:" in text_content
    assert "MEDIUM:" in text_content
    assert "LOW:" in text_content
    assert "OK:" in text_content


def test_text_report_with_critical_issues(temp_manifest):
    """Test text report when CRITICAL issues are detected."""
    # Create datasets with exact duplicates (CRITICAL)
    train_df = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CCO', 'CC(=O)C'],
        'label': [0, 1, 0]
    })
    test_df = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CCCC'],  # benzene is duplicate
        'label': [1, 0]
    })
    
    train_file = _store_resource(train_df, temp_manifest, "train", "Train", "csv")
    test_file = _store_resource(test_df, temp_manifest, "test", "Test", "csv")
    
    text_result = generate_split_quality_text_report(
        train_path=train_file,
        test_path=test_file,
        val_path=None,
        project_manifest_path=temp_manifest,
        smiles_col='smiles',
        label_col='label',
        output_filename='critical_report_text'
    )
    
    text_content = _load_resource(temp_manifest, text_result['output_filename'])
    
    # Check for CRITICAL severity
    assert "CRITICAL" in text_content
    assert "🔴 CRITICAL" in text_content or "CRITICAL:" in text_content
    assert "duplicate" in text_content.lower()
    assert "c1ccccc1" in text_content  # benzene should be mentioned


def test_text_report_with_validation_set(temp_manifest):
    """Test text report with three-way split."""
    train_df = pd.DataFrame({'smiles': ['c1ccccc1', 'CCO'], 'label': [0, 1]})
    test_df = pd.DataFrame({'smiles': ['CCCC'], 'label': [1]})
    val_df = pd.DataFrame({'smiles': ['CCCN'], 'label': [0]})
    
    train_file = _store_resource(train_df, temp_manifest, "train", "Train", "csv")
    test_file = _store_resource(test_df, temp_manifest, "test", "Test", "csv")
    val_file = _store_resource(val_df, temp_manifest, "val", "Val", "csv")
    
    text_result = generate_split_quality_text_report(
        train_path=train_file,
        test_path=test_file,
        val_path=val_file,
        project_manifest_path=temp_manifest,
        smiles_col='smiles',
        label_col='label',
        output_filename='three_way_report_text'
    )
    
    text_content = _load_resource(temp_manifest, text_result['output_filename'])
    
    # Check val set is mentioned (split characteristics might have N/A but Val File should be listed)
    assert "Val File:" in text_content


def test_text_report_split_characteristics_section(temp_manifest):
    """Test split characteristics section formatting."""
    train_df = pd.DataFrame({
        'smiles': ['c1ccccc1'] * 40 + ['CCO'] * 40,
        'label': [0] * 40 + [1] * 40
    })
    test_df = pd.DataFrame({
        'smiles': ['CCCC'] * 10 + ['CCCN'] * 10,
        'label': [0] * 10 + [1] * 10
    })
    
    train_file = _store_resource(train_df, temp_manifest, "train", "Train", "csv")
    test_file = _store_resource(test_df, temp_manifest, "test", "Test", "csv")
    
    text_result = generate_split_quality_text_report(
        train_path=train_file,
        test_path=test_file,
        val_path=None,
        project_manifest_path=temp_manifest,
        smiles_col='smiles',
        label_col='label',
        output_filename='split_char_report_text'
    )
    
    text_content = _load_resource(temp_manifest, text_result['output_filename'])
    
    # Check split characteristics details are present (some might be N/A)
    assert "Train Size:" in text_content
    assert "Test Size:" in text_content
    assert "Task Type: Classification" in text_content
    assert "Class Distribution" in text_content or "N/A" in text_content


def test_text_report_readable_format(temp_manifest):
    """Test that report is human-readable (has proper spacing, headers)."""
    train_df = pd.DataFrame({'smiles': ['c1ccccc1', 'CCO'], 'label': [0, 1]})
    test_df = pd.DataFrame({'smiles': ['CCCC'], 'label': [1]})
    
    train_file = _store_resource(train_df, temp_manifest, "train", "Train", "csv")
    test_file = _store_resource(test_df, temp_manifest, "test", "Test", "csv")
    
    text_result = generate_split_quality_text_report(
        train_path=train_file,
        test_path=test_file,
        val_path=None,
        project_manifest_path=temp_manifest,
        smiles_col='smiles',
        label_col='label',
        output_filename='readable_report_text'
    )
    
    text_content = _load_resource(temp_manifest, text_result['output_filename'])
    
    # Check formatting elements
    assert "=" * 80 in text_content  # Section dividers
    assert "\n\n" in text_content  # Paragraph spacing
    assert "  - " in text_content or "    - " in text_content  # Indentation
    lines = text_content.split("\n")
    assert len(lines) > 100  # Should be substantial


def test_text_report_captures_all_checks(temp_manifest):
    """Test that text report includes all 8 quality checks."""
    train_df = pd.DataFrame({'smiles': ['c1ccccc1', 'CCO'], 'label': [0, 1]})
    test_df = pd.DataFrame({'smiles': ['CCCC'], 'label': [1]})
    
    train_file = _store_resource(train_df, temp_manifest, "train", "Train", "csv")
    test_file = _store_resource(test_df, temp_manifest, "test", "Test", "csv")
    
    text_result = generate_split_quality_text_report(
        train_path=train_file,
        test_path=test_file,
        val_path=None,
        project_manifest_path=temp_manifest,
        smiles_col='smiles',
        label_col='label',
        output_filename='all_checks_report_text'
    )
    
    text_content = _load_resource(temp_manifest, text_result['output_filename'])
    
    # Verify all 8 checks are present (by counting "Overall Severity:" mentions in check sections)
    # Note: There's also an "Overall Severity:" in the summary section at the top
    overall_severity_mentions = text_content.count("Overall Severity:")
    assert overall_severity_mentions == 9, f"Expected 9 'Overall Severity:' mentions (1 summary + 8 checks), found {overall_severity_mentions}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
