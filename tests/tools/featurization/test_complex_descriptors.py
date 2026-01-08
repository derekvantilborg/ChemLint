"""Tests for complex_descriptors.py - Molecular fingerprints (ECFP, MACCS, RDKit)."""

import pandas as pd
import numpy as np
import pytest
from molml_mcp.tools.featurization.supported.ecfps import smiles_to_ecfp_dataset
from molml_mcp.tools.featurization.supported.maccs import smiles_to_maccs_dataset
from molml_mcp.tools.featurization.supported.rdkit import smiles_to_rdkit_fp_dataset
from molml_mcp.infrastructure.resources import create_project_manifest, _store_resource, _load_resource


def test_smiles_to_ecfp_dataset(session_workdir, request):
    """Test ECFP (Morgan) fingerprint calculation."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create test dataset
    df = pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1", "CC(C)O"],
        "id": [1, 2, 3]
    })
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Calculate ECFP fingerprints
    result = smiles_to_ecfp_dataset(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        output_filename="ecfp_fingerprints",
        explanation="ECFP test",
        radius=2,
        nbits=2048
    )
    
    # Check return structure
    assert "output_filename" in result
    assert "n_molecules" in result
    assert "fingerprint_type" in result
    assert "radius" in result
    assert "nbits" in result
    
    # Check values
    assert result["n_molecules"] == 3
    assert result["fingerprint_type"] == "ECFP"
    assert result["radius"] == 2
    assert result["nbits"] == 2048
    
    # Load fingerprints and verify structure
    fingerprints = _load_resource(manifest_path, result["output_filename"])
    assert isinstance(fingerprints, dict)
    assert len(fingerprints) == 3
    
    # Check fingerprint dimensions
    for smiles, fp in fingerprints.items():
        assert isinstance(fp, np.ndarray)
        assert fp.shape == (2048,)
        assert fp.dtype in [np.float64, np.float32, np.int64, np.int32]


def test_smiles_to_ecfp_custom_params(session_workdir, request):
    """Test ECFP with custom radius and bit size."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    df = pd.DataFrame({"smiles": ["CCO", "CCC"]})
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Use custom parameters
    result = smiles_to_ecfp_dataset(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        output_filename="ecfp_custom",
        explanation="Custom ECFP",
        radius=3,
        nbits=1024
    )
    
    assert result["radius"] == 3
    assert result["nbits"] == 1024
    
    # Verify fingerprint size
    fingerprints = _load_resource(manifest_path, result["output_filename"])
    for fp in fingerprints.values():
        assert fp.shape == (1024,)


def test_smiles_to_maccs_dataset(session_workdir, request):
    """Test MACCS keys fingerprint calculation."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create test dataset
    df = pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1", "CCN"],
        "name": ["ethanol", "benzene", "ethylamine"]
    })
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Calculate MACCS fingerprints
    result = smiles_to_maccs_dataset(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        output_filename="maccs_fingerprints",
        explanation="MACCS test"
    )
    
    # Check return structure
    assert "output_filename" in result
    assert "n_molecules" in result
    assert "fingerprint_type" in result
    assert "nbits" in result
    
    # Check values
    assert result["n_molecules"] == 3
    assert result["fingerprint_type"] == "MACCS"
    assert result["nbits"] == 167  # MACCS keys are always 167 bits
    
    # Load and verify fingerprints
    fingerprints = _load_resource(manifest_path, result["output_filename"])
    assert len(fingerprints) == 3
    
    # MACCS keys should be 167 bits
    for smiles, fp in fingerprints.items():
        assert isinstance(fp, np.ndarray)
        assert fp.shape == (167,)


def test_smiles_to_rdkit_fp_dataset(session_workdir, request):
    """Test RDKit topological fingerprint calculation."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create test dataset
    df = pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
        "id": [1, 2, 3]
    })
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Calculate RDKit fingerprints
    result = smiles_to_rdkit_fp_dataset(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        output_filename="rdkit_fingerprints",
        explanation="RDKit FP test",
        fp_size=2048
    )
    
    # Check return structure
    assert "output_filename" in result
    assert "n_molecules" in result
    assert "fingerprint_type" in result
    assert "fp_size" in result
    
    # Check values
    assert result["n_molecules"] == 3
    assert result["fingerprint_type"] == "RDKit"
    assert result["fp_size"] == 2048
    
    # Load and verify fingerprints
    fingerprints = _load_resource(manifest_path, result["output_filename"])
    assert len(fingerprints) == 3
    
    for smiles, fp in fingerprints.items():
        assert isinstance(fp, np.ndarray)
        assert fp.shape == (2048,)


def test_rdkit_fp_custom_size(session_workdir, request):
    """Test RDKit fingerprint with custom size."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    df = pd.DataFrame({"smiles": ["CCO"]})
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Use custom fingerprint size
    result = smiles_to_rdkit_fp_dataset(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        output_filename="rdkit_custom",
        explanation="Custom RDKit FP",
        fp_size=512
    )
    
    assert result["fp_size"] == 512
    
    # Verify fingerprint size
    fingerprints = _load_resource(manifest_path, result["output_filename"])
    for fp in fingerprints.values():
        assert fp.shape == (512,)


def test_fingerprint_invalid_column(session_workdir, request):
    """Test error handling for missing SMILES column."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    df = pd.DataFrame({"smiles": ["CCO"], "id": [1]})
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Test ECFP error
    with pytest.raises(ValueError, match="not found in dataset"):
        smiles_to_ecfp_dataset(
            input_filename=input_file,
            project_manifest_path=manifest_path,
            smiles_column="nonexistent",
            output_filename="output",
            explanation="test"
        )
    
    # Test MACCS error
    with pytest.raises(ValueError, match="not found in dataset"):
        smiles_to_maccs_dataset(
            input_filename=input_file,
            project_manifest_path=manifest_path,
            smiles_column="nonexistent",
            output_filename="output",
            explanation="test"
        )
    
    # Test RDKit FP error
    with pytest.raises(ValueError, match="not found in dataset"):
        smiles_to_rdkit_fp_dataset(
            input_filename=input_file,
            project_manifest_path=manifest_path,
            smiles_column="nonexistent",
            output_filename="output",
            explanation="test"
        )
