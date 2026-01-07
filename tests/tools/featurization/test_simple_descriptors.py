"""Tests for simple_descriptors.py - RDKit descriptor calculation."""

import pandas as pd
import pytest
from molml_mcp.tools.featurization.simple_descriptors import (
    list_rdkit_descriptors,
    calculate_simple_descriptors,
)
from molml_mcp.infrastructure.resources import create_project_manifest, _store_resource


def test_list_rdkit_descriptors():
    """Test listing available RDKit descriptors."""
    descriptors = list_rdkit_descriptors()
    
    # Should return a non-empty list
    assert isinstance(descriptors, list)
    assert len(descriptors) > 100  # RDKit has 200+ descriptors
    
    # Each item should have required keys
    for desc in descriptors:
        assert "descriptor name" in desc
        assert "explanation" in desc
        assert isinstance(desc["descriptor name"], str)
        assert isinstance(desc["explanation"], str)
    
    # Check for some common descriptors
    descriptor_names = [d["descriptor name"] for d in descriptors]
    assert "MolWt" in descriptor_names
    assert "TPSA" in descriptor_names
    assert "MolLogP" in descriptor_names


def test_calculate_simple_descriptors_basic(session_workdir, request):
    """Test basic descriptor calculation."""
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
    
    # Calculate some common descriptors
    result = calculate_simple_descriptors(
        input_filename=input_file,
        smiles_column="smiles",
        descriptor_names=["MolWt", "TPSA", "NumHDonors"],
        project_manifest_path=manifest_path,
        output_filename="with_descriptors",
        explanation="calculated descriptors"
    )
    
    # Check return structure
    assert "output_filename" in result
    assert "n_rows" in result
    assert "columns" in result
    assert "descriptors_added" in result
    assert "n_failed" in result
    
    # Check values
    assert result["n_rows"] == 3
    assert "MolWt" in result["columns"]
    assert "TPSA" in result["columns"]
    assert "NumHDonors" in result["columns"]
    assert result["descriptors_added"] == ["MolWt", "TPSA", "NumHDonors"]
    
    # Check preview has reasonable values
    preview = result["preview"]
    assert len(preview) == 3
    # Ethanol (CCO) should have MolWt around 46, TPSA around 20, 1 H-donor
    assert 45 < preview[0]["MolWt"] < 47
    assert preview[0]["NumHDonors"] == 1


def test_calculate_simple_descriptors_invalid_smiles(session_workdir, request):
    """Test descriptor calculation handles invalid SMILES."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create dataset with invalid SMILES
    df = pd.DataFrame({
        "smiles": ["CCO", "INVALID", None, "c1ccccc1"],
        "id": [1, 2, 3, 4]
    })
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Calculate descriptors
    result = calculate_simple_descriptors(
        input_filename=input_file,
        smiles_column="smiles",
        descriptor_names=["MolWt"],
        project_manifest_path=manifest_path,
        output_filename="with_descriptors",
        explanation="calculated descriptors"
    )
    
    # Should complete but report failures
    assert result["n_rows"] == 4
    assert result["n_failed"]["MolWt"] >= 2  # At least 2 failed (INVALID and None)
    
    # Preview should have None or NaN for invalid entries
    import math
    preview = result["preview"]
    assert preview[0]["MolWt"] is not None and not (isinstance(preview[0]["MolWt"], float) and math.isnan(preview[0]["MolWt"]))  # Valid SMILES
    # Invalid SMILES should have None or NaN
    assert preview[1]["MolWt"] is None or (isinstance(preview[1]["MolWt"], float) and math.isnan(preview[1]["MolWt"]))
    assert preview[2]["MolWt"] is None or (isinstance(preview[2]["MolWt"], float) and math.isnan(preview[2]["MolWt"]))


def test_calculate_simple_descriptors_invalid_descriptor(session_workdir, request):
    """Test error handling for invalid descriptor names."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    df = pd.DataFrame({"smiles": ["CCO"], "id": [1]})
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Try to use invalid descriptor name
    with pytest.raises(ValueError, match="Invalid descriptor names"):
        calculate_simple_descriptors(
            input_filename=input_file,
            smiles_column="smiles",
            descriptor_names=["FakeDescriptor"],
            project_manifest_path=manifest_path,
            output_filename="output",
            explanation="test"
        )


def test_calculate_simple_descriptors_multiple_descriptors(session_workdir, request):
    """Test calculating multiple descriptors at once."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create test dataset with drug-like molecules
    df = pd.DataFrame({
        "smiles": ["CCO", "CC(=O)O", "c1ccccc1"],
        "name": ["ethanol", "acetic acid", "benzene"]
    })
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Calculate multiple descriptors
    descriptors = ["MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors"]
    result = calculate_simple_descriptors(
        input_filename=input_file,
        smiles_column="smiles",
        descriptor_names=descriptors,
        project_manifest_path=manifest_path,
        output_filename="with_many_descriptors",
        explanation="multiple descriptors"
    )
    
    # All descriptors should be added
    assert result["n_rows"] == 3
    assert len(result["descriptors_added"]) == 5
    for desc in descriptors:
        assert desc in result["columns"]
        assert desc in result["descriptors_added"]
    
    # Check that all molecules succeeded
    for desc in descriptors:
        assert result["n_failed"][desc] == 0
