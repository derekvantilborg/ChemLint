"""
Statistical tests for dataset analysis.

Normality Tests:
- Shapiro-Wilk test: checks if data is normally distributed (best for small to medium samples)
- Kolmogorov-Smirnov test: alternative normality test (good for larger samples)
- Anderson-Darling test: another normality test (more sensitive to tails)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from scipy import stats
from molml_mcp.infrastructure.resources import _load_resource


def test_shapiro_wilk(
    input_filename: str,
    project_manifest_path: str,
    column: str,
    alpha: float = 0.05
) -> Dict:
    """
    Perform Shapiro-Wilk test for normality on a dataset column.
    
    The Shapiro-Wilk test checks if data is normally distributed. It's most
    appropriate for small to medium-sized samples (n < 5000).
    
    Null hypothesis (H0): The data is normally distributed.
    - p-value > alpha: Fail to reject H0 (data appears normally distributed)
    - p-value <= alpha: Reject H0 (data does not appear normally distributed)
    
    Args:
        input_filename: CSV dataset resource filename
        project_manifest_path: Path to project manifest.json
        column: Column name to test
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary containing:
            - statistic: W statistic from the test
            - p_value: p-value from the test
            - alpha: Significance level used
            - is_normal: Boolean indicating if data appears normally distributed
            - n_samples: Number of samples tested
            - interpretation: Human-readable interpretation
            
    Example:
        >>> result = test_shapiro_wilk(
        ...     "dataset.csv",
        ...     "manifest.json",
        ...     "molecular_weight"
        ... )
        >>> if result['is_normal']:
        ...     print("Data is normally distributed")
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate column
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. "
            f"Available: {list(df.columns)}"
        )
    
    # Get data and remove NaN values
    data = df[column].dropna()
    n_samples = len(data)
    
    if n_samples == 0:
        raise ValueError(f"No valid (non-NaN) data in column '{column}'")
    
    if n_samples < 3:
        raise ValueError(
            f"Shapiro-Wilk test requires at least 3 samples. "
            f"Found: {n_samples}"
        )
    
    # Perform Shapiro-Wilk test
    statistic, p_value = stats.shapiro(data)
    
    # Interpret result
    is_normal = p_value > alpha
    
    if is_normal:
        interpretation = (
            f"Data appears normally distributed (p={p_value:.4f} > α={alpha}). "
            f"Fail to reject null hypothesis."
        )
    else:
        interpretation = (
            f"Data does NOT appear normally distributed (p={p_value:.4f} ≤ α={alpha}). "
            f"Reject null hypothesis."
        )
    
    return {
        "test": "Shapiro-Wilk",
        "column": column,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "is_normal": is_normal,
        "n_samples": n_samples,
        "interpretation": interpretation,
        "summary": f"Shapiro-Wilk test: W={statistic:.4f}, p={p_value:.4f}, normal={is_normal}"
    }


def test_kolmogorov_smirnov_norm(
    input_filename: str,
    project_manifest_path: str,
    column: str,
    alpha: float = 0.05
) -> Dict:
    """
    Perform Kolmogorov-Smirnov test for normality on a dataset column.
    
    The K-S test compares the empirical distribution with a normal distribution.
    It's suitable for larger samples and is less sensitive than Shapiro-Wilk.
    
    Null hypothesis (H0): The data follows a normal distribution.
    - p-value > alpha: Fail to reject H0 (data appears normally distributed)
    - p-value <= alpha: Reject H0 (data does not appear normally distributed)
    
    Args:
        input_filename: CSV dataset resource filename
        project_manifest_path: Path to project manifest.json
        column: Column name to test
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary containing:
            - statistic: K-S statistic (maximum distance between distributions)
            - p_value: p-value from the test
            - alpha: Significance level used
            - is_normal: Boolean indicating if data appears normally distributed
            - n_samples: Number of samples tested
            - mean: Sample mean
            - std: Sample standard deviation
            - interpretation: Human-readable interpretation
            
    Example:
        >>> result = test_kolmogorov_smirnov(
        ...     "dataset.csv",
        ...     "manifest.json",
        ...     "logP",
        ...     alpha=0.01
        ... )
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate column
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. "
            f"Available: {list(df.columns)}"
        )
    
    # Get data and remove NaN values
    data = df[column].dropna()
    n_samples = len(data)
    
    if n_samples == 0:
        raise ValueError(f"No valid (non-NaN) data in column '{column}'")
    
    # Calculate mean and std for the normal distribution
    mean = data.mean()
    std = data.std()
    
    if std == 0:
        raise ValueError(f"Standard deviation is zero for column '{column}'")
    
    # Perform K-S test comparing data to normal distribution
    statistic, p_value = stats.kstest(data, 'norm', args=(mean, std))
    
    # Interpret result
    is_normal = p_value > alpha
    
    if is_normal:
        interpretation = (
            f"Data appears normally distributed (p={p_value:.4f} > α={alpha}). "
            f"Fail to reject null hypothesis."
        )
    else:
        interpretation = (
            f"Data does NOT appear normally distributed (p={p_value:.4f} ≤ α={alpha}). "
            f"Reject null hypothesis."
        )
    
    return {
        "test": "Kolmogorov-Smirnov",
        "column": column,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "is_normal": is_normal,
        "n_samples": n_samples,
        "mean": float(mean),
        "std": float(std),
        "interpretation": interpretation,
        "summary": f"K-S test: D={statistic:.4f}, p={p_value:.4f}, normal={is_normal}"
    }


def test_anderson_darling(
    input_filename: str,
    project_manifest_path: str,
    column: str,
    significance_level: str = "5%"
) -> Dict:
    """
    Perform Anderson-Darling test for normality on a dataset column.
    
    The Anderson-Darling test is more sensitive to deviations in the tails of
    the distribution compared to K-S test. It provides critical values at
    different significance levels (15%, 10%, 5%, 2.5%, 1%).
    
    Null hypothesis (H0): The data follows a normal distribution.
    - statistic < critical_value: Fail to reject H0 (data appears normally distributed)
    - statistic >= critical_value: Reject H0 (data does not appear normally distributed)
    
    Args:
        input_filename: CSV dataset resource filename
        project_manifest_path: Path to project manifest.json
        column: Column name to test
        significance_level: One of "15%", "10%", "5%", "2.5%", "1%" (default: "5%")
        
    Returns:
        Dictionary containing:
            - statistic: Anderson-Darling statistic
            - critical_values: Critical values at different significance levels
            - significance_levels: Corresponding significance levels (as percentages)
            - selected_alpha: The selected significance level
            - critical_value: Critical value at selected significance level
            - is_normal: Boolean indicating if data appears normally distributed
            - n_samples: Number of samples tested
            - interpretation: Human-readable interpretation
            
    Example:
        >>> result = test_anderson_darling(
        ...     "dataset.csv",
        ...     "manifest.json",
        ...     "pIC50",
        ...     significance_level="5%"
        ... )
    """
    # Validate significance level
    valid_levels = ["15%", "10%", "5%", "2.5%", "1%"]
    if significance_level not in valid_levels:
        raise ValueError(
            f"significance_level must be one of {valid_levels}. "
            f"Got: {significance_level}"
        )
    
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate column
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. "
            f"Available: {list(df.columns)}"
        )
    
    # Get data and remove NaN values
    data = df[column].dropna()
    n_samples = len(data)
    
    if n_samples == 0:
        raise ValueError(f"No valid (non-NaN) data in column '{column}'")
    
    # Perform Anderson-Darling test
    result = stats.anderson(data, dist='norm')
    
    statistic = result.statistic
    critical_values = result.critical_values
    significance_levels = result.significance_level
    
    # Map significance level to index
    level_map = {"15%": 0, "10%": 1, "5%": 2, "2.5%": 3, "1%": 4}
    idx = level_map[significance_level]
    
    critical_value = critical_values[idx]
    is_normal = statistic < critical_value
    
    # Create critical values dict
    critical_values_dict = {
        f"{int(sig)}%": float(cv) 
        for sig, cv in zip(significance_levels, critical_values)
    }
    
    if is_normal:
        interpretation = (
            f"Data appears normally distributed "
            f"(statistic={statistic:.4f} < critical_value={critical_value:.4f} at α={significance_level}). "
            f"Fail to reject null hypothesis."
        )
    else:
        interpretation = (
            f"Data does NOT appear normally distributed "
            f"(statistic={statistic:.4f} ≥ critical_value={critical_value:.4f} at α={significance_level}). "
            f"Reject null hypothesis."
        )
    
    return {
        "test": "Anderson-Darling",
        "column": column,
        "statistic": float(statistic),
        "critical_values": critical_values_dict,
        "significance_levels": [f"{int(s)}%" for s in significance_levels],
        "selected_alpha": significance_level,
        "critical_value": float(critical_value),
        "is_normal": is_normal,
        "n_samples": n_samples,
        "interpretation": interpretation,
        "summary": f"Anderson-Darling test: A²={statistic:.4f}, critical={critical_value:.4f}, normal={is_normal}"
    }


def get_all_normality_test_tools():
    """
    Returns a list of MCP-exposed normality test functions for server registration.
    """
    return [
        test_shapiro_wilk,
        test_kolmogorov_smirnov_norm,
        test_anderson_darling,
    ]