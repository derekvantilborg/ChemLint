"""
Statistical tests for dataset analysis.

Normality Tests:
- Shapiro-Wilk test: checks if data is normally distributed (best for small to medium samples)
- Kolmogorov-Smirnov test: alternative normality test (good for larger samples)
- Anderson-Darling test: another normality test (more sensitive to tails)

Paired Comparison Tests:
- Paired t-test: compares means of two related samples (assumes normality)
- Wilcoxon signed-rank test: non-parametric alternative to paired t-test

Correlation Tests:
- Pearson correlation: measures linear correlation (assumes normality)
- Spearman correlation: measures monotonic correlation (rank-based, non-parametric)
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


def test_paired_ttest(
    input_filename_a: str,
    input_filename_b: str,
    project_manifest_path: str,
    column_a: str,
    column_b: str,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> Dict:
    """
    Perform paired t-test comparing two related samples.
    
    The paired t-test checks if the mean difference between paired observations
    is significantly different from zero. Assumes that the differences follow a
    normal distribution. Use this when comparing before/after measurements or
    matched pairs.
    
    Null hypothesis (H0): The mean difference between pairs is zero.
    - p-value > alpha: Fail to reject H0 (no significant difference)
    - p-value <= alpha: Reject H0 (significant difference exists)
    
    Args:
        input_filename_a: First CSV dataset resource filename
        input_filename_b: Second CSV dataset resource filename
        project_manifest_path: Path to project manifest.json
        column_a: Column name in dataset A
        column_b: Column name in dataset B
        alpha: Significance level (default: 0.05)
        alternative: Type of test - "two-sided", "less", or "greater" (default: "two-sided")
        
    Returns:
        Dictionary containing:
            - statistic: t-statistic from the test
            - p_value: p-value from the test
            - alpha: Significance level used
            - alternative: Type of test performed
            - is_significant: Boolean indicating if difference is significant
            - n_pairs: Number of paired samples
            - mean_diff: Mean of differences (A - B)
            - std_diff: Standard deviation of differences
            - interpretation: Human-readable interpretation
            
    Example:
        >>> result = test_paired_ttest(
        ...     "before_treatment.csv",
        ...     "after_treatment.csv",
        ...     "manifest.json",
        ...     "score",
        ...     "score",
        ...     alternative="greater"
        ... )
    """
    # Validate alternative
    valid_alternatives = ["two-sided", "less", "greater"]
    if alternative not in valid_alternatives:
        raise ValueError(
            f"alternative must be one of {valid_alternatives}. "
            f"Got: {alternative}"
        )
    
    # Load datasets
    df_a = _load_resource(project_manifest_path, input_filename_a)
    df_b = _load_resource(project_manifest_path, input_filename_b)
    
    # Validate columns
    if column_a not in df_a.columns:
        raise ValueError(
            f"Column '{column_a}' not found in dataset A. "
            f"Available: {list(df_a.columns)}"
        )
    if column_b not in df_b.columns:
        raise ValueError(
            f"Column '{column_b}' not found in dataset B. "
            f"Available: {list(df_b.columns)}"
        )
    
    # Get data
    data_a = df_a[column_a].values
    data_b = df_b[column_b].values
    
    # Check equal lengths
    if len(data_a) != len(data_b):
        raise ValueError(
            f"Datasets must have equal length for paired test. "
            f"Dataset A: {len(data_a)}, Dataset B: {len(data_b)}"
        )
    
    # Remove pairs with NaN in either column
    valid_mask = ~(pd.isna(data_a) | pd.isna(data_b))
    data_a_clean = data_a[valid_mask]
    data_b_clean = data_b[valid_mask]
    n_pairs = len(data_a_clean)
    
    if n_pairs == 0:
        raise ValueError("No valid paired samples (all contain NaN)")
    
    if n_pairs < 2:
        raise ValueError(f"Paired t-test requires at least 2 pairs. Found: {n_pairs}")
    
    # Calculate differences
    differences = data_a_clean - data_b_clean
    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))
    
    # Perform paired t-test
    statistic, p_value = stats.ttest_rel(data_a_clean, data_b_clean, alternative=alternative)
    
    # Interpret result
    is_significant = p_value <= alpha
    
    if alternative == "two-sided":
        if is_significant:
            direction = "greater" if mean_diff > 0 else "less"
            interpretation = (
                f"Significant difference detected (p={p_value:.4f} ≤ α={alpha}). "
                f"Mean difference: {mean_diff:.4f}. Dataset A is {direction} than Dataset B."
            )
        else:
            interpretation = (
                f"No significant difference (p={p_value:.4f} > α={alpha}). "
                f"Mean difference: {mean_diff:.4f}."
            )
    elif alternative == "greater":
        if is_significant:
            interpretation = (
                f"Dataset A is significantly GREATER than Dataset B (p={p_value:.4f} ≤ α={alpha}). "
                f"Mean difference: {mean_diff:.4f}."
            )
        else:
            interpretation = (
                f"Dataset A is NOT significantly greater than Dataset B (p={p_value:.4f} > α={alpha}). "
                f"Mean difference: {mean_diff:.4f}."
            )
    else:  # alternative == "less"
        if is_significant:
            interpretation = (
                f"Dataset A is significantly LESS than Dataset B (p={p_value:.4f} ≤ α={alpha}). "
                f"Mean difference: {mean_diff:.4f}."
            )
        else:
            interpretation = (
                f"Dataset A is NOT significantly less than Dataset B (p={p_value:.4f} > α={alpha}). "
                f"Mean difference: {mean_diff:.4f}."
            )
    
    return {
        "test": "Paired t-test",
        "dataset_a": input_filename_a,
        "dataset_b": input_filename_b,
        "column_a": column_a,
        "column_b": column_b,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "alternative": alternative,
        "is_significant": is_significant,
        "n_pairs": n_pairs,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "interpretation": interpretation,
        "summary": f"Paired t-test: t={statistic:.4f}, p={p_value:.4f}, significant={is_significant}"
    }


def test_wilcoxon_signed_rank(
    input_filename_a: str,
    input_filename_b: str,
    project_manifest_path: str,
    column_a: str,
    column_b: str,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> Dict:
    """
    Perform Wilcoxon signed-rank test comparing two related samples.
    
    The Wilcoxon signed-rank test is a non-parametric alternative to the paired
    t-test. It tests whether the median difference between pairs is zero, without
    assuming normality. Use this when data is not normally distributed or when
    dealing with ordinal data.
    
    Null hypothesis (H0): The median difference between pairs is zero.
    - p-value > alpha: Fail to reject H0 (no significant difference)
    - p-value <= alpha: Reject H0 (significant difference exists)
    
    Args:
        input_filename_a: First CSV dataset resource filename
        input_filename_b: Second CSV dataset resource filename
        project_manifest_path: Path to project manifest.json
        column_a: Column name in dataset A
        column_b: Column name in dataset B
        alpha: Significance level (default: 0.05)
        alternative: Type of test - "two-sided", "less", or "greater" (default: "two-sided")
        
    Returns:
        Dictionary containing:
            - statistic: W statistic (sum of positive ranks)
            - p_value: p-value from the test
            - alpha: Significance level used
            - alternative: Type of test performed
            - is_significant: Boolean indicating if difference is significant
            - n_pairs: Number of paired samples
            - median_diff: Median of differences (A - B)
            - n_positive: Number of positive differences
            - n_negative: Number of negative differences
            - n_zero: Number of zero differences (excluded from test)
            - interpretation: Human-readable interpretation
            
    Example:
        >>> result = test_wilcoxon_signed_rank(
        ...     "before.csv",
        ...     "after.csv",
        ...     "manifest.json",
        ...     "rank",
        ...     "rank"
        ... )
    """
    # Validate alternative
    valid_alternatives = ["two-sided", "less", "greater"]
    if alternative not in valid_alternatives:
        raise ValueError(
            f"alternative must be one of {valid_alternatives}. "
            f"Got: {alternative}"
        )
    
    # Load datasets
    df_a = _load_resource(project_manifest_path, input_filename_a)
    df_b = _load_resource(project_manifest_path, input_filename_b)
    
    # Validate columns
    if column_a not in df_a.columns:
        raise ValueError(
            f"Column '{column_a}' not found in dataset A. "
            f"Available: {list(df_a.columns)}"
        )
    if column_b not in df_b.columns:
        raise ValueError(
            f"Column '{column_b}' not found in dataset B. "
            f"Available: {list(df_b.columns)}"
        )
    
    # Get data
    data_a = df_a[column_a].values
    data_b = df_b[column_b].values
    
    # Check equal lengths
    if len(data_a) != len(data_b):
        raise ValueError(
            f"Datasets must have equal length for paired test. "
            f"Dataset A: {len(data_a)}, Dataset B: {len(data_b)}"
        )
    
    # Remove pairs with NaN in either column
    valid_mask = ~(pd.isna(data_a) | pd.isna(data_b))
    data_a_clean = data_a[valid_mask]
    data_b_clean = data_b[valid_mask]
    n_pairs = len(data_a_clean)
    
    if n_pairs == 0:
        raise ValueError("No valid paired samples (all contain NaN)")
    
    # Calculate differences
    differences = data_a_clean - data_b_clean
    median_diff = float(np.median(differences))
    n_positive = int(np.sum(differences > 0))
    n_negative = int(np.sum(differences < 0))
    n_zero = int(np.sum(differences == 0))
    
    # Perform Wilcoxon signed-rank test
    result = stats.wilcoxon(data_a_clean, data_b_clean, alternative=alternative)
    statistic = result.statistic
    p_value = result.pvalue
    
    # Interpret result
    is_significant = p_value <= alpha
    
    if alternative == "two-sided":
        if is_significant:
            direction = "greater" if median_diff > 0 else "less"
            interpretation = (
                f"Significant difference detected (p={p_value:.4f} ≤ α={alpha}). "
                f"Median difference: {median_diff:.4f}. Dataset A is {direction} than Dataset B."
            )
        else:
            interpretation = (
                f"No significant difference (p={p_value:.4f} > α={alpha}). "
                f"Median difference: {median_diff:.4f}."
            )
    elif alternative == "greater":
        if is_significant:
            interpretation = (
                f"Dataset A is significantly GREATER than Dataset B (p={p_value:.4f} ≤ α={alpha}). "
                f"Median difference: {median_diff:.4f}."
            )
        else:
            interpretation = (
                f"Dataset A is NOT significantly greater than Dataset B (p={p_value:.4f} > α={alpha}). "
                f"Median difference: {median_diff:.4f}."
            )
    else:  # alternative == "less"
        if is_significant:
            interpretation = (
                f"Dataset A is significantly LESS than Dataset B (p={p_value:.4f} ≤ α={alpha}). "
                f"Median difference: {median_diff:.4f}."
            )
        else:
            interpretation = (
                f"Dataset A is NOT significantly less than Dataset B (p={p_value:.4f} > α={alpha}). "
                f"Median difference: {median_diff:.4f}."
            )
    
    return {
        "test": "Wilcoxon signed-rank",
        "dataset_a": input_filename_a,
        "dataset_b": input_filename_b,
        "column_a": column_a,
        "column_b": column_b,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "alternative": alternative,
        "is_significant": is_significant,
        "n_pairs": n_pairs,
        "median_diff": median_diff,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
        "interpretation": interpretation,
        "summary": f"Wilcoxon test: W={statistic:.4f}, p={p_value:.4f}, significant={is_significant}"
    }


def test_pearson_correlation(
    input_filename_a: str,
    input_filename_b: str,
    project_manifest_path: str,
    column_a: str,
    column_b: str,
    alpha: float = 0.05
) -> Dict:
    """
    Calculate Pearson correlation coefficient between two variables.
    
    Pearson correlation measures the linear relationship between two continuous
    variables. It assumes that both variables are normally distributed and tests
    whether the correlation is significantly different from zero.
    
    Correlation coefficient (r) ranges from -1 to 1:
    - r = 1: Perfect positive linear correlation
    - r = 0: No linear correlation
    - r = -1: Perfect negative linear correlation
    
    Null hypothesis (H0): The correlation is zero (no linear relationship).
    - p-value > alpha: Fail to reject H0 (no significant correlation)
    - p-value <= alpha: Reject H0 (significant correlation exists)
    
    Args:
        input_filename_a: First CSV dataset resource filename
        input_filename_b: Second CSV dataset resource filename
        project_manifest_path: Path to project manifest.json
        column_a: Column name in dataset A
        column_b: Column name in dataset B
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary containing:
            - correlation: Pearson correlation coefficient (r)
            - p_value: p-value testing if correlation is significantly different from 0
            - alpha: Significance level used
            - is_significant: Boolean indicating if correlation is significant
            - n_samples: Number of paired samples
            - interpretation: Human-readable interpretation
            - strength: Qualitative strength assessment
            
    Example:
        >>> result = test_pearson_correlation(
        ...     "dataset_x.csv",
        ...     "dataset_y.csv",
        ...     "manifest.json",
        ...     "variable_x",
        ...     "variable_y"
        ... )
    """
    # Load datasets
    df_a = _load_resource(project_manifest_path, input_filename_a)
    df_b = _load_resource(project_manifest_path, input_filename_b)
    
    # Validate columns
    if column_a not in df_a.columns:
        raise ValueError(
            f"Column '{column_a}' not found in dataset A. "
            f"Available: {list(df_a.columns)}"
        )
    if column_b not in df_b.columns:
        raise ValueError(
            f"Column '{column_b}' not found in dataset B. "
            f"Available: {list(df_b.columns)}"
        )
    
    # Get data
    data_a = df_a[column_a].values
    data_b = df_b[column_b].values
    
    # Check equal lengths
    if len(data_a) != len(data_b):
        raise ValueError(
            f"Datasets must have equal length. "
            f"Dataset A: {len(data_a)}, Dataset B: {len(data_b)}"
        )
    
    # Remove pairs with NaN in either column
    valid_mask = ~(pd.isna(data_a) | pd.isna(data_b))
    data_a_clean = data_a[valid_mask]
    data_b_clean = data_b[valid_mask]
    n_samples = len(data_a_clean)
    
    if n_samples == 0:
        raise ValueError("No valid paired samples (all contain NaN)")
    
    if n_samples < 3:
        raise ValueError(f"Correlation requires at least 3 samples. Found: {n_samples}")
    
    # Calculate Pearson correlation
    correlation, p_value = stats.pearsonr(data_a_clean, data_b_clean)
    
    # Interpret correlation strength (using common thresholds)
    abs_corr = abs(correlation)
    if abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    direction = "positive" if correlation > 0 else "negative"
    
    # Interpret significance
    is_significant = p_value <= alpha
    
    if is_significant:
        interpretation = (
            f"Significant {strength} {direction} correlation detected "
            f"(r={correlation:.4f}, p={p_value:.4f} ≤ α={alpha}). "
            f"Variables show a linear relationship."
        )
    else:
        interpretation = (
            f"No significant correlation (r={correlation:.4f}, p={p_value:.4f} > α={alpha}). "
            f"Variables do not show a significant linear relationship."
        )
    
    return {
        "test": "Pearson correlation",
        "dataset_a": input_filename_a,
        "dataset_b": input_filename_b,
        "column_a": column_a,
        "column_b": column_b,
        "correlation": float(correlation),
        "p_value": float(p_value),
        "alpha": alpha,
        "is_significant": is_significant,
        "n_samples": n_samples,
        "strength": strength,
        "direction": direction,
        "interpretation": interpretation,
        "summary": f"Pearson r={correlation:.4f}, p={p_value:.4f}, {strength} {direction}, significant={is_significant}"
    }


def test_spearman_correlation(
    input_filename_a: str,
    input_filename_b: str,
    project_manifest_path: str,
    column_a: str,
    column_b: str,
    alpha: float = 0.05
) -> Dict:
    """
    Calculate Spearman rank correlation coefficient between two variables.
    
    Spearman correlation measures monotonic relationships (whether variables
    tend to change together, not necessarily linearly). It's a non-parametric
    measure based on ranks, so it doesn't assume normality and is robust to
    outliers.
    
    Correlation coefficient (ρ or rho) ranges from -1 to 1:
    - ρ = 1: Perfect monotonic increasing relationship
    - ρ = 0: No monotonic relationship
    - ρ = -1: Perfect monotonic decreasing relationship
    
    Null hypothesis (H0): The correlation is zero (no monotonic relationship).
    - p-value > alpha: Fail to reject H0 (no significant correlation)
    - p-value <= alpha: Reject H0 (significant correlation exists)
    
    Args:
        input_filename_a: First CSV dataset resource filename
        input_filename_b: Second CSV dataset resource filename
        project_manifest_path: Path to project manifest.json
        column_a: Column name in dataset A
        column_b: Column name in dataset B
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary containing:
            - correlation: Spearman correlation coefficient (ρ)
            - p_value: p-value testing if correlation is significantly different from 0
            - alpha: Significance level used
            - is_significant: Boolean indicating if correlation is significant
            - n_samples: Number of paired samples
            - interpretation: Human-readable interpretation
            - strength: Qualitative strength assessment
            
    Example:
        >>> result = test_spearman_correlation(
        ...     "dataset_x.csv",
        ...     "dataset_y.csv",
        ...     "manifest.json",
        ...     "rank_x",
        ...     "rank_y"
        ... )
    """
    # Load datasets
    df_a = _load_resource(project_manifest_path, input_filename_a)
    df_b = _load_resource(project_manifest_path, input_filename_b)
    
    # Validate columns
    if column_a not in df_a.columns:
        raise ValueError(
            f"Column '{column_a}' not found in dataset A. "
            f"Available: {list(df_a.columns)}"
        )
    if column_b not in df_b.columns:
        raise ValueError(
            f"Column '{column_b}' not found in dataset B. "
            f"Available: {list(df_b.columns)}"
        )
    
    # Get data
    data_a = df_a[column_a].values
    data_b = df_b[column_b].values
    
    # Check equal lengths
    if len(data_a) != len(data_b):
        raise ValueError(
            f"Datasets must have equal length. "
            f"Dataset A: {len(data_a)}, Dataset B: {len(data_b)}"
        )
    
    # Remove pairs with NaN in either column
    valid_mask = ~(pd.isna(data_a) | pd.isna(data_b))
    data_a_clean = data_a[valid_mask]
    data_b_clean = data_b[valid_mask]
    n_samples = len(data_a_clean)
    
    if n_samples == 0:
        raise ValueError("No valid paired samples (all contain NaN)")
    
    if n_samples < 3:
        raise ValueError(f"Correlation requires at least 3 samples. Found: {n_samples}")
    
    # Calculate Spearman correlation
    correlation, p_value = stats.spearmanr(data_a_clean, data_b_clean)
    
    # Interpret correlation strength (using common thresholds)
    abs_corr = abs(correlation)
    if abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    direction = "positive" if correlation > 0 else "negative"
    
    # Interpret significance
    is_significant = p_value <= alpha
    
    if is_significant:
        interpretation = (
            f"Significant {strength} {direction} correlation detected "
            f"(ρ={correlation:.4f}, p={p_value:.4f} ≤ α={alpha}). "
            f"Variables show a monotonic relationship."
        )
    else:
        interpretation = (
            f"No significant correlation (ρ={correlation:.4f}, p={p_value:.4f} > α={alpha}). "
            f"Variables do not show a significant monotonic relationship."
        )
    
    return {
        "test": "Spearman correlation",
        "dataset_a": input_filename_a,
        "dataset_b": input_filename_b,
        "column_a": column_a,
        "column_b": column_b,
        "correlation": float(correlation),
        "p_value": float(p_value),
        "alpha": alpha,
        "is_significant": is_significant,
        "n_samples": n_samples,
        "strength": strength,
        "direction": direction,
        "interpretation": interpretation,
        "summary": f"Spearman ρ={correlation:.4f}, p={p_value:.4f}, {strength} {direction}, significant={is_significant}"
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


def get_all_paired_test_tools():
    """
    Returns a list of MCP-exposed paired comparison test functions for server registration.
    """
    return [
        test_paired_ttest,
        test_wilcoxon_signed_rank,
    ]


def get_all_correlation_test_tools():
    """
    Returns a list of MCP-exposed correlation test functions for server registration.
    """
    return [
        test_pearson_correlation,
        test_spearman_correlation,
    ]