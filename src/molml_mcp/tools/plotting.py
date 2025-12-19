"""
Plotting functions for molecular data visualization.
"""

import pandas as pd
import io
from plotnine import (
    ggplot, aes, geom_histogram, theme_minimal, theme, element_text, 
    element_line, element_rect, labs, scale_fill_manual
)
from mcp.server.fastmcp import Image
from molml_mcp.infrastructure.resources import _load_resource


def plot_histogram(
    input_filename: str,
    column: str,
    project_manifest_path: str,
    bins: int = 30,
    fill_color: str = "#577788",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Count",
    width: float = 6.0,
    height: float = 4.0,
    dpi: int = 300
) -> list:
    """
    Create a publication-quality histogram using plotnine (Nature paper style).
    
    Generates a clean, professional histogram with minimalist styling suitable
    for scientific publications. Returns the image for inline display.
    
    Parameters
    ----------
    input_filename : str
        Base filename of the input dataset resource.
    column : str
        Name of the column to plot.
    project_manifest_path : str
        Path to the project manifest JSON file.
    bins : int
        Number of histogram bins (default: 30).
    fill_color : str
        Hex color for histogram bars (default: "#577788" - blue-gray).
    title : str | None
        Plot title (default: None for no title).
    xlabel : str | None
        X-axis label (default: column name).
    ylabel : str
        Y-axis label (default: "Count").
    width : float
        Figure width in inches (default: 6.0).
    height : float
        Figure height in inches (default: 4.0).
    dpi : int
        Resolution in dots per inch (default: 300).
    
    Returns
    -------
    list
        [Image, str] - FastMCP Image object and summary statistics string
    
    Examples
    --------
    Basic histogram:
    
        img, stats = plot_histogram(
            input_filename='dataset_AB12CD34.csv',
            column='molecular_weight',
            project_manifest_path='/path/to/manifest.json'
        )
    
    Customized histogram:
    
        img, stats = plot_histogram(
            input_filename='dataset_AB12CD34.csv',
            column='logP',
            project_manifest_path='/path/to/manifest.json',
            bins=40,
            fill_color='#3498DB',
            title='LogP Distribution',
            xlabel='LogP',
            ylabel='Frequency'
        )
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate column exists
    if column not in df.columns:
        available_columns = df.columns.tolist()
        raise ValueError(
            f"Column '{column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    # Extract data and remove NaN values
    data = df[column].dropna()
    
    if len(data) == 0:
        raise ValueError(f"Column '{column}' contains no valid (non-NaN) values")
    
    # Calculate statistics
    stats = {
        "n_values": len(data),
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "median": float(data.median())
    }
    
    # Create DataFrame for plotnine
    plot_df = pd.DataFrame({column: data})
    
    # Set default labels
    if xlabel is None:
        xlabel = column
    
    # Create the plot with Nature paper styling
    p = (
        ggplot(plot_df, aes(x=column))
        + geom_histogram(bins=bins, fill=fill_color, color='white', size=0.3, alpha=0.9)
        + labs(
            title=title if title else '',
            x=xlabel,
            y=ylabel
        )
        + theme_minimal()
        + theme(
            # Text elements - clean and legible
            text=element_text(family='Arial', size=11, color='#2C3E50'),
            plot_title=element_text(size=13, face='bold', margin={'b': 15}) if title else element_text(size=0),
            axis_title_x=element_text(size=11, face='bold', margin={'t': 10}),
            axis_title_y=element_text(size=11, face='bold', margin={'r': 10}),
            axis_text=element_text(size=9, color='#34495E'),
            
            # Grid - subtle and minimal
            panel_grid_major=element_line(color='#ECF0F1', size=0.5),
            panel_grid_minor=element_line(color='#ECF0F1', size=0.25),
            
            # Background - clean white
            panel_background=element_rect(fill='white'),
            plot_background=element_rect(fill='white'),
            
            # Axes - subtle lines
            axis_line=element_line(color='#95A5A6', size=0.5),
            
            # Remove top and right spines for cleaner look
            panel_border=element_rect(color='none'),
            
            # Adjust plot margins
            plot_margin=0.05
        )
    )
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    p.save(buf, format='png', width=width, height=height, dpi=dpi, verbose=False)
    buf.seek(0)
    png_bytes = buf.read()
    buf.close()
    
    # Create FastMCP Image object
    img = Image(data=png_bytes, format="png")
    
    # Create summary statistics string
    summary = (
        f"Histogram of '{column}': {stats['n_values']} values, "
        f"range [{stats['min']:.2f}, {stats['max']:.2f}], "
        f"mean={stats['mean']:.2f}, median={stats['median']:.2f}"
    )
    
    return [img, summary]
