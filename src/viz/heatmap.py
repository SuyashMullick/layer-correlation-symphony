from __future__ import annotations
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def save_correlation_heatmap(
    df: pd.DataFrame,
    out_path: Path,
    title: str = "Correlation Heatmap"
):
    """
    Calculates the correlation matrix of a DataFrame and saves it as a heatmap image.
    """
    if df.shape[1] < 2:
        # No point in creating a heatmap for a single variable
        return

    # Calculate Pearson correlation
    corr_matrix = df.corr(method="pearson")

    # Create the plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr_matrix,
        annot=True,      # Show the correlation values on the heatmap
        cmap="coolwarm", # Use a clear blue-to-red colormap
        fmt=".2f",       # Format values to two decimal places
        linewidths=.5
    )
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Ensure the output directory exists and save the figure
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close() # Close the plot to free up memory
    print(f"[viz] Saved correlation heatmap to: {out_path}")
