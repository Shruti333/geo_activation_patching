import plotly.express as px
import pandas as pd
import numpy as np

def plot_patching_heatmap(
    results_matrix: np.ndarray, 
    token_labels: list, 
    title: str = "Activation Patching Results"
):
    """
    Renders a standard publication-ready Plotly heatmap for mechanistic interpretability.
    """
    fig = px.imshow(
        results_matrix,
        x=token_labels,
        labels={"x": "Sequence Tokens", "y": "Transformer Layer"},
        title=title,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0
    )
    
    fig.update_layout(
        xaxis_title="Tokens (Context Window Position)",
        yaxis_title="Layer Index",
        coloraxis_colorbar=dict(title="Normalized Recovery")
    )
    return fig