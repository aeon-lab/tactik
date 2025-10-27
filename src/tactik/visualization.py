import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Union, List, Dict, Tuple

# Default style settings
DEFAULT_STYLE = {
    'figure.figsize': (13, 9),
    'axes.facecolor': 'white',
    'axes.grid': False
}

def set_visualization_style(style_params: Optional[dict] = None) -> None:
    """
    Set consistent visualization style for all plots using matplotlib rcParams.
    
    This function configures the default visual styling for matplotlib plots by
    updating rcParams. It applies a base style and allows for custom overrides.
    
    Args:
        style_params: Optional dictionary of style parameters to override defaults.
                     Common keys include:
                     - 'figure.figsize': Tuple of (width, height) in inches
                     - 'axes.facecolor': Background color for plot axes
                     - 'axes.grid': Boolean to enable/disable grid
                     
    Returns:
        None. Updates matplotlib's global rcParams.
        
    """
    final_style = DEFAULT_STYLE.copy()
    if style_params:
        final_style.update(style_params)
    
    # Update matplotlib rcParams
    plt.rcParams.update({
        'axes.facecolor': final_style['axes.facecolor'],
        'axes.grid': final_style['axes.grid'],
        'figure.figsize': final_style['figure.figsize']
    })
    
    # Use seaborn's modern API
    sns.set_theme(style='white', rc={'figure.figsize': final_style['figure.figsize']})

def get_cluster_palette(n_clusters: int, l: float = 0.4, s: float = 0.9) -> List[tuple]:
    """
    Generate a visually distinct color palette for cluster visualization.
    
    Creates evenly-spaced colors in HLS (Hue, Lightness, Saturation) color space
    to ensure maximum visual distinction between clusters.
    
    Args:
        n_clusters: Number of distinct clusters to generate colors for.
                   Must be a positive integer.
        l: Lightness parameter, controls how light or dark colors are.
           Range: 0.0 (black) to 1.0 (white). Default: 0.4 (medium-dark)
        s: Saturation parameter, controls color intensity/vividness.
           Range: 0.0 (grayscale) to 1.0 (fully saturated). Default: 0.9 (highly saturated)
        
    Returns:
        List of RGB tuples, each tuple containing (R, G, B) values in range [0, 1].
        Length of list equals n_clusters.
        
    Raises:
        ValueError: If n_clusters is less than 1
        ValueError: If l or s are outside [0, 1] range
    """
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    if not (0 <= l <= 1):
        raise ValueError(f"Lightness l must be in [0, 1], got {l}")
    if not (0 <= s <= 1):
        raise ValueError(f"Saturation s must be in [0, 1], got {s}")
    
    return sns.hls_palette(n_clusters, l=l, s=s)

def get_cluster_markers(n_clusters: int, base_markers: Optional[List[str]] = None) -> Dict[int, str]:
    """
    Generate a mapping of cluster IDs to matplotlib marker styles.
    
    Creates a dictionary mapping cluster labels to marker symbols, cycling through
    available markers if the number of clusters exceeds the base marker set.
    
    Args:
        n_clusters: Number of clusters to generate markers for.
                   Must be a positive integer.
        base_markers: Optional list of matplotlib marker symbol strings.
                     If None, uses a default set of 10 distinct markers.
                     Valid markers: 'o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X', etc.
                     See matplotlib.markers documentation for full list.
        
    Returns:
        Dictionary mapping cluster indices (0 to n_clusters-1) to marker symbols.
        Format: {0: 'o', 1: 's', 2: 'D', ...}
        
    Raises:
        ValueError: If n_clusters is less than 1
        ValueError: If base_markers is an empty list
    """
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    
    if base_markers is None:
        base_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X']
    
    if len(base_markers) == 0:
        raise ValueError("base_markers cannot be an empty list")
    
    # Create marker list by cycling through base markers
    markers_list = base_markers * ((n_clusters // len(base_markers)) + 1)
    markers_list = markers_list[:n_clusters]
    
    # Return as dictionary mapping cluster index to marker
    return {i: marker for i, marker in enumerate(markers_list)}
def plot_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "Cluster Visualization",
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
    show_legend: bool = True,
    palette: Optional[List[tuple]] = None,
    markers: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
    style_params: Optional[dict] = None,
    show_outliers: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Create a scatter plot visualization of clustered data with optional outliers.

    Visualizes 2D embeddings (e.g., from UMAP, t-SNE, or PCA) with colors and
    markers indicating cluster membership. Supports outlier detection where
    points labeled as -1 are treated as outliers.

    Args:
        embeddings: 2D numpy array of shape (n_samples, 2) containing the
                   coordinates of each point in 2D space. Typically from
                   dimensionality reduction techniques.
        labels: 1D numpy array of shape (n_samples,) containing cluster labels.
               Values should be integers where -1 indicates outliers and
               0, 1, 2, ... indicate cluster membership.
        title: Title text for the plot. Default: "Cluster Visualization"
        xlabel: Label for the x-axis. Default: "Dimension 1"
        ylabel: Label for the y-axis. Default: "Dimension 2"
        show_legend: Whether to display a legend showing cluster labels.
                    Default: True
        palette: Optional custom color palette as list of RGB tuples.
                If None, generates colors using get_cluster_palette().
        markers: Optional custom marker mapping as dictionary {label: marker}.
                If None, generates markers using get_cluster_markers().
        save_path: Optional file path to save the figure. Supports formats:
                  png, pdf, svg, jpg. If None, figure is not saved.
        style_params: Optional dictionary of style parameters to override
                     defaults. Passed to set_visualization_style().
        show_outliers: If True, includes outliers (points with label -1) in the plot.
                      If False, filters out outliers before plotting.
                      Default: True
        show: Whether to call plt.show() to display the plot immediately.
             Default: False. Set to True for interactive use.
        
    Returns:
        matplotlib.figure.Figure object containing the plot. Can be further
        customized or saved by the caller.
        
    Raises:
        ValueError: If embeddings is not 2D or has wrong shape
        ValueError: If embeddings and labels have different lengths
        ValueError: If embeddings contains NaN or infinite values
        ValueError: If no points remain after filtering outliers
        
    """
    # Input validation
    if embeddings.ndim != 2 or embeddings.shape[1] != 2:
        raise ValueError(f"embeddings must be 2D array with shape (n, 2), got {embeddings.shape}")
    if len(embeddings) != len(labels):
        raise ValueError(f"embeddings and labels must have same length, got {len(embeddings)} and {len(labels)}")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("embeddings contains NaN or infinite values")
    
    set_visualization_style(style_params)
    
    # Filter outliers if requested
    if not show_outliers:
        mask = labels != -1
        if not np.any(mask):
            raise ValueError("No points remain after filtering outliers")
        embeddings = embeddings[mask]
        labels = labels[mask]
    
    # Get unique clusters
    unique_clusters = np.unique(labels)
    
    # Generate or validate palette and markers
    if palette is None:
        n_clusters = len(unique_clusters) if -1 not in unique_clusters else len(unique_clusters) - 1
        palette = get_cluster_palette(n_clusters)
        # Add gray color for outliers if they exist and we're showing them
        if show_outliers and -1 in unique_clusters:
            palette = list(palette) + [(0.5, 0.5, 0.5)]  # Add gray for outliers
    
    if markers is None:
        # Use only filled markers to avoid seaborn's mixed marker restriction
        filled_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'H', '8']
        markers = {}
        for i, label in enumerate(unique_clusters):
            if label == -1:
                # For outliers, use a filled marker that stands out
                markers[-1] = 'D'  # Diamond marker for outliers
            else:
                # Get the index for regular clusters (skip -1)
                cluster_idx = list(unique_clusters).index(label)
                if label == -1:
                    actual_idx = cluster_idx - 1
                else:
                    actual_idx = cluster_idx
                markers[label] = filled_markers[actual_idx % len(filled_markers)]
    
    # Create a single figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each cluster separately
    for i, cluster_label in enumerate(unique_clusters):
        mask = labels == cluster_label
        if np.any(mask):
            cluster_embeddings = embeddings[mask]
            
            # Determine color
            if cluster_label == -1:
                # Use gray color for outliers
                color = (0.5, 0.5, 0.5)
                label_name = 'Outliers'
            else:
                # Get color from palette based on cluster index
                color_idx = i if cluster_label != -1 else len(palette) - 1
                color = palette[color_idx % len(palette)]
                label_name = f'Cluster {cluster_label}'
            
            ax.scatter(
                x=cluster_embeddings[:, 0],
                y=cluster_embeddings[:, 1],
                c=[color],
                marker=markers[cluster_label],
                label=label_name,
                s=60,  # marker size
                alpha=0.7
            )
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig
def plot_clusters_with_annotations(
    embeddings: np.ndarray,
    labels: np.ndarray,
    annotations: np.ndarray,
    title: str = "Cluster Visualization with Annotations",
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
    show_legend: bool = True,
    palette: Optional[List[tuple]] = None,
    markers: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
    style_params: Optional[dict] = None,
    filter_outliers: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Visualize clusters with additional annotation labels (e.g., ground truth categories).
    
    Creates a scatter plot where points are colored and shaped according to
    annotation labels rather than cluster assignments. Useful for comparing
    clustering results against known categories or ground truth labels.
    
    Args:
        embeddings: 2D numpy array of shape (n_samples, 2) containing point
                   coordinates in 2D embedding space.
        labels: 1D numpy array of shape (n_samples,) containing cluster labels
               from clustering algorithm. Used only for filtering outliers if
               filter_outliers=True. Values of -1 indicate outliers.
        annotations: 1D numpy array of shape (n_samples,) containing category
                    labels for each point. Can be integers or strings.
                    These labels determine point colors and markers.
        title: Title text for the plot.
              Default: "Cluster Visualization with Annotations"
        xlabel: Label for the x-axis. Default: "Dimension 1"
        ylabel: Label for the y-axis. Default: "Dimension 2"
        show_legend: Whether to display legend showing annotation categories.
                    Default: True
        palette: Optional custom color palette as list of RGB tuples.
                If None, generates colors based on number of unique annotations.
        markers: Optional custom marker mapping as dictionary {annotation: marker}.
                If None, generates markers automatically.
        save_path: Optional file path to save the figure. Supports png, pdf, svg, jpg.
        style_params: Optional dictionary to override default style settings.
        filter_outliers: If True, removes points where labels == -1 before plotting.
                        Useful to exclude outliers from visualization.
                        Default: True
        show: Whether to call plt.show() to display the plot immediately.
             Default: False
        
    Returns:
        matplotlib.figure.Figure object containing the plot.
        
    Raises:
        ValueError: If arrays have mismatched shapes
        ValueError: If embeddings is not 2D or contains invalid values
        ValueError: If no points remain after filtering outliers
    """
    # Input validation
    if embeddings.ndim != 2 or embeddings.shape[1] != 2:
        raise ValueError(f"embeddings must be 2D array with shape (n, 2), got {embeddings.shape}")
    if not (len(embeddings) == len(labels) == len(annotations)):
        raise ValueError(f"All arrays must have same length, got {len(embeddings)}, {len(labels)}, {len(annotations)}")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("embeddings contains NaN or infinite values")
    
    set_visualization_style(style_params)
    
    # Filter outliers if requested
    if filter_outliers:
        mask = labels != -1
        if not np.any(mask):
            raise ValueError("No points remain after filtering outliers")
        embeddings = embeddings[mask]
        labels = labels[mask]
        annotations = annotations[mask]
    
    # Get unique annotation categories
    unique_annotations = np.unique(annotations)
    n_categories = len(unique_annotations)
    
    # Generate or validate palette and markers
    if palette is None:
        palette = get_cluster_palette(n_categories)
    if markers is None:
        markers = get_cluster_markers(n_categories)
        # Map annotation values to markers
        markers = {ann: markers[i] for i, ann in enumerate(unique_annotations)}
    
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        hue=annotations,
        style=annotations,
        markers=markers,
        legend=show_legend,
        palette=palette,
        ax=ax
    )
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig

def plot_cluster_comparison(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    annotation_labels: np.ndarray,
    title: str = "Cluster vs Annotation Comparison",
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Create side-by-side comparison of clustering results and annotation labels.
    
    Generates a figure with two subplots: left shows cluster assignments from
    an algorithm, right shows ground truth or other annotation labels. This
    enables visual comparison of how well clustering matches expected categories.
    
    Args:
        embeddings: 2D numpy array of shape (n_samples, 2) containing point
                   coordinates in embedding space (e.g., from UMAP or t-SNE).
        cluster_labels: 1D numpy array of shape (n_samples,) containing cluster
                       assignments from clustering algorithm. Integer values
                       where -1 indicates outliers.
        annotation_labels: 1D numpy array of shape (n_samples,) containing
                          reference labels (e.g., ground truth categories, 
                          manual annotations, or alternative clustering).
                          Can be integers or strings.
        title: Overall title for the entire figure.
              Default: "Cluster vs Annotation Comparison"
        save_path: Optional file path to save the figure.
                  Supports formats: png, pdf, svg, jpg.
        show: Whether to call plt.show() to display the plot immediately.
             Default: False
        
    Returns:
        matplotlib.figure.Figure object containing both subplots.
        Figure size is automatically set to (20, 9) for side-by-side layout.
        
    Raises:
        ValueError: If arrays have mismatched shapes
        ValueError: If embeddings is not 2D or contains invalid values
    """
    # Input validation
    if embeddings.ndim != 2 or embeddings.shape[1] != 2:
        raise ValueError(f"embeddings must be 2D array with shape (n, 2), got {embeddings.shape}")
    if not (len(embeddings) == len(cluster_labels) == len(annotation_labels)):
        raise ValueError(f"All arrays must have same length, got {len(embeddings)}, {len(cluster_labels)}, {len(annotation_labels)}")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("embeddings contains NaN or infinite values")
    
    set_visualization_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle(title, fontsize=18)
    
    # Left subplot: Cluster assignments
    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
    n_clusters = len(unique_clusters)
    cluster_palette = get_cluster_palette(n_clusters)
    cluster_markers = get_cluster_markers(n_clusters)
    cluster_markers = {label: cluster_markers[i] for i, label in enumerate(unique_clusters)}
    if -1 in cluster_labels:
        cluster_markers[-1] = 'x'
    
    sns.scatterplot(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        hue=cluster_labels,
        style=cluster_labels,
        markers=cluster_markers,
        palette=cluster_palette,
        legend=False,
        ax=ax1
    )
    ax1.set_title("Cluster Assignments", fontsize=16)
    ax1.set_xlabel("Dimension 1", fontsize=14)
    ax1.set_ylabel("Dimension 2", fontsize=14)
    
    # Right subplot: Annotation labels
    unique_annotations = np.unique(annotation_labels)
    n_categories = len(unique_annotations)
    annotation_palette = get_cluster_palette(n_categories)
    annotation_markers = get_cluster_markers(n_categories)
    annotation_markers = {label: annotation_markers[i] for i, label in enumerate(unique_annotations)}
    
    sns.scatterplot(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        hue=annotation_labels,
        style=annotation_labels,
        markers=annotation_markers,
        palette=annotation_palette,
        legend=True,
        ax=ax2
    )
    ax2.set_title("Annotation Labels", fontsize=16)
    ax2.set_xlabel("Dimension 1", fontsize=14)
    ax2.set_ylabel("", fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig